import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os
import pickle
import datetime as dt
warnings.filterwarnings('ignore')

from renewal import (
    load_bigquery_renewal_data,
    get_renewal_forecast_data,
    get_prophet_cost_forecast,
    solve_breakeven_logic,
    calculate_cumulative_profit,
    calculate_cumulative_profit_base_only, 
    client, 
    BQ_RENEWAL_TABLE_PATH,
    BQ_DAILY_TABLE_PATH, 
    FORECAST_HORIZON_DAYS
)

# CONFIGURATION
APP_ID_LIST = ['6736361418', '6503937276', '6738117098', 'ai.generated.art.photo']
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
BQ_CLIENT = client


# ==============================
# DAILY CACHE HELPERS
# ==============================
def get_today_cache_path():
    today_str = dt.date.today().strftime("%Y%m%d")
    return os.path.join(CACHE_DIR, f"forecast_cache_{today_str}.pkl")

def load_daily_cache():
    path = get_today_cache_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_daily_cache(data):
    path = get_today_cache_path()
    with open(path, "wb") as f:
        pickle.dump(data, f)


# AUTO REFRESH TRIGGER (for GitHub Action or ?refresh=1)
query_params = st.query_params
if query_params and "refresh" in query_params:
    try:
        os.remove(get_today_cache_path())
        st.warning("♻️ Cache cleared by scheduler (auto-refresh triggered). Rebuilding now...")
        st.stop()
    except Exception:
        pass


# ==============================
# LOAD FORECAST DAILY
# ==============================
@st.cache_resource(ttl=86400, show_spinner="⏳ Loading or building daily Prophet/Renewal cache...")
def load_forecast_daily():
    cache = load_daily_cache()
    if cache:
        return cache

    def load_all_daily_data(_bq_client, app_id_list):
        app_ids_str = ", ".join([f"'{aid}'" for aid in app_id_list])
        query = f"""
            SELECT date, app_id, app_name, cost, revenue, (revenue - cost) AS profit
            FROM `{BQ_DAILY_TABLE_PATH}`
            WHERE app_id IN ({app_ids_str}) and date <= date_sub(current_date(), interval 2 day)
        """
        df = _bq_client.query(query).to_dataframe()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["app_id"] = df["app_id"].astype(str)
        df = df.sort_values("date")
        return df

    df_all_daily = load_all_daily_data(BQ_CLIENT, APP_ID_LIST)

    query_renewal = f"""
        SELECT First_Date, App_Id, App_Name, Product_ID, Price_USD, Renew_Times,
               Total_Renew, Cohort_Buy, Revenue_Renew, SubsStart_TrialConvert
        FROM `{BQ_RENEWAL_TABLE_PATH}`
    """
    df_renewal_raw = load_bigquery_renewal_data(BQ_CLIENT, query_renewal)

    forecast_dict = {}
    for app_id in APP_ID_LIST:
        df_prop = get_prophet_cost_forecast(
            df_all_daily[df_all_daily["app_id"] == app_id][["date", "cost"]]
        )
        df_renewal = get_renewal_forecast_data(df_renewal_raw, app_id)
        forecast_dict[app_id] = (df_prop, df_renewal)

    data = {
        "df_all_daily": df_all_daily,
        "df_renewal_raw": df_renewal_raw,
        "forecast_dict": forecast_dict,
    }
    save_daily_cache(data)
    return data


# ==============================
# LOGIN
# ==============================
def check_password():
    if st.session_state.get("password_correct", False):
        return True

    if "credentials" not in st.secrets:
        valid_users = {"admin": "admin_pass"}
        st.warning("⚠️ Dummy login (use secrets.toml for production).")
    else:
        credentials = st.secrets["credentials"]
        valid_users = {
            credentials.get("username_admin"): credentials.get("password_admin"),
            credentials.get("password_admin"): credentials.get("password_admin"),
            credentials.get("username_viewer"): credentials.get("password_viewer"),
        }
        valid_users = {k: v for k, v in valid_users.items() if k and v}

    st.title("Login: Breakeven Optimization")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if username in valid_users and valid_users[username] == password:
            st.session_state["password_correct"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Wrong username or password")
            return False
    return False


# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(layout="wide", page_title="Breakeven Dashboard", page_icon="💰")

if not check_password():
    st.stop()

st.title("Breakeven Optimization")

# Admin tools
is_admin = st.session_state.get("username") == "admin"
if is_admin:
    if st.sidebar.button("🔄 Force Refresh Now"):
        try:
            os.remove(get_today_cache_path())
        except FileNotFoundError:
            pass
        st.experimental_rerun()

# ==============================
# LOAD CACHED DATA
# ==============================
data_cache = load_forecast_daily()
df_all_daily = data_cache["df_all_daily"]
df_renewal_raw = data_cache["df_renewal_raw"]
forecast_dict = data_cache["forecast_dict"]

if df_all_daily.empty or df_renewal_raw.empty:
    st.error("❌ No data available from BigQuery.")
    st.stop()

st.caption("Dashboard auto-refresh every 24h")

# ==============================
# PREPARE APP DATA
# ==============================
@st.cache_data(ttl=86400, show_spinner="Processing App Data and Models...")
def prepare_app_data(df_all_data, df_renewal_raw, selected_app_id):
    """
    Loads, filters, processes data, and runs Prophet/Renewal models for the selected app.
    Added quality checks for zero spend & abnormal ROAS.
    """
    df = df_all_data[df_all_data['app_id'] == selected_app_id].sort_values('date').reset_index(drop=True)
    if df.empty:
        raise ValueError("Historical data for selected app is empty.")

    df['roas'] = df.apply(lambda row: row['revenue'] / row['cost'] if row['cost'] > 0 else 0.0, axis=1)
    df['cumulative_profit'] = df['profit'].cumsum() 
    current_loss = df['cumulative_profit'].iloc[-1]
    last_history_date = df['date'].iloc[-1]
    cost_hist_last = df['cost'].iloc[-1] 

    # 7-day rolling baseline
    df_7_days = df.tail(7)
    roas_baseline = df_7_days['revenue'].sum() / df_7_days['cost'].sum() if df_7_days['cost'].sum() > 0 else 1.0
    if np.isnan(roas_baseline) or np.isinf(roas_baseline):
        roas_baseline = 1.0

    # --- QUALITY CHECK LOGIC ---
    zero_spend_days = (df_7_days['cost'] <= 0).sum()
    df_last_30 = df.tail(30)
    roas_last_30 = df_last_30.apply(lambda r: r['revenue'] / r['cost'] if r['cost'] > 0 else np.nan, axis=1).dropna()
    roas_median_30 = roas_last_30.median() if not roas_last_30.empty else 1.0
    roas_abnormal = roas_baseline > (3 * roas_median_30)
    is_quality_issue = (zero_spend_days > 0) or roas_abnormal

    # (Thông tin tham khảo thêm cho UI/metric 7 ngày)
    df_7_days = df_7_days[df_7_days['cost'] > 0]  # remove zero spend days from last-7 calc view

    roas_hist_last = df['roas'].iloc[-1] if not df['roas'].iloc[-1] < 0.01 else roas_baseline
    roas_hist_last = roas_hist_last if not np.isnan(roas_hist_last) else roas_baseline

    # 🔒 Prophet input: remove zero spend days before training
    df_history_prophet_input = df[df['cost'] > 0][['date', 'cost']].copy()

    df_renewal_f = get_renewal_forecast_data(df_renewal_raw, selected_app_id)
    df_propensity = get_prophet_cost_forecast(df_history_prophet_input) 

    default_cost_day_1 = df_propensity['cost_prophet_f'].iloc[0] if not df_propensity.empty and df_propensity['cost_prophet_f'].iloc[0] > 0 else 1000

    return (
        df, current_loss, last_history_date, cost_hist_last, roas_hist_last, 
        df_propensity, df_renewal_f, roas_baseline, default_cost_day_1, df_7_days, is_quality_issue
    )


# ==============================
# SIDEBAR SETTINGS
# ==============================
with st.sidebar:
    st.header("🛠️ Settings")

    # 1. APPLICATION SELECTION
    st.subheader("1. Application Selection")

    df_valid_apps = df_all_daily[df_all_daily['app_id'].isin(APP_ID_LIST)]
    app_name_mapping = df_valid_apps[['app_id', 'app_name']].drop_duplicates().set_index('app_id').to_dict().get('app_name', {})

    app_options_valid = {app_name_mapping.get(aid, f"App ID {aid}"): aid for aid in APP_ID_LIST if aid in app_name_mapping}
    app_names = list(app_options_valid.keys())

    selected_app_name = st.selectbox("Select App:", app_names, index=0)
    selected_app_id = app_options_valid[selected_app_name]

    try:
        (df, current_loss, last_history_date, cost_hist_last, roas_hist_last, 
        df_propensity, df_renewal_f, roas_baseline, default_cost_day_1, df_7_days, is_quality_issue) = \
            prepare_app_data(df_all_daily, df_renewal_raw, selected_app_id)
    except ValueError as e:
        st.error(f"Data Preparation Error: {str(e)}")
        st.stop()

    st.markdown("---")

    # 2. OPTIMIZATION PARAMETERS (SOLVER)
    st.subheader("2. Optimization Parameters")

    mode_options = {
        "Find Days (Fixed ROAS & Cost)": "Days",
        "Find Required ROAS (Fixed Cost & Days)": "ROAS",
        "Find Required Day 1 Cost (Fixed ROAS & Days)": "Cost",
    }
    selected_mode_label = st.radio("Choose Variable to Solve for:", list(mode_options.keys()))
    selected_mode = mode_options[selected_mode_label]


    st.markdown("---")
    st.subheader("3. Recovery Target")
    
    # RECOVERY TARGET (PERCENTAGE SLIDER)
    recovery_target_percent = st.slider(
        "Recovery Target (%)",
        min_value=0, 
        max_value=100, 
        value=100, # Default to 100% (Full Breakeven)
        step=10,
        key="recovery_target_slider",
        help="100% means full breakeven (Cumulative Profit >= 0). 0% means just stopping further loss."
    )
    
    # Calculate Threshold based on Percentage
    recovery_threshold = current_loss * (1 - (recovery_target_percent / 100)) if current_loss < 0 else 0.0
    if current_loss >= 0 and recovery_target_percent < 100: 
        recovery_threshold = 0.0

    st.info(f"Required Profit Threshold: **{recovery_threshold:,.2f} $**")
    st.markdown(f"*(Current Cumulative Profit: {current_loss:,.2f} $)*")


    # --- FIXED INPUTS ---
    st.markdown("##### Fixed Input Values")

    def safe_float(x, default=0.0):
        try:
            v = float(x)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except Exception:
            return default

    def safe_int(x, default=0):
        try:
            return int(float(x))
        except Exception:
            return default

    # ----- DAYS: tất cả là int -----
    target_days_input = st.number_input(
        "Target Breakeven Days:",
        min_value=int(1),
        value=int(180),
        step=int(30),
        disabled=(selected_mode == 'Days'),
        help="Maximum or target days to achieve breakeven."
    )

    col_a, col_b = st.columns(2)

    # ----- ROAS: tất cả là float -----
    roas_val = max(safe_float(roas_baseline, 1.0), 0.1)
    target_roas_input = col_a.number_input(
        "Target ROAS:",
        min_value=0.01,                 # float
        value=float(roas_val),          # float
        step=0.05,                      # float
        format="%.2f",
        disabled=(selected_mode == 'ROAS'),
        help=f"7-Day Baseline ROAS: {roas_baseline:,.2f}"
    )

    # ----- COST: tất cả là float -----
    cost_default = max(safe_float(default_cost_day_1, 0.01), 0.01)
    target_cost_day_1_input = col_b.number_input(
        "Day 1 Cost Target ($):",
        min_value=0.01,                 # float
        value=float(cost_default),      # float
        step=100.0,                     # float
        format="%.2f",
        disabled=(selected_mode == 'Cost'),
        help="Target Cost for the first day of the forecast period."
    )

# ----------------- MAIN CONTENT: PERFORMANCE METRICS -----------------

st.subheader(f"Strategy Dashboard for: **{selected_app_name}**")
st.markdown("---")

# 1. PERFORMANCE SNAPSHOT (Current State)
with st.container():
    st.markdown("#### Current Performance Snapshot (7-Day Rolling)")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    # 7-Day Rolling Metrics (chỉ tính trên df_7_days đã loại zero spend ở phần prepare)
    avg_cost_7 = df_7_days['cost'].mean() if not df_7_days.empty else 0.0
    avg_rev_7 = df_7_days['revenue'].mean() if not df_7_days.empty else 0.0
    avg_roas_7 = (df_7_days['revenue'].sum() / df_7_days['cost'].sum()) if not df_7_days.empty and df_7_days['cost'].sum() > 0 else roas_baseline

    col_m1.metric("Current Cumulative Profit", f"{current_loss:,.2f} $",
                  delta=f"Data up to {last_history_date.strftime('%Y-%m-%d')}", delta_color="inverse")
    col_m2.metric("7-Day Avg. Cost", f"{avg_cost_7:,.2f} $", help="Average daily Cost over the last 7 days (non-zero spend days).")
    col_m3.metric("7-Day Avg. Revenue", f"{avg_rev_7:,.2f} $", help="Average daily Revenue over the last 7 days (non-zero spend days).")
    col_m4.metric("7-Day Rolling ROAS", f"{avg_roas_7:,.2f}", help="Revenue / Cost over the last 7 days (non-zero spend days).")

st.markdown("---")

# ==============================
# MAIN EXECUTION (QUALITY GATE)
# ==============================
if df_propensity.empty:
    st.error("Error: Cost Prophet model failed. Calculation stopped.")
    is_error_case = True
    error_msg = "Prophet model failed (Empty or invalid historical data)."
else:
    is_error_case = False
    error_msg = None

# Nếu có bất thường 7 ngày (zero spend hoặc ROAS đột biến) → chỉ hiển thị base forecast
# (Dùng cờ từ prepare_app_data)
if is_quality_issue:
    st.warning("⚠️ Detected zero spend or abnormal ROAS in last 7 days. Displaying Renewal Forecast (LTV) only.")
    
    renew_data, renew_days = calculate_cumulative_profit_base_only(
        df_propensity[['ds']].copy(),
        df_renewal_f,
        current_loss,
        recovery_threshold
    )
    
    # Xác định ngày hoàn vốn nếu có
    renew_breakeven_date = None
    if not renew_data.empty:
        reached = renew_data[renew_data['cumulative_profit'] >= recovery_threshold]
        if not reached.empty:
            renew_breakeven_date = pd.to_datetime(reached.iloc[0]['ds']).date()
    
    # --- Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=renew_data['ds'], y=renew_data['cumulative_profit'],
        mode='lines', name='Renew Forecast (LTV)', line=dict(color='green', width=3)
    ))
    
    # Vẽ đường ngưỡng hoàn vốn
    fig.add_shape(
        type="line",
        x0=renew_data['ds'].min(), x1=renew_data['ds'].max(),
        y0=recovery_threshold, y1=recovery_threshold,
        line=dict(color="red", width=2, dash="dot")
    )
    
    # Đánh dấu ngày hoàn vốn nếu có
    if renew_breakeven_date:
        fig.add_trace(go.Scatter(
            x=[renew_breakeven_date], y=[recovery_threshold],
            mode='markers+text',
            name=f"Recovery ({renew_breakeven_date})",
            text=[renew_breakeven_date.strftime('%Y-%m-%d')],
            textposition="top center",
            marker=dict(size=10, color='red', symbol='star')
        ))
    
    fig.update_layout(
        title="Cumulative Profit (Renew Forecast Only)",
        xaxis_title="Date",
        yaxis_title="Cumulative Profit ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Summary text ---
    if renew_breakeven_date:
        st.success(f"✅ Estimated recovery date (Renew Forecast): **{renew_breakeven_date.strftime('%Y-%m-%d')}**")
    else:
        st.info("ℹ️ No recovery reached within the forecast horizon (Renew Forecast).")
    
    st.stop()

# ----------------- SOLVER EXECUTION -----------------
is_error_case = False if not is_error_case else True
final_roas = target_roas_input
final_cost = target_cost_day_1_input
final_days = target_days_input
forecast_data = pd.DataFrame()
forecast_period_data = pd.DataFrame()
error_msg = None

# -------------------- 1. RUN OPTIMIZED FORECAST --------------------
if not is_error_case:
    try:
        if selected_mode == 'ROAS':
            forecast_data, solved_roas, final_cost, final_days, error_msg = solve_breakeven_logic(
                df_propensity, df_renewal_f, current_loss, recovery_threshold,
                mode='ROAS', target_value=target_days_input, fixed_cost_day_1=target_cost_day_1_input,
                cost_hist_last=cost_hist_last, roas_hist_last=roas_hist_last
            )
            final_roas = solved_roas
        elif selected_mode == 'Cost':
            forecast_data, final_roas, solved_cost, final_days, error_msg = solve_breakeven_logic(
                df_propensity, df_renewal_f, current_loss, recovery_threshold,
                mode='Cost', target_value=target_days_input, fixed_roas=target_roas_input,
                cost_hist_last=cost_hist_last, roas_hist_last=roas_hist_last
            )
            final_cost = solved_cost
        else: # selected_mode == 'Days'
            forecast_data, final_roas, final_cost, final_days, error_msg = solve_breakeven_logic(
                df_propensity, df_renewal_f, current_loss, recovery_threshold,
                mode='Days', fixed_roas=target_roas_input, fixed_cost_day_1=target_cost_day_1_input,
                cost_hist_last=cost_hist_last, roas_hist_last=roas_hist_last
            )
        
        if error_msg:
            is_error_case = True

    except Exception as e:
        st.error(f"❌ ITERATIVE SOLVER ERROR (Optimized): {e}")
        is_error_case = True
        error_msg = str(e)

# -------------------- 2. RUN BASE ONLY (STOP SPEND) FORECAST --------------------
base_only_data, base_only_days = calculate_cumulative_profit_base_only(
    df_propensity[['ds']].copy(), 
    df_renewal_f, 
    current_loss, 
    recovery_threshold
)
base_only_data_plot = base_only_data.copy()

# ----------------- BREAKEVEN DATE + DATA SLICING -----------------
history_end_ts = pd.to_datetime(df['date'].iloc[-1])
breakeven_ts = None
breakeven_date = None
base_only_breakeven_ts = None
base_only_breakeven_date = None

# Optimized Scenario Slicing
if not is_error_case and not forecast_data.empty and final_days <= FORECAST_HORIZON_DAYS:
    breakeven_ts = history_end_ts + pd.to_timedelta(final_days, unit='D')
    breakeven_date = breakeven_ts.date() 
    
    # Plot up to the optimized breakeven date + 30 days, or max horizon
    optimized_plot_end_ts = min(breakeven_ts + pd.to_timedelta(30, unit='D'), 
                                pd.to_datetime(df_propensity['ds'].max())) # Max date from prophet timeline
    forecast_data_plot = forecast_data[forecast_data['ds'] <= optimized_plot_end_ts].copy()
    forecast_period_data = forecast_data[forecast_data['ds'] <= breakeven_ts].copy()
else:
    # Fallback plot data (show up to target_days + 30 or max horizon)
    max_plot_days_fallback = min(int(target_days_input if selected_mode != 'Days' else 180) + 30, FORECAST_HORIZON_DAYS) 
    forecast_data_plot = forecast_data.head(max_plot_days_fallback).copy()
    forecast_period_data = forecast_data_plot # Use this for summary if error

# Base Only Scenario Slicing
if base_only_days <= FORECAST_HORIZON_DAYS:
    base_only_breakeven_ts = history_end_ts + pd.to_timedelta(base_only_days, unit='D')
    base_only_breakeven_date = base_only_breakeven_ts.date()

# Determine the overall max plot date for BOTH scenarios
overall_max_plot_ds = pd.to_datetime(df_propensity['ds'].max()) # Default to max forecast horizon
if breakeven_ts is not None:
    overall_max_plot_ds = max(overall_max_plot_ds, breakeven_ts + pd.to_timedelta(30, unit='D'))
if base_only_breakeven_ts is not None:
    overall_max_plot_ds = max(overall_max_plot_ds, base_only_breakeven_ts + pd.to_timedelta(30, unit='D'))

# Slice both forecast dataframes to this common end point
forecast_data_plot = forecast_data_plot[forecast_data_plot['ds'] <= overall_max_plot_ds].copy()
base_only_data_plot = base_only_data_plot[base_only_data_plot['ds'] <= overall_max_plot_ds].copy()

# Standardize date column for Plotly
forecast_data_plot['date'] = pd.to_datetime(forecast_data_plot['ds'])
base_only_data_plot['date'] = pd.to_datetime(base_only_data_plot['ds'])

# ----------------- 2. STRATEGY SUMMARY & RESULTS (ĐÃ CẬP NHẬT) -----------------
if not is_error_case and not forecast_period_data.empty:
    
    # Tính toán các chỉ số trung bình
    avg_profit_forecast = forecast_period_data['profit_f'].mean()
    avg_cost_forecast = forecast_period_data['cost_f'].mean()
    avg_base_rev_forecast = forecast_period_data['base_revenue_f'].mean()
    avg_acq_rev_forecast = forecast_period_data['rev_from_cost_f'].mean()

    total_estimated_base_revenue = df_renewal_f.attrs.get('total_estimated_base_revenue', 0.0)

    with st.container():
        st.markdown("#### Optimized Strategy Results vs. Stop Spend")
        col_res_a, col_res_b, col_res_c = st.columns(3) 

        # 1. Variable Solved/Fixed (Days to Recovery)
        days_text = f"{final_days} days"
        days_delta = "Solved Result" if selected_mode == 'Days' else "Fixed Input"
        col_res_a.metric("Days to Recovery (Optimized)", days_text, delta=days_delta)
        
        # 2. Variable Solved/Fixed (Target ROAS)
        roas_text = f"{final_roas:,.2f}"
        roas_delta = "Solved Result" if selected_mode == 'ROAS' else "Fixed Input"
        col_res_b.metric("Target ROAS", roas_text, delta=roas_delta, delta_color='off')
        
        # 3. Variable Solved/Fixed (Day 1 Cost)
        cost_text = f"{final_cost:,.0f} $"
        cost_delta = "Solved Result" if selected_mode == 'Cost' else "Fixed Input"
        col_res_c.metric("Day 1 Cost Target", cost_text, delta=cost_delta, delta_color='off')
        
        
    st.markdown("---")
    
    # HIỂN THỊ SO SÁNH BASE ONLY VÀ NET PROFIT
    with st.container():
        col_comp_a, col_comp_b, col_comp_c = st.columns(3)
        
        # Days to Recovery (Stop Spend)
        base_days_text = f"{base_only_days} days"
        base_delta = f"Est. Date: {base_only_breakeven_date.strftime('%Y-%m-%d')}" if base_only_breakeven_date else "Not Reached"
        col_comp_a.metric("Days to Recovery (Stop Spend)", base_days_text, delta=base_delta, delta_color='off')
        
        # Avg. Net Profit
        col_comp_b.metric("Avg. Net Profit / Day (Optimized)", 
                         f"{avg_profit_forecast:,.2f} $",
                         help=f"Average Daily Net Profit (Base + Acquisition) during the {final_days} days.")
        
        # Base Revenue LTV
        col_comp_c.metric("Total Estimated Base Revenue (LTV)",
                         f"{total_estimated_base_revenue:,.2f} $",
                         help="LTV estimate from existing cohorts until decay reduces to 1% retention.")

    st.markdown("---")

elif is_error_case:
    st.error(f"❌ Conclusion (Optimized Scenario): {error_msg if error_msg else 'Unknown error'}")
    
    # Still show Base Only comparison
    with st.container():
        st.markdown("#### Stop Spend Scenario Comparison")
        base_days_text = f"{base_only_days} days"
        base_delta = f"Est. Date: {base_only_breakeven_date.strftime('%Y-%m-%d')}" if base_only_breakeven_date else "Not Reached"
        st.metric("Days to Recovery (Stop Spend)", base_days_text, delta=base_delta, delta_color='off')
        
    st.markdown("---")

# ----------------- 3. VISUALIZATION TABS -----------------

tab1, tab2 = st.tabs(["Chart 1: Cumulative Profit", "Chart 2: Daily P&L (Cost & Revenue)"])

# ------------------ TAB 1: CUMULATIVE PROFIT CHART ------------------
with tab1:
    
    # 1. Historical Data
    df_history_plot = df[['date', 'cumulative_profit']].copy()
    
    fig = go.Figure()
    
    # 1. Historical
    fig.add_trace(go.Scatter(x=df_history_plot['date'], y=df_history_plot['cumulative_profit'], mode='lines',
                             name='Historical Cumulative Profit', line=dict(color='#ff7f0e', width=3)))
    
    # 2. Optimized Forecast
    if not forecast_data_plot.empty:
        fig.add_trace(go.Scatter(x=forecast_data_plot['date'], y=forecast_data_plot['cumulative_profit'], mode='lines',
                                 name=f'Forecast (Opt. ROAS {final_roas:,.2f}, Cost {final_cost:,.0f} $)', 
                                 line=dict(color='#1f77b4', width=3, dash='dash')))

    # 3. Base Only (Stop Spend) Forecast
    if not base_only_data_plot.empty:
        fig.add_trace(go.Scatter(x=base_only_data_plot['date'], y=base_only_data_plot['cumulative_profit'], mode='lines',
                                 name=f'Base Only (Stop Spend)', 
                                 line=dict(color='#2ca02c', width=3, dash='dot'))) # Green dot

    # 4. Recovery line
    fig.add_shape(type="line", x0=df_history_plot['date'].min(), y0=recovery_threshold, x1=overall_max_plot_ds, y1=recovery_threshold,
                  line=dict(color="Red", width=2, dash="dot"), name=f'Recovery Threshold ({recovery_target_percent}%)')
    # 5. Breakeven Markers
    if breakeven_ts is not None:
        fig.add_trace(go.Scatter(x=[breakeven_ts], y=[recovery_threshold], mode='markers',
                                 name=f'Est. Recovery (Optimized)', 
                                 marker=dict(size=12, color='Blue', symbol='circle')))
    
    if base_only_breakeven_ts is not None:
        fig.add_trace(go.Scatter(x=[base_only_breakeven_ts], y=[recovery_threshold], mode='markers',
                                 name=f'Est. Recovery (Base Only)', 
                                 marker=dict(size=12, color='Green', symbol='star')))


    fig.update_layout(title={'text': f'Cumulative Profit (Recovery Target: {recovery_target_percent}%)',
                             'x': 0.5, 'xanchor': 'center'},
                      xaxis_title="Date", yaxis_title="Cumulative Profit ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TAB 2: DAILY COST & REVENUE CHART ------------------
with tab2:
    # Historical
    df_daily_hist = df[['date', 'cost', 'revenue']].copy()
    df_daily_hist = df_daily_hist.rename(columns={'revenue': 'total_revenue'})

    # Forecast (Optimized)
    df_daily_forecast_opt = forecast_data_plot[['date', 'cost_f', 'total_revenue_f']].copy() if not forecast_data_plot.empty else pd.DataFrame(columns=['date','cost_f','total_revenue_f'])
    df_daily_forecast_opt = df_daily_forecast_opt.rename(columns={'cost_f': 'cost_opt', 'total_revenue_f': 'total_revenue_opt'})

    # Forecast (Base Only)
    df_daily_forecast_base = base_only_data_plot[['date', 'cost_f', 'total_revenue_f']].copy() if not base_only_data_plot.empty else pd.DataFrame(columns=['date','cost_f','total_revenue_f'])
    df_daily_forecast_base = df_daily_forecast_base.rename(columns={'cost_f': 'cost_base', 'total_revenue_f': 'total_revenue_base'})

    # ✅ Fix dtype mismatch before comparison
    last_actual_date = pd.to_datetime(df_daily_hist['date'].max())
    if not df_daily_forecast_opt.empty:
        df_daily_forecast_opt['date'] = pd.to_datetime(df_daily_forecast_opt['date'])
    if not df_daily_forecast_base.empty:
        df_daily_forecast_base['date'] = pd.to_datetime(df_daily_forecast_base['date'])

    # Filter forecast to show only after last actual date
    if not df_daily_forecast_opt.empty:
        df_daily_forecast_opt = df_daily_forecast_opt[df_daily_forecast_opt['date'] > last_actual_date]
    if not df_daily_forecast_base.empty:
        df_daily_forecast_base = df_daily_forecast_base[df_daily_forecast_base['date'] > last_actual_date]

    # Combine Cost
    combined_cost = pd.concat([
        df_daily_hist[['date', 'cost']].set_index('date').rename(columns={'cost': 'historical_cost'}),
        df_daily_forecast_opt[['date', 'cost_opt']].set_index('date') if not df_daily_forecast_opt.empty else pd.DataFrame(),
        df_daily_forecast_base[['date', 'cost_base']].set_index('date') if not df_daily_forecast_base.empty else pd.DataFrame()
    ], axis=1).reset_index()

    # Combine Revenue
    combined_rev = pd.concat([
        df_daily_hist[['date', 'total_revenue']].set_index('date').rename(columns={'total_revenue': 'historical_revenue'}),
        df_daily_forecast_opt[['date', 'total_revenue_opt']].set_index('date') if not df_daily_forecast_opt.empty else pd.DataFrame(),
        df_daily_forecast_base[['date', 'total_revenue_base']].set_index('date') if not df_daily_forecast_base.empty else pd.DataFrame()
    ], axis=1).reset_index()

    # Fill missing values only where needed
    if 'cost_opt' in combined_cost.columns and 'cost_base' in combined_cost.columns:
        combined_cost[['cost_opt', 'cost_base']] = combined_cost[['cost_opt', 'cost_base']].fillna(0)
    elif 'cost_opt' in combined_cost.columns:
        combined_cost[['cost_opt']] = combined_cost[['cost_opt']].fillna(0)
    elif 'cost_base' in combined_cost.columns:
        combined_cost[['cost_base']] = combined_cost[['cost_base']].fillna(0)

    if 'total_revenue_opt' in combined_rev.columns and 'total_revenue_base' in combined_rev.columns:
        combined_rev[['total_revenue_opt', 'total_revenue_base']] = combined_rev[['total_revenue_opt', 'total_revenue_base']].fillna(0)
    elif 'total_revenue_opt' in combined_rev.columns:
        combined_rev[['total_revenue_opt']] = combined_rev[['total_revenue_opt']].fillna(0)
    elif 'total_revenue_base' in combined_rev.columns:
        combined_rev[['total_revenue_base']] = combined_rev[['total_revenue_base']].fillna(0)

    # --- PLOT ---
    fig_cr = go.Figure()

    # 1) Historical
    fig_cr.add_trace(go.Scatter(
        x=combined_cost['date'], y=combined_cost['historical_cost'],
        mode='lines', name='Historical Cost', line=dict(color='#2ECC71', width=2)
    ))
    fig_cr.add_trace(go.Scatter(
        x=combined_rev['date'], y=combined_rev['historical_revenue'],
        mode='lines', name='Historical Revenue', line=dict(color='#3498DB', width=2)
    ))

    # 2) Optimized Forecast
    if 'cost_opt' in combined_cost.columns:
        fig_cr.add_trace(go.Scatter(
            x=combined_cost['date'], y=combined_cost['cost_opt'],
            mode='lines', name='Forecasted Cost (Optimized)', line=dict(color='#2ECC71', width=3, dash='dash')
        ))
    if 'total_revenue_opt' in combined_rev.columns:
        fig_cr.add_trace(go.Scatter(
            x=combined_rev['date'], y=combined_rev['total_revenue_opt'],
            mode='lines', name='Forecasted Revenue (Optimized)', line=dict(color='#3498DB', width=3, dash='dash')
        ))

    # 3) Base Only Forecast
    if 'total_revenue_base' in combined_rev.columns:
        fig_cr.add_trace(go.Scatter(
            x=combined_rev['date'], y=combined_rev['total_revenue_base'],
            mode='lines', name='Revenue (Base Only)', line=dict(color='darkorange', width=3, dash='dot')
        ))

    # --- Vertical Lines ---
    # End of History
    fig_cr.add_shape(
        type="line",
        x0=history_end_ts.to_pydatetime(), x1=history_end_ts.to_pydatetime(), y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="gray", width=2, dash="dot")
    )

    # Optimized Breakeven
    if breakeven_ts is not None:
        be_dt = breakeven_ts.to_pydatetime()
        fig_cr.add_shape(type="line", x0=be_dt, x1=be_dt, y0=0, y1=1, xref="x", yref="paper",
            line=dict(color="blue", width=2, dash="dash"))
        fig_cr.add_annotation(x=be_dt, y=1.05, xref="x", yref="paper",
            text=f"Opt. Recovery ({breakeven_ts.strftime('%Y-%m-%d')})",
            showarrow=False, font=dict(size=12, color="blue"))

    # Base-Only Breakeven
    if base_only_breakeven_ts is not None:
        be_dt_base = base_only_breakeven_ts.to_pydatetime()
        fig_cr.add_shape(type="line", x0=be_dt_base, x1=be_dt_base, y0=0, y1=1, xref="x", yref="paper",
            line=dict(color="green", width=2, dash="dot"))
        fig_cr.add_annotation(x=be_dt_base, y=1.00, xref="x", yref="paper",
            text=f"Base Rec. ({base_only_breakeven_ts.strftime('%Y-%m-%d')})",
            showarrow=False, font=dict(size=12, color="green"))

    fig_cr.update_layout(
        title={'text': 'Daily Cost and Total Revenue: History and Forecast Comparison',
               'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Amount ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_cr, use_container_width=True)

st.markdown("---")
# Thay thế toàn bộ khối st.expander bằng code này
with st.expander("Detailed Strategy Summary", expanded=False):
    if not is_error_case and not forecast_period_data.empty:
        summary_data = {
            'Metric': [
                "History End Date", 
                "Current Cumulative Profit", 
                "Last History Day Cost", 
                "Last History Day ROAS", 
                "7-Day Rolling ROAS Baseline",
                "--- OPTIMIZED SOLVER PARAMETERS ---", 
                "Final ROAS (Solved/Fixed)",
                "Day 1 Cost (Solved/Fixed)",
                "Days to Recovery (Optimized)",
                "Recovery Profit Threshold",
                "--- BASE ONLY SCENARIO ---", 
                "Days to Recovery (Stop Spend)",
                "--- FORECAST PERFORMANCE (Recovery Period) ---", 
                "Avg. Net Profit / Day",
                "Avg. Acquisition Revenue / Day (from Cost)",
                "Avg. Cost / Day",
                "Avg. Base Revenue / Day (Renewal)", 
                "Total Estimated Base Revenue (LTV)"
            ],
            'Value':   [''] * 18, 
            'Notes':   [''] * 18 
        }
        summary_df = pd.DataFrame(summary_data).set_index('Metric')
        # Gán giá trị
        summary_df.loc["History End Date", 'Value'] = last_history_date.strftime('%Y-%m-%d')
        summary_df.loc["Current Cumulative Profit", 'Value'] = f"{current_loss:,.2f} $"
        summary_df.loc["Last History Day Cost", 'Value'] = f"{cost_hist_last:,.2f} $"
        summary_df.loc["Last History Day ROAS", 'Value'] = f"{roas_hist_last:,.2f}" 
        summary_df.loc["7-Day Rolling ROAS Baseline", 'Value'] = f"{roas_baseline:,.2f}"
        
        # Gán giá trị đã giải/cố định
        summary_df.loc["Final ROAS (Solved/Fixed)", 'Value'] = f"{final_roas:,.2f}"
        summary_df.loc["Day 1 Cost (Solved/Fixed)", 'Value'] = f"{final_cost:,.2f} $"
        summary_df.loc["Days to Recovery (Optimized)", 'Value'] = f"{final_days}"
        
        summary_df.loc["Recovery Profit Threshold", 'Value'] = f"{recovery_threshold:,.2f} $"
        summary_df.loc["Days to Recovery (Stop Spend)", 'Value'] = f"{base_only_days}"
        # Gán các giá trị trung bình đã tính 
        summary_df.loc["Avg. Net Profit / Day", 'Value'] = f"{avg_profit_forecast:,.2f} $"
        summary_df.loc["Avg. Acquisition Revenue / Day (from Cost)", 'Value'] = f"{avg_acq_rev_forecast:,.2f} $"
        summary_df.loc["Avg. Cost / Day", 'Value'] = f"{avg_cost_forecast:,.2f} $"
        summary_df.loc["Avg. Base Revenue / Day (Renewal)", 'Value'] = f"{avg_base_rev_forecast:,.2f} $"
        summary_df.loc["Total Estimated Base Revenue (LTV)", 'Value'] = f"{total_estimated_base_revenue:,.2f} $"
        # Cập nhật Notes
        summary_df.loc["Final ROAS (Solved/Fixed)", 'Notes'] = "Solved for this metric." if selected_mode == 'ROAS' else "Fixed input for calculation."
        summary_df.loc["Day 1 Cost (Solved/Fixed)", 'Notes'] = "Solved for this metric." if selected_mode == 'Cost' else "Fixed input for calculation."
        summary_df.loc["Days to Recovery (Optimized)", 'Notes'] = "Solved for this metric." if selected_mode == 'Days' else "Fixed input for calculation."
        summary_df.loc["Recovery Profit Threshold", 'Notes'] = f"Cumulative profit required to meet the {recovery_target_percent}% recovery target."
        summary_df.loc["Days to Recovery (Stop Spend)", 'Notes'] = "Time to reach threshold with Cost = 0 and Revenue = Base Only."
        
        summary_df.loc["Avg. Net Profit / Day", 'Notes'] = "Net Profit = (Acq. Revenue - Cost) + Base Revenue."
        summary_df.loc["Avg. Acquisition Revenue / Day (from Cost)", 'Notes'] = "Avg. daily revenue from new users, driven by cost."
        summary_df.loc["Avg. Cost / Day", 'Notes'] = "Avg. daily ad spend during the recovery period."
        summary_df.loc["Avg. Base Revenue / Day (Renewal)", 'Notes'] = "Avg. daily revenue from existing users renewing."
        summary_df.loc["Total Estimated Base Revenue (LTV)", 'Notes'] = "LTV estimate from existing cohorts until decay reduces to 1% retention."
        # Clear section headers
        for metric in summary_df.index:
            if "---" in metric:
                 summary_df.loc[metric, ['Value', 'Notes']] = ["", ""]
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.warning("No detailed summary available due to solver error or failure to meet the recovery target.")
