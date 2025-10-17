# app.py (Full English UI, Percentage Target, Login, and Base Only Scenario)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os
import io

# Suppress warnings for smooth execution
warnings.filterwarnings('ignore')

# Import core functions and BQ client from the refactored module
# CHÚ Ý: Đảm bảo file renewal.py (hoặc renewal2.py) đã được đặt tên chính xác
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

# --- CONFIGURATION & CONSTANTS ---
APP_ID_LIST = ['6736361418', '6503937276', '6738117098'] 
BQ_CLIENT = client

# --- STREAMLIT AUTHENTICATION LOGIC (RESTORED) ---

def check_password():
    """Returns True if the user enters the correct password (from st.secrets)."""
    
    if st.session_state.get("password_correct", False):
        return True

    if "credentials" not in st.secrets:
        valid_users = {"admin": "admin_pass"} 
        st.warning("⚠️ Using dummy credentials for local run. Configure 'secrets.toml' for deployment.")
    else:
        credentials = st.secrets["credentials"]
        valid_users = {
            credentials.get("username_admin"): credentials.get("password_admin"), 
            credentials.get("username_viewer"): credentials.get("password_viewer"),
        }
        valid_users = {k: v for k, v in valid_users.items() if k and v}
        if not valid_users:
             st.error("Error: Login credentials in secrets.toml are empty.")
             st.stop()


    def authenticate_user():
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        if username in valid_users and valid_users[username] == password: 
            st.session_state["password_correct"] = True
            st.session_state["username_logged"] = username
            st.rerun()
        else:
            st.error("❌ Incorrect username or password.")

    st.title("Login")
    with st.form("login_form"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.form_submit_button("Log in", on_click=authenticate_user)
        
    return False


# --- CACHED DATA LOADERS ---

@st.cache_data(show_spinner="Loading Daily Data from BigQuery...")
def load_all_daily_data(_bq_client, app_id_list):
    """Loads daily data (Cost, Revenue) from BigQuery."""
    
    if _bq_client is None:
        try:
            # Fallback to local file for testing if BQ client fails
            df_all = pd.read_csv("data/raw_data/daily_data.csv")
            df_all['date'] = pd.to_datetime(df_all['date']).dt.date
            df_all['app_id'] = df_all['App_ID'].astype(str)
            df_all['revenue'] = df_all['Revenue']
            df_all['cost'] = df_all['Cost']
            df_all['app_name'] = df_all['App_Name']
            df_all['profit'] = df_all['revenue'] - df_all['cost']
            return df_all[df_all['app_id'].isin(app_id_list)].copy()
        except:
            st.error("Error: BigQuery client not initialized and local fallback failed.")
            return pd.DataFrame()
            
    app_ids_str = ', '.join([f"'{aid}'" for aid in app_id_list])
    query_daily = f"""
        SELECT 
            date, 
            app_id, 
            app_name, 
            cost, 
            revenue, 
            (revenue - cost) AS profit
        FROM `{BQ_DAILY_TABLE_PATH}` 
        WHERE app_id IN ({app_ids_str})
    """
    
    try:
        df = _bq_client.query(query_daily).to_dataframe()
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['app_id'] = df['app_id'].astype(str)
        df = df.sort_values('date')
        return df
    
    except Exception as e:
        st.error(f"BIGQUERY QUERY EXECUTION ERROR (Daily Data): {e}. Could not load data.")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Processing App Data and Models...") 
def prepare_app_data(df_all_data, df_renewal_raw, selected_app_id):
    """
    Loads, filters, processes data, and runs Prophet/Renewal models for the selected app.
    """
    
    # 1. Filter Data
    df = df_all_data[df_all_data['app_id'] == selected_app_id].sort_values('date').reset_index(drop=True)
    if df.empty:
        raise ValueError("Historical data for selected app is empty.")

    df['roas'] = df.apply(
        lambda row: row['revenue'] / row['cost'] if row['cost'] > 0 else 0.0, axis=1
    )
    
    # 2. Calculate Key Variables
    df['cumulative_profit'] = df['profit'].cumsum() 
    current_loss = df['cumulative_profit'].iloc[-1]
    last_history_date = df['date'].iloc[-1]
    cost_hist_last = df['cost'].iloc[-1] 
    
    # 3. Calculate ROAS Baseline (7-Day Rolling)
    df_7_days = df.tail(7)
    roas_baseline = df_7_days['revenue'].sum() / df_7_days['cost'].sum() if df_7_days['cost'].sum() > 0 else 1.0
    if np.isnan(roas_baseline) or np.isinf(roas_baseline): roas_baseline = 1.0
    
    roas_hist_last = df['roas'].iloc[-1] if not df['roas'].iloc[-1] < 0.01 else roas_baseline
    roas_hist_last = roas_hist_last if not np.isnan(roas_hist_last) else roas_baseline

    # 4. Run Prophet & Renewal Models
    df_history_prophet_input = df[['date', 'cost']].copy() 
    df_renewal_f = get_renewal_forecast_data(df_renewal_raw, selected_app_id)
    df_propensity = get_prophet_cost_forecast(df_history_prophet_input) 

    # 5. Default Cost
    default_cost_day_1 = df_propensity['cost_prophet_f'].iloc[0] if not df_propensity.empty and df_propensity['cost_prophet_f'].iloc[0] > 0 else 1000

    return (df, current_loss, last_history_date, cost_hist_last, roas_hist_last, 
            df_propensity, df_renewal_f, roas_baseline, default_cost_day_1, df_7_days)


# --- MAIN STREAMLIT APPLICATION ---

if not check_password():
    st.stop() 

st.set_page_config(layout="wide", page_title="Breakeven", page_icon="💰")
st.title("Breakeven Optimization Dashboard")

# ----------------- DATA LOADING -----------------

df_all_daily = load_all_daily_data(BQ_CLIENT, APP_ID_LIST)

query_renewal = f"""
    SELECT First_Date, App_Id, App_Name, Product_ID, Price_USD, Renew_Times, Total_Renew, Cohort_Buy, Revenue_Renew, SubsStart_TrialConvert
    FROM `{BQ_RENEWAL_TABLE_PATH}` 
"""
df_renewal_raw = load_bigquery_renewal_data(BQ_CLIENT, query_renewal)


if df_all_daily.empty or df_renewal_raw.empty:
    st.error("Insufficient historical data from BigQuery to run the model.")
    st.stop()

# --- SIDEBAR: CONFIGURATION AND INPUTS (ENGLISH UI) ---

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
        df_propensity, df_renewal_f, roas_baseline, default_cost_day_1, df_7_days) = \
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

    target_days_input = st.number_input(
        "Target Breakeven Days:",
        min_value=1,
        value=180,
        step=30,
        disabled=(selected_mode == 'Days'),
        help="Maximum or target days to achieve breakeven."
    )

    col_a, col_b = st.columns(2)

    target_roas_input = col_a.number_input(
        "Target ROAS:",
        min_value=0.01,
        value=max(roas_baseline, 0.1),
        step=0.05, format="%.2f",
        disabled=(selected_mode == 'ROAS'),
        help=f"7-Day Baseline ROAS: {roas_baseline:,.2f}"
    )

    target_cost_day_1_input = col_b.number_input(
        "Day 1 Cost Target ($):",
        min_value=0.01,
        value=default_cost_day_1,
        step=100.0, format="%.2f",
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

    # 7-Day Rolling Metrics
    avg_cost_7 = df_7_days['cost'].mean()
    avg_rev_7 = df_7_days['revenue'].mean()
    avg_roas_7 = roas_baseline 

    col_m1.metric("Current Cumulative Profit", f"{current_loss:,.2f} $",
                  delta=f"Data up to {last_history_date.strftime('%Y-%m-%d')}", delta_color="inverse")
    col_m2.metric("7-Day Avg. Cost", f"{avg_cost_7:,.2f} $", help="Average daily Cost over the last 7 days.")
    col_m3.metric("7-Day Avg. Revenue", f"{avg_rev_7:,.2f} $", help="Average daily Revenue over the last 7 days.")
    col_m4.metric("7-Day Rolling ROAS", f"{avg_roas_7:,.2f}", help="Revenue / Cost over the last 7 days.")

st.markdown("---")

# ----------------- SOLVER EXECUTION -----------------
is_error_case = False
error_msg = None
final_roas = target_roas_input
final_cost = target_cost_day_1_input
final_days = target_days_input
forecast_data = pd.DataFrame()
forecast_period_data = pd.DataFrame()

# -------------------- 1. RUN OPTIMIZED FORECAST --------------------

if df_propensity.empty:
    st.error("Error: Cost Prophet model failed. Calculation stopped.")
    is_error_case = True
else:
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
if not is_error_case and final_days <= FORECAST_HORIZON_DAYS:
    breakeven_ts = history_end_ts + pd.to_timedelta(final_days, unit='D')
    breakeven_date = breakeven_ts.date() 
    
    # Plot up to the optimized breakeven date + 30 days, or max horizon
    optimized_plot_end_ts = min(breakeven_ts + pd.to_timedelta(30, unit='D'), 
                                df_propensity['ds'].max()) # Max date from prophet timeline
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
# The plot should extend to the *later* of the two breakeven points (+ margin)
# or the forecast horizon, whichever comes first.

overall_max_plot_ds = df_propensity['ds'].max() # Default to max forecast horizon
if breakeven_ts:
    overall_max_plot_ds = max(overall_max_plot_ds, breakeven_ts + pd.to_timedelta(30, unit='D'))
if base_only_breakeven_ts:
    overall_max_plot_ds = max(overall_max_plot_ds, base_only_breakeven_ts + pd.to_timedelta(30, unit='D'))

# Slice both forecast dataframes to this common end point
forecast_data_plot = forecast_data[forecast_data['ds'] <= overall_max_plot_ds].copy()
base_only_data_plot = base_only_data[base_only_data['ds'] <= overall_max_plot_ds].copy()


# Standardize date column for Plotly
forecast_data_plot['date'] = pd.to_datetime(forecast_data_plot['ds'])
base_only_data_plot['date'] = pd.to_datetime(base_only_data_plot['ds'])


# ----------------- 2. STRATEGY SUMMARY & RESULTS -----------------

if not is_error_case and not forecast_period_data.empty:
    
    avg_total_rev_forecast = forecast_period_data['total_revenue_f'].mean()
    avg_cost_forecast = forecast_period_data['cost_f'].mean()
    avg_profit_forecast = avg_total_rev_forecast - avg_cost_forecast
    
    # total_estimated_base_revenue = df_renewal_f.attrs.get('total_estimated_base_revenue', 0.0) # Ẩn đi

    with st.container():
        st.markdown("#### Optimized Strategy Results vs. Stop Spend")
        col_res_a, col_res_b, col_res_c = st.columns(3) # Đã giảm số cột

        # Display solved variable
        if selected_mode == 'ROAS':
            var_name = "Required ROAS"
            var_value = f"{final_roas:,.2f}"
        elif selected_mode == 'Cost':
            var_name = "Required Day 1 Cost"
            var_value = f"{final_cost:,.0f} $"
        else:
            var_name = "Days to Recovery (Optimized)"
            var_value = f"{final_days} days"
        
        # Optimized Breakeven/Recovery Day
        opt_days_text = f"{final_days} days"
        opt_delta = f"Est. Date: {breakeven_date.strftime('%Y-%m-%d')}" if breakeven_date else "Not Reached"
        
        # Base Only Breakeven/Recovery Day
        base_days_text = f"{base_only_days} days"
        base_delta = f"Est. Date: {base_only_breakeven_date.strftime('%Y-%m-%d')}" if base_only_breakeven_date else "Not Reached"

        # Col A: Days to Recovery Comparison
        col_res_a.metric("Days to Recovery (Optimized)", opt_days_text, delta=opt_delta)
        col_res_b.metric("Days to Recovery (Stop Spend)", base_days_text, delta=base_delta, delta_color='off')
        
        # Col C: Other Metrics
        col_res_c.metric("Avg. Net Profit / Day (Optimized)", 
                         f"{avg_profit_forecast:,.2f} $",
                         help=f"Average Daily Net Profit (Base + Acquisition) during the {final_days} days.")
        
        # Không hiển thị Total Base Revenue (LTV) ở đây nữa
        # col_res_d.metric("Total Base Revenue (LTV)",
        #                  f"{total_estimated_base_revenue:,.0f} $",
        #                  help="Total estimated Renewal Revenue from existing cohorts until decay reduces to 1%.")
        
    st.markdown("---")

elif is_error_case:
    st.error(f"❌ Conclusion (Optimized Scenario): {error_msg}")
    
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
    fig.add_trace(go.Scatter(x=forecast_data_plot['date'], y=forecast_data_plot['cumulative_profit'], mode='lines',
                             name=f'Forecast (Opt. ROAS {final_roas:,.2f}, Cost {final_cost:,.0f} $)', 
                             line=dict(color='#1f77b4', width=3, dash='dash')))

    # 3. Base Only (Stop Spend) Forecast
    fig.add_trace(go.Scatter(x=base_only_data_plot['date'], y=base_only_data_plot['cumulative_profit'], mode='lines',
                             name=f'Base Only (Stop Spend)', 
                             line=dict(color='#2ca02c', width=3, dash='dot'))) # Green dot

    # 4. Recovery line
    # Mở rộng đường Recovery đến cuối trục x của dữ liệu dự báo được vẽ
    fig.add_shape(type="line", x0=df_history_plot['date'].min(), y0=recovery_threshold, x1=overall_max_plot_ds, y1=recovery_threshold,
                  line=dict(color="Red", width=2, dash="dot"), name=f'Recovery Threshold ({recovery_target_percent}%)')

    # 5. Breakeven Markers
    if breakeven_ts:
        fig.add_trace(go.Scatter(x=[breakeven_ts], y=[recovery_threshold], mode='markers',
                                 name=f'Est. Recovery (Optimized)', 
                                 marker=dict(size=12, color='Blue', symbol='circle')))
    
    if base_only_breakeven_ts:
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
    df_daily_forecast_opt = forecast_data_plot[['date', 'cost_f', 'total_revenue_f']].copy()
    df_daily_forecast_opt = df_daily_forecast_opt.rename(columns={'cost_f': 'cost_opt', 'total_revenue_f': 'total_revenue_opt'})

    # Forecast (Base Only)
    df_daily_forecast_base = base_only_data_plot[['date', 'cost_f', 'total_revenue_f']].copy()
    df_daily_forecast_base = df_daily_forecast_base.rename(columns={'cost_f': 'cost_base', 'total_revenue_f': 'total_revenue_base'})

    # Combine Cost
    combined_cost = pd.concat([
        df_daily_hist[['date', 'cost']].set_index('date').rename(columns={'cost': 'historical_cost'}),
        df_daily_forecast_opt[['date', 'cost_opt']].set_index('date'),
        df_daily_forecast_base[['date', 'cost_base']].set_index('date')
    ], axis=1).reset_index()

    # Combine Revenue
    combined_rev = pd.concat([
        df_daily_hist[['date', 'total_revenue']].set_index('date').rename(columns={'total_revenue': 'historical_revenue'}),
        df_daily_forecast_opt[['date', 'total_revenue_opt']].set_index('date'),
        df_daily_forecast_base[['date', 'total_revenue_base']].set_index('date')
    ], axis=1).reset_index()
    
    # Fill NaN for historical parts in forecast columns 
    combined_cost = combined_cost.fillna(value={'cost_opt': 0, 'cost_base': 0})
    combined_rev = combined_rev.fillna(value={'total_revenue_opt': 0, 'total_revenue_base': 0})


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
    fig_cr.add_trace(go.Scatter(
        x=combined_cost['date'], y=combined_cost['cost_opt'],
        mode='lines', name='Forecasted Cost (Optimized)', line=dict(color='#2ECC71', width=3, dash='dash')
    ))
    fig_cr.add_trace(go.Scatter(
        x=combined_rev['date'], y=combined_rev['total_revenue_opt'],
        mode='lines', name='Forecasted Revenue (Optimized)', line=dict(color='#3498DB', width=3, dash='dash')
    ))

    # 3) Base Only Forecast (Revenue = Base Only)
    fig_cr.add_trace(go.Scatter(
        x=combined_rev['date'], y=combined_rev['total_revenue_base'],
        mode='lines', name='Revenue (Base Only)', line=dict(color='darkorange', width=3, dash='dot')
    ))


    # "End of History" Line
    fig_cr.add_shape(
        type="line",
        x0=history_end_ts.to_pydatetime(), x1=history_end_ts.to_pydatetime(), y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="gray", width=2, dash="dot")
    )

    # Breakeven Marker Line (Optimized)
    if breakeven_ts:
        be_dt = breakeven_ts.to_pydatetime()
        fig_cr.add_shape(type="line", x0=be_dt, x1=be_dt, y0=0, y1=1, xref="x", yref="paper",
            line=dict(color="blue", width=2, dash="dash"))
        fig_cr.add_annotation(x=be_dt, y=1.05, xref="x", yref="paper",
            text=f"Opt. Recovery ({breakeven_ts.strftime('%Y-%m-%d')})",
            showarrow=False, font=dict(size=12, color="blue"))

    # Breakeven Marker Line (Base Only)
    if base_only_breakeven_ts:
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

# ----------------- 4. DEEP DIVE (Detailed Summary Table) -----------------

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
                "--- FORECAST PERFORMANCE ---", 
                "Avg. Net Profit / Day (Recovery Period - Optimized)",
                "Avg. Base Revenue / Day (Renewal)", 
                "Total Estimated Base Revenue (LTV)"
            ],
            'Value':   [''] * 16, 
            'Notes':   [''] * 16 
        }
        summary_df = pd.DataFrame(summary_data).set_index('Metric')

        # Gán giá trị
        summary_df.loc["History End Date", 'Value'] = last_history_date.strftime('%Y-%m-%d')
        summary_df.loc["Current Cumulative Profit", 'Value'] = f"{current_loss:,.2f} $"
        summary_df.loc["Last History Day Cost", 'Value'] = f"{cost_hist_last:,.2f} $"
        summary_df.loc["Last History Day ROAS", 'Value'] = f"{roas_hist_last:,.2f}" 
        summary_df.loc["7-Day Rolling ROAS Baseline", 'Value'] = f"{roas_baseline:,.2f}"
        
        summary_df.loc["Final ROAS (Solved/Fixed)", 'Value'] = f"{final_roas:,.2f}"
        summary_df.loc["Day 1 Cost (Solved/Fixed)", 'Value'] = f"{final_cost:,.2f} $"
        summary_df.loc["Days to Recovery (Optimized)", 'Value'] = f"{final_days}"
        summary_df.loc["Recovery Profit Threshold", 'Value'] = f"{recovery_threshold:,.2f} $"
        
        summary_df.loc["Days to Recovery (Stop Spend)", 'Value'] = f"{base_only_days}"

        summary_df.loc["Avg. Net Profit / Day (Recovery Period - Optimized)", 'Value'] = f"{avg_profit_forecast:,.2f} $"
        summary_df.loc["Avg. Base Revenue / Day (Renewal)", 'Value'] = f"{forecast_period_data['base_revenue_f'].mean():,.2f} $"
        
        # Total Estimated Base Revenue (LTV) vẫn hiển thị trong phần chi tiết này
        total_estimated_base_revenue = df_renewal_f.attrs.get('total_estimated_base_revenue', 0.0)
        summary_df.loc["Total Estimated Base Revenue (LTV)", 'Value'] = f"{total_estimated_base_revenue:,.2f} $"
        
        # Update Notes
        summary_df.loc["Final ROAS (Solved/Fixed)", 'Notes'] = "Solved for this metric." if selected_mode == 'ROAS' else "Fixed input for calculation."
        summary_df.loc["Day 1 Cost (Solved/Fixed)", 'Notes'] = "Solved for this metric." if selected_mode == 'Cost' else "Fixed input for calculation."
        summary_df.loc["Days to Recovery (Optimized)", 'Notes'] = "Solved for this metric." if selected_mode == 'Days' else "Fixed input for calculation."
        summary_df.loc["Recovery Profit Threshold", 'Notes'] = f"Cumulative profit required to meet the {recovery_target_percent}% recovery target."
        summary_df.loc["Days to Recovery (Stop Spend)", 'Notes'] = "Time to reach threshold with Cost = 0 and Revenue = Base Only."

        summary_df.loc["Avg. Net Profit / Day (Recovery Period - Optimized)", 'Notes'] = "Includes both Base (Renewal) and Acquisition Revenue."
        summary_df.loc["Total Estimated Base Revenue (LTV)", 'Notes'] = "LTV estimate from existing cohorts until decay reduces to 1% retention."

        # Clear section headers
        for metric in summary_df.index:
            if "---" in metric:
                 summary_df.loc[metric, ['Value', 'Notes']] = ["", ""]
        
        st.dataframe(summary_df, use_container_width=True)
    
    else:
        st.warning("No detailed summary available due to solver error or failure to meet the recovery target.")