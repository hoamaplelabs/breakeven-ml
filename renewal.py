# renewal.py

import pandas as pd
import numpy as np
import re
from prophet import Prophet
from datetime import datetime, timedelta, date
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
import warnings
import streamlit as st 
from google.cloud import bigquery
from google.oauth2 import service_account

# Suppress warnings for smooth execution
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Bỏ SERVICE_ACCOUNT_PATH vì dùng st.secrets
BQ_RENEWAL_TABLE_PATH = 'applesearchads-305307.ab_testing.revenue_renew'
BQ_DAILY_TABLE_PATH = 'applesearchads-305307.forecast.raw_data'

# --- MODEL LOGIC CONFIGURATION ---
RAMP_UP_DAYS = 14 
SMOOTHING_DAYS = 7 
FORECAST_HORIZON_DAYS = 1000

# --- BIGQUERY CLIENT INITIALIZATION ---
client = None
try:
    # 1. Tải credentials từ st.secrets
    if 'gcp_service_account' in st.secrets:
        credentials_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        
except Exception as e:
    # print(f"BIGQUERY CONNECTION ERROR: Could not initialize client. Error: {e}")
    client = None

# --- RENEWAL MODEL CLASS (Giữ nguyên) ---
class RenewalModel:
    """
    Sử dụng Decay Model dựa trên Cohort để dự báo Renewal Revenue (Base Revenue) 
    trong tương lai cho một ứng dụng cụ thể.
    """
    def __init__(self, df_renewal_raw, app_id):
        df_temp = df_renewal_raw[df_renewal_raw['App_Id'] == app_id].copy()
        self.df_raw = df_temp
        self.app_id = app_id
        self.models = {}
        
        if not self.df_raw.empty and 'First_Date' in self.df_raw.columns:
            self.last_history_date = self.df_raw['First_Date'].max()
        else:
            self.last_history_date = datetime.now().date() - timedelta(days=1) 
        
        self._preprocess_data()
        self._train_decay_models()

    def _categorize_subscription(self, product_id):
        if re.search(r'weekly|week', str(product_id), re.IGNORECASE):
            return 'weekly'
        elif re.search(r'monthly|month', str(product_id), re.IGNORECASE):
            return 'monthly'
        elif re.search(r'quarterly|quarter', str(product_id), re.IGNORECASE):
            return 'quarterly'
        elif re.search(r'yearly|year', str(product_id), re.IGNORECASE):
            return 'yearly'
        return 'unknown'

    def _preprocess_data(self):
        if self.df_raw.empty:
            self.df_all = pd.DataFrame()
            return
            
        self.df_raw['Subscription_Type'] = self.df_raw['Product_ID'].apply(self._categorize_subscription)
        self.df_all = self.df_raw[self.df_raw['Subscription_Type'].isin(['weekly', 'monthly', 'quarterly', 'yearly'])].copy()
        
        if self.df_all.empty: return
        
        self.df_all['Price_Per_Renew'] = self.df_all.apply(
            lambda row: row['Revenue_Renew'] / row['Total_Renew'] if row['Total_Renew'] > 0 else 0.0, axis=1
        )
        self.df_all = self.df_all[self.df_all['Cohort_Buy'].fillna(0) > 0].copy()
        
        if not self.df_all.empty:
            self.df_all['Retention_Percentage'] = (self.df_all['Total_Renew'] / self.df_all['Cohort_Buy']) * 100
            self.df_all = self.df_all[self.df_all['Retention_Percentage'] > 0].copy()
            
    def _estimate_lifespan(self, k, exponent, max_periods=1000, threshold=0.1):
        if k <= 0 or exponent <= 0:
            return 1 
            
        for t in range(1, max_periods + 1):
            retention_rate = (1 + k * t) ** (-exponent)
            if retention_rate < threshold:
                return t
        return max_periods

    def _train_decay_models(self):
        if self.df_all.empty: return

        def get_cohort_retention_data(group):
            group = group.sort_values('Renew_Times')
            early_data = group[group['Renew_Times'] <= 3]
            
            if len(early_data) >= 2:
                pct_change = early_data['Retention_Percentage'].pct_change()
                drops = -pct_change[pct_change.notna()] * 100
                avg_early_drop = drops.mean() if not drops.empty else 0.0
            else:
                avg_early_drop = 0.0

            avg_early_retention = early_data['Retention_Percentage'].mean() if not early_data.empty else 0.0
            
            trial_convert = group[group['Renew_Times'] == 1]['Total_Renew'].iloc[0] if 1 in group['Renew_Times'].values else 0
            
            return pd.Series({
                'Avg_Early_Drop': avg_early_drop,
                'Early_Retention': avg_early_retention,
                'Trial_Convert': trial_convert
            })

        cohort_features = self.df_all.groupby(['First_Date', 'Product_ID']).apply(get_cohort_retention_data).reset_index()
        self.df_all = self.df_all.merge(cohort_features, on=['First_Date', 'Product_ID'], how='left')
        
        if self.df_all.empty: return

        for product_id in self.df_all['Product_ID'].unique():
            product_data = self.df_all[self.df_all['Product_ID'] == product_id].copy()
            if product_data.empty: continue
            
            subscription_type = product_data['Subscription_Type'].iloc[0]
            cohort_buy = product_data['Cohort_Buy'].iloc[0] 

            max_trial_convert = product_data['Trial_Convert'].max()
            product_data['Trial_Weight'] = product_data['Trial_Convert'] / max_trial_convert if max_trial_convert > 0 else 1.0
            product_data['Trial_Weight'] = product_data['Trial_Weight'].fillna(0)
            
            drop_counts = product_data.groupby('First_Date')['Avg_Early_Drop'].first().apply(
                lambda x: 'strong' if x >= 65.0 else 'slow'
            ).value_counts()
            majority_behavior = drop_counts.idxmax() if not drop_counts.empty else 'slow'
            majority_proportion = drop_counts[majority_behavior] / drop_counts.sum() if not drop_counts.empty else 1.0
            product_data['Drop_Category'] = product_data['Avg_Early_Drop'].apply(lambda x: 'strong' if x >= 65.0 else 'slow')
            product_data['Stability_Weight'] = product_data['Drop_Category'].apply(
                lambda x: 1.0 if x == majority_behavior else 0.1 
            ) * majority_proportion
            product_data['Stability_Weight'] = product_data['Stability_Weight'].clip(lower=0.05).fillna(0.05)
            
            max_early_retention = product_data['Early_Retention'].max()
            if max_early_retention > 0:
                if majority_behavior == 'strong':
                    product_data['Early_Retention_Weight'] = (max_early_retention - product_data['Early_Retention']) / max_early_retention
                else:
                    product_data['Early_Retention_Weight'] = product_data['Early_Retention'] / max_early_retention
            else:
                product_data['Early_Retention_Weight'] = 1.0
            product_data['Early_Retention_Weight'] = product_data['Early_Retention_Weight'].clip(lower=0.01).fillna(0.01)

            product_data['Renewal_Weight'] = product_data['Total_Renew'] / product_data['Total_Renew'].max()
            product_data['Subs_Weight'] = product_data['Cohort_Buy'] / product_data['Cohort_Buy'].max()
            
            product_data['Weight'] = (product_data['Renewal_Weight'] *
                                      product_data['Subs_Weight'] *
                                      product_data['Trial_Weight'] *
                                      (product_data['Stability_Weight'] ** 2) *
                                      product_data['Early_Retention_Weight'])
            
            mask_early_renew = product_data['Renew_Times'] <= 2
            product_data.loc[mask_early_renew, 'Weight'] *= 5 

            mask_low_cohort = product_data['Cohort_Buy'] < 10
            product_data.loc[mask_low_cohort, 'Weight'] *= 0.1 
            
            product_data['Weight'] = product_data['Weight'].clip(lower=0.01).fillna(0.01)

            product_data['Log_Retention'] = np.log(product_data['Retention_Percentage'])
            renew_depth = product_data['Renew_Times'].max()

            filtered = product_data[
                (product_data['Retention_Percentage'] > 0.1) &
                (product_data['Weight'] > 0.01) &
                (product_data['Renew_Times'] <= renew_depth)
            ].copy()

            if len(filtered) < 3: continue

            X_log = filtered[['Renew_Times']].values
            y_log = filtered['Log_Retention'].values
            weights = filtered['Weight'].values

            model = LinearRegression()
            model.fit(X_log, y_log, sample_weight=weights)
            k = -model.coef_[0]

            MIN_K = 0.20 
            MAX_K = 0.7 

            if renew_depth < 3:
                k = np.clip(k, 0.25, MAX_K) 
            elif product_data['Avg_Early_Drop'].mean() > 60:
                k = np.clip(k, MIN_K, MAX_K)
            else:
                k = np.clip(k, MIN_K, 0.6) 

            if cohort_buy < 10 or renew_depth < 4:
                existing_ks = [m['k'] for m in self.models.values() if m['subscription_type'] == subscription_type]
                if existing_ks:
                    k_mean = np.mean(existing_ks)
                    k = 0.6 * k + 0.4 * k_mean 

            if subscription_type == 'weekly':
                existing_weekly = [m['k'] for m in self.models.values() if m['subscription_type'] == 'weekly']
                if existing_weekly:
                    k_median = np.median(existing_weekly)
                    k = 0.5 * k + 0.5 * k_median

            if subscription_type == 'weekly': renewal_interval = 7; exponent = 3.7
            elif subscription_type == 'monthly': renewal_interval = 30; exponent = 3.2 
            elif subscription_type == 'quarterly': renewal_interval = 90; exponent = 2.9
            else: renewal_interval = 365; exponent = 2.3

            if renew_depth < 4:
                exponent += 0.3

            self.models[product_id] = {
                'k': k,
                'price_per_renew': product_data['Price_Per_Renew'].iloc[0],
                'cohort_buy_avg': product_data['Cohort_Buy'].mean(),
                'renewal_interval': renewal_interval,
                'subscription_type': subscription_type,
                'exponent': exponent,
                'last_renew_time': product_data['Renew_Times'].max() 
            }

    def predict_renewal_revenue(self, max_days=FORECAST_HORIZON_DAYS):
        if not self.models:
            return pd.DataFrame({'date': [self.last_history_date + timedelta(days=i) for i in range(1, max_days + 1)], 
                                 'base_revenue_f': [0.0] * max_days})

        future_dates = pd.date_range(start=self.last_history_date + timedelta(days=1), periods=max_days, freq='D')
        df_forecast = pd.DataFrame({'date': future_dates.date, 'base_revenue_f': 0.0})
        
        total_estimated_base_revenue = 0.0 

        for product_id, model_info in self.models.items():
            
            product_cohorts = self.df_all[self.df_all['Product_ID'] == product_id].copy()
            
            max_renews = self._estimate_lifespan(model_info['k'], model_info['exponent'])
            price = model_info['price_per_renew']
            interval = model_info['renewal_interval']

            active_cohorts = product_cohorts.groupby('First_Date').agg({
                'Cohort_Buy': 'first',
                'Renew_Times': 'max',
            }).reset_index()
            
            active_cohorts = active_cohorts[active_cohorts['Cohort_Buy'] > 0]
            
            for _, cohort in active_cohorts.iterrows():
                
                initial_subs = cohort['Cohort_Buy']
                last_renew_times = cohort['Renew_Times']
                next_renew_time = last_renew_times + 1
                
                for t in range(next_renew_time, max_renews + 1):
                    
                    renew_date = cohort['First_Date'] + timedelta(days=interval * t)
                    
                    if renew_date <= self.last_history_date:
                        continue
                        
                    if renew_date > future_dates.max().date():
                        break
                        
                    retention_rate = max((1 + model_info['k'] * t) ** (-model_info['exponent']), 0)
                    
                    renewals = initial_subs * retention_rate
                    revenue = renewals * price
                    
                    total_estimated_base_revenue += revenue
                    
                    if renew_date in df_forecast['date'].values:
                         df_forecast.loc[df_forecast['date'] == renew_date, 'base_revenue_f'] += revenue
        
        df_forecast = df_forecast.sort_values('date')
        
        df_forecast.attrs['total_estimated_base_revenue'] = total_estimated_base_revenue
        
        return df_forecast.sort_values('date')

# --- CACHED DATA LOADERS (Giữ nguyên) ---

@st.cache_data(show_spinner="Loading Renewal Data from BigQuery...")
def load_bigquery_renewal_data(_bq_client, sql_query):
    if _bq_client is None:
        st.warning("BigQuery client is not initialized. Assuming zero renewal data.")
        return pd.DataFrame()
        
    try:
        df_renewal = _bq_client.query(sql_query).to_dataframe()
        df_renewal['First_Date'] = pd.to_datetime(df_renewal['First_Date']).dt.date
        df_renewal['App_Id'] = df_renewal['App_Id'].astype(str)
        return df_renewal
    
    except Exception as e:
        st.error(f"BIGQUERY QUERY EXECUTION ERROR: {e}. Returning empty renewal data.")
        return pd.DataFrame()

@st.cache_data(show_spinner="Running Decay LTV Model (Renewal Revenue)...")
def get_renewal_forecast_data(df_renewal_raw, app_id):
    try:
        renewal_model = RenewalModel(df_renewal_raw, app_id)
        df_renewal_f = renewal_model.predict_renewal_revenue()
        
        df_renewal_f['date'] = df_renewal_f['date'].apply(lambda x: x.date() if isinstance(x, datetime) else x)

        return df_renewal_f
    except Exception as e:
        st.warning(f"Could not run Renewal Model for App ID {app_id}. Base Revenue assumed to be $0. Error: {e}")
        df_err = pd.DataFrame({'date': [datetime.now().date() + timedelta(days=i) for i in range(1, FORECAST_HORIZON_DAYS + 1)], 
                               'base_revenue_f': [0.0] * FORECAST_HORIZON_DAYS})
        df_err.attrs['total_estimated_base_revenue'] = 0.0
        return df_err

@st.cache_data(show_spinner="Running Cost Prophet Model...")
def get_prophet_cost_forecast(df, max_days=FORECAST_HORIZON_DAYS):
    
    # BƯỚC 1: Xử lý df đầu vào và đổi tên cột an toàn
    # df_cost là DataFrame ĐÃ được đổi tên, df là DataFrame GỐC
    df_cost = df.copy()
    if 'date' in df_cost.columns:
        df_cost = df_cost.rename(columns={'date': 'ds'})
    if 'cost' in df_cost.columns:
        df_cost = df_cost.rename(columns={'cost': 'y'})
        
    df_cost = df_cost[['ds', 'y']] # Chỉ giữ lại 2 cột cần thiết
    
    # BƯỚC 2: Kiểm tra lỗi và xử lý
    if df_cost.empty or 'ds' not in df_cost.columns or 'y' not in df_cost.columns:
        # Trường hợp này không nên xảy ra nếu dữ liệu BigQuery đã tải đúng
        st.error("Prophet Model: Dữ liệu lịch sử không hợp lệ (thiếu ds hoặc y).")
        return pd.DataFrame()


    df_cost['y'] = df_cost['y'].rolling(3, min_periods=1).mean().fillna(df_cost['y'])
    
    m_cost = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.3)
    m_cost.fit(df_cost)
    
    future = m_cost.make_future_dataframe(periods=max_days)
    forecast_cost = m_cost.predict(future)
    
    # BƯỚC 3: FIX LỖI TRIỆT ĐỂ KeyERROR: 'ds'
    # last_date_history phải được lấy từ df_cost (đã đổi tên)
    last_date_history = df_cost['ds'].max() 
    
    df_future = forecast_cost[['ds', 'yhat']].rename(columns={'yhat': 'cost_prophet_f'})
    
    # FIX LỖI TRIỆT ĐỂ TypeError: Invalid comparison
    if isinstance(last_date_history, date) and not isinstance(last_date_history, datetime):
        last_date_history = pd.to_datetime(last_date_history) 
        
    df_future = df_future[df_future['ds'] > last_date_history].reset_index(drop=True)
    df_future['cost_prophet_f'] = np.maximum(0, df_future['cost_prophet_f'])

    if df_future.empty or df_future['cost_prophet_f'].iloc[0] <= 0:
        propensity_rate_base = 1.0
    else:
        propensity_rate_base = df_future['cost_prophet_f'].iloc[0]
        
    df_future['propensity_rate'] = df_future['cost_prophet_f'] / propensity_rate_base
    df_future['date'] = df_future['ds'].dt.date
    
    return df_future

# --- LOGIC TĂNG TRƯỞNG & LÀM MƯỢT (Giữ nguyên) ---

def apply_cost_smoothing(df, cost_hist_last, smoothing_days=SMOOTHING_DAYS):
    df_out = df.copy()
    max_days = len(df_out)
    
    cost_model_target = df_out['cost_f'].copy()
    
    df_out['smoothing_index'] = np.arange(1, max_days + 1)
    alpha = np.clip(df_out['smoothing_index'] / smoothing_days, 0, 1)
    mask = df_out['smoothing_index'] <= smoothing_days
    df_out.loc[mask, 'cost_f'] = (1 - alpha[mask]) * cost_hist_last + alpha[mask] * cost_model_target[mask]
    
    return df_out.drop(columns=['smoothing_index']).copy()

def apply_ramp_up(df, target_roas, roas_hist_last, ramp_up_days=RAMP_UP_DAYS):
    df_out = df.copy()
    df_out['day_index'] = np.arange(1, len(df_out) + 1)
    ramp_factor = np.minimum(1.0, df_out['day_index'] / ramp_up_days)
    df_out['roas_day'] = roas_hist_last + (target_roas - roas_hist_last) * ramp_factor
    df_out['rev_from_cost_f'] = df_out['cost_f'] * df_out['roas_day']
    
    return df_out.drop(columns=['day_index', 'roas_day']).copy()


def calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, 
                               target_roas, initial_cost_day_1, recovery_threshold, 
                               cost_hist_last, roas_hist_last):
    # Đã sửa: Sử dụng profit_f, cumulative_profit
    df_temp = df_propensity.copy()
    
    df_temp['cost_f'] = initial_cost_day_1 * df_temp['propensity_rate']
    
    df_temp = pd.merge(df_temp, df_renewal_f[['date', 'base_revenue_f']], on='date', how='left')
    df_temp['base_revenue_f'] = df_temp['base_revenue_f'].fillna(0.0)

    df_temp = apply_cost_smoothing(df_temp, cost_hist_last) 

    df_temp = apply_ramp_up(df_temp, target_roas, roas_hist_last) 
    
    df_temp['total_revenue_f'] = df_temp['rev_from_cost_f'] + df_temp['base_revenue_f']
    df_temp['profit_f'] = df_temp['total_revenue_f'] - df_temp['cost_f'] 
    df_temp['cumulative_profit'] = current_cumulative_profit + df_temp['profit_f'].cumsum()

    breakeven_row = df_temp[df_temp['cumulative_profit'] >= recovery_threshold]
    MAX_DAYS = len(df_temp)

    if breakeven_row.empty:
        return df_temp, MAX_DAYS + 1, df_temp['cumulative_profit'].iloc[-1] if not df_temp.empty else current_cumulative_profit
    
    # Đảm bảo df_temp có cột 'ds' (Prophet convention)
    if 'ds' not in df_temp.columns:
        df_temp['ds'] = pd.to_datetime(df_temp['date']) 
        
    days_to_breakeven = (breakeven_row['ds'].iloc[0].date() - df_temp['ds'].iloc[0].date()).days + 1
    
    return df_temp, days_to_breakeven, breakeven_row['cumulative_profit'].iloc[0]


def calculate_cumulative_profit_base_only(df_propensity_history, df_renewal_f, current_loss, recovery_threshold):
    """
    Tính toán lợi nhuận tích lũy chỉ sử dụng Doanh thu Base (Renewal) và Cost = 0.
    
    Args:
        df_propensity_history (pd.DataFrame): DataFrame chứa cột 'ds' (datetime64[ns]) cho khung thời gian dự báo.
        df_renewal_f (pd.DataFrame): DataFrame chứa dữ liệu gia hạn, bao gồm cột 'base_revenue_f' và cột ngày tháng ('ds' hoặc tương đương).
        current_loss (float): Lợi nhuận tích lũy hiện tại (có thể âm).
        recovery_threshold (float): Ngưỡng lợi nhuận tích lũy cần đạt.

    Returns: 
        tuple: (df_forecast, final_days)
    """
    
    # 1. Khởi tạo DataFrame dự báo dựa trên timeline
    df_forecast = df_propensity_history[['ds']].copy()
    
    # 2. Xử lý df_renewal_f: Đảm bảo cột ngày tháng là 'ds' và kiểu datetime64[ns]
    
    # --- Khắc phục lỗi KeyError: Đảm bảo tên cột là 'ds' ---
    if 'ds' not in df_renewal_f.columns:
        # Nếu cột ngày tháng trong df_renewal_f không phải là 'ds', bạn cần đổi tên nó ở đây.
        # Ví dụ: Giả sử cột ngày tháng là 'date'
        if 'date' in df_renewal_f.columns:
            df_renewal_f = df_renewal_f.rename(columns={'date': 'ds'})
        elif 'Date' in df_renewal_f.columns:
             df_renewal_f = df_renewal_f.rename(columns={'Date': 'ds'})
        else:
             # Nếu không tìm thấy cột ngày tháng nào, logic sẽ bị lỗi.
             # Tốt nhất là đảm bảo hàm get_renewal_forecast_data đã chuẩn hóa.
             pass 

    # --- Khắc phục lỗi ValueError: Ép kiểu dữ liệu ---
    # Chuyển đổi cột 'ds' trong df_renewal_f thành datetime để hợp nhất
    df_renewal_f['ds'] = pd.to_datetime(df_renewal_f['ds'])
    
    # 3. Hợp nhất với Doanh thu Base từ mô hình Renewal
    df_forecast = pd.merge(df_forecast, df_renewal_f[['ds', 'base_revenue_f']], on='ds', how='left')
    df_forecast['base_revenue_f'] = df_forecast['base_revenue_f'].fillna(0)
    
    # 4. Tính toán các cột chính cho kịch bản Base Only
    df_forecast['cost_f'] = 0.0
    df_forecast['acquisition_revenue_f'] = 0.0 # Không có người dùng mới
    df_forecast['total_revenue_f'] = df_forecast['base_revenue_f']
    df_forecast['profit_f'] = df_forecast['total_revenue_f'] - df_forecast['cost_f']

    # 5. Tính toán Lợi nhuận Tích lũy
    df_forecast['cumulative_profit'] = current_loss + df_forecast['profit_f'].cumsum()
    
    # 6. Tìm ngày hòa vốn (nếu có)
    df_breakeven = df_forecast[df_forecast['cumulative_profit'] >= recovery_threshold]
    
    final_days = FORECAST_HORIZON_DAYS + 1 # Mặc định là không hòa vốn

    if not df_breakeven.empty:
        first_breakeven_ds = df_breakeven['ds'].iloc[0]
        # Tính toán ngày cuối lịch sử (Giả định df_forecast bắt đầu ngay sau lịch sử)
        history_end_ds = df_forecast['ds'].iloc[0] - pd.Timedelta(days=1)
        
        final_days = (first_breakeven_ds - history_end_ds).days
        
    return df_forecast, final_days

def solve_breakeven_logic(df_propensity, df_renewal_f, current_cumulative_profit, recovery_threshold, 
                           mode='Days', target_value=None, fixed_roas=None, fixed_cost_day_1=None, 
                           cost_hist_last=None, roas_hist_last=None):
    if df_propensity.empty:
        return pd.DataFrame(), fixed_roas, fixed_cost_day_1, 0, "Prophet model failed (Empty historical data)."

    # Giữ nguyên logic tối ưu hóa (Brentq)

    if mode == 'Days':
        df_f, days, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit,
                                                    fixed_roas, fixed_cost_day_1, recovery_threshold, 
                                                    cost_hist_last, roas_hist_last)
        return df_f, fixed_roas, fixed_cost_day_1, days, None
    
    from scipy.optimize import brentq 

    if mode == 'ROAS':
        target_days = int(target_value)
        
        def func_roas(roas):
            df_f, days, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit,
                                                        roas, fixed_cost_day_1, recovery_threshold, 
                                                        cost_hist_last, roas_hist_last)
            return days - target_days
        
        a = 0.01  
        b = 10.0  
        
        try:
            if func_roas(a) <= 0: 
                df_f_min, _, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, a, fixed_cost_day_1, recovery_threshold, cost_hist_last, roas_hist_last)
                return df_f_min, a, fixed_cost_day_1, target_days, "Minimum ROAS (0.01) is sufficient to breakeven faster than target days."
            if func_roas(b) > 0:
                df_f_max, _, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, b, fixed_cost_day_1, recovery_threshold, cost_hist_last, roas_hist_last)
                return df_f_max, b, fixed_cost_day_1, target_days, "Cannot breakeven within target days (ROAS > 10.0 required)."

            result_roas = brentq(func_roas, a, b, xtol=1e-3)
            df_f, days_check, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit,
                                                              result_roas, fixed_cost_day_1, recovery_threshold, 
                                                              cost_hist_last, roas_hist_last)
            
            return df_f, result_roas, fixed_cost_day_1, days_check, None

        except ValueError:
            df_f_b, _, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, b, fixed_cost_day_1, recovery_threshold, cost_hist_last, roas_hist_last)
            return df_f_b, b, fixed_cost_day_1, target_days, "Solver failed. Cannot find ROAS within constraints [0.01, 10.0]."

    if mode == 'Cost':
        target_days = int(target_value)
        
        def func_cost(cost_day_1):
            df_f, days, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit,
                                                        fixed_roas, cost_day_1, recovery_threshold, 
                                                        cost_hist_last, roas_hist_last)
            return days - target_days
        
        cost_range_max = df_propensity['cost_prophet_f'].iloc[0] * 10 if not df_propensity.empty and df_propensity['cost_prophet_f'].iloc[0] > 0 else 10000 
        
        a = 0.01
        b = cost_range_max
        
        try:
            if func_cost(a) <= 0: 
                df_f_min, _, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, fixed_roas, a, recovery_threshold, cost_hist_last, roas_hist_last)
                return df_f_min, fixed_roas, a, target_days, "Minimum Cost ($0.01) is sufficient to breakeven faster than target days."
            if func_cost(b) > 0: 
                df_f_max, _, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, fixed_roas, b, recovery_threshold, cost_hist_last, roas_hist_last)
                return df_f_max, fixed_roas, b, target_days, f"Cost > ${cost_range_max:,.2f} is required to breakeven within target days."

            result_cost_day_1 = brentq(func_cost, a, b, xtol=1e-3)
            df_f, days_check, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit,
                                                              fixed_roas, result_cost_day_1, recovery_threshold, 
                                                              cost_hist_last, roas_hist_last)
            
            return df_f, fixed_roas, result_cost_day_1, days_check, None

        except ValueError:
            df_f_b, _, _ = calculate_cumulative_profit(df_propensity, df_renewal_f, current_cumulative_profit, fixed_roas, b, recovery_threshold, cost_hist_last, roas_hist_last)
            return df_f_b, fixed_roas, b, target_days, "Solver failed. Cannot find Day 1 Cost within constraints."

# --- EXPORT VARIABLES (For app.py to import) ---
__all__ = [
    "load_bigquery_renewal_data",
    "get_renewal_forecast_data",
    "get_prophet_cost_forecast",
    "solve_breakeven_logic",
    "calculate_cumulative_profit",
    "calculate_cumulative_profit_base_only",
    "client",
    "BQ_RENEWAL_TABLE_PATH",
    "BQ_DAILY_TABLE_PATH",
    "FORECAST_HORIZON_DAYS"
]