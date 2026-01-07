# File name: sales_forecast_xai_app.py
# Run with: streamlit run sales_forecast_xai_app.py
# Install: pip install streamlit pandas numpy xgboost scikit-learn plotly openpyxl shap groq

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Groq API for AI assistant
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

SHELF_LIFE_MAP = {
    'Actimel': 28, 'Activa Set YoghLight': 21, 'Activia Greek Drink': 21,
    'Activia Laban FF': 14, 'Activia Laban Light': 14, 'Activia Set Yoghurt': 21,
    'Activia Stirred Youg': 21, 'Activia YGO': 28, 'Alpro': 180, 'Cheese': 60,
    'Cream': 14, 'Creme Caramel': 90, 'Danao': 180, 'Danette': 180, 'Dunkin': 90,
    'Fresh Juice': 5, 'Greek Yoghurt': 21, 'Laban': 14, 'Labneh UHT': 180,
    'Milk': 7, 'Multi Purpose Cream': 90, 'Organic Juice': 180, 'Rashaka Milk': 180,
    'Safio Fresh Drink': 7, 'Safio Fresh Spoon': 21, 'Safio UHT': 180,
    'UHT Milk': 180, 'Yoghurt': 21,
}

COLUMN_MAPPING = {
    'Channel Name': 'channel', 'English Month': 'month', 'Calendar Year': 'year',
    'Brand': 'brand', 'Prod Name': 'product', 'Net Sales Value CAF': 'net_sales_value',
    'Exp Returns Litres': 'exp_returns_litres', 'Gross Sales Lit': 'gross_sales_lit',
    'Exp Returns Value': 'exp_returns_value'
}

MONTH_TO_NUM = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# ════════════════════════════════════════════════════════════════
# GROQ AI ASSISTANT
# ════════════════════════════════════════════════════════════════

def get_ai_insights(groq_api_key, context_data, user_question=None):
    """Get AI-powered insights using Groq API"""
    if not GROQ_AVAILABLE or not groq_api_key:
        return None
    
    try:
        client = Groq(api_key=gsk_8h6hd31kthKsKlTRVvqcWGdyb3FY25FbkscZkIzKRrpSCiidjxD8y)
        
        # Build comprehensive context
        prompt = f"""You are an expert AI assistant for supply chain and inventory management. 
Analyze the following forecasting data and provide actionable insights:

BRAND: {context_data.get('brand', 'Unknown')}
SHELF LIFE: {context_data.get('shelf_life', 'N/A')} days
HISTORICAL DATA: {context_data.get('history_months', 0)} months
FORECAST PERIOD: {context_data.get('forecast_months', 0)} months

MODEL PERFORMANCE:
- Sales Model RMSE: {context_data.get('sales_rmse', 'N/A')}
- Returns Model RMSE: {context_data.get('returns_rmse', 'N/A')}
- Sales Model R²: {context_data.get('sales_r2', 'N/A')}
- Returns Model R²: {context_data.get('returns_r2', 'N/A')}

FORECASTED TOTALS:
- Total Predicted Sales: {context_data.get('total_sales', 0):,.0f}
- Total Predicted Returns: {context_data.get('total_returns', 0):,.0f}
- Return Rate: {context_data.get('return_rate', 0):.2%}
- Recommended Stock: {context_data.get('total_stock', 0):,.0f}

KEY INSIGHTS FROM DATA:
- Sales Trend: {context_data.get('sales_trend', 'unknown')}
- Return Trend: {context_data.get('return_trend', 'unknown')}
- Seasonality Detected: {context_data.get('seasonality', 'unknown')}
- Data Quality Score: {context_data.get('data_quality', 'N/A')}/10

TOP FEATURES INFLUENCING SALES:
{context_data.get('sales_features', 'N/A')}

TOP FEATURES INFLUENCING RETURNS:
{context_data.get('returns_features', 'N/A')}

{"USER QUESTION: " + user_question if user_question else ""}

Please provide:
1. **Model Trustworthiness Assessment**: Evaluate the reliability of these predictions based on RMSE, R², and historical data length
2. **Key Risk Factors**: Identify potential risks in inventory management for this product
3. **Actionable Recommendations**: Specific steps the practitioner should take
4. **Decision Support**: What decisions should be made based on this forecast?
5. **Areas of Uncertainty**: What should the human verify or investigate further?

Be specific, concise, and actionable. Focus on practical business decisions."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

# ════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_prepare_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAPPING)

    required = ['channel', 'month', 'year', 'brand', 'net_sales_value', 'exp_returns_value']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['net_sales_value'] = pd.to_numeric(df['net_sales_value'], errors='coerce').clip(lower=0)
    df['exp_returns_value'] = pd.to_numeric(df['exp_returns_value'], errors='coerce').clip(lower=0)
    df['month_num'] = df['month'].map(MONTH_TO_NUM)
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01', errors='coerce')
    df = df.dropna(subset=['date'])
    df['shelf_life_days'] = df['brand'].map(SHELF_LIFE_MAP).fillna(30)

    monthly = df.groupby(['date', 'brand']).agg({
        'net_sales_value': 'sum',
        'exp_returns_value': 'sum',
        'shelf_life_days': 'first'
    }).reset_index().rename(columns={'net_sales_value': 'sales', 'exp_returns_value': 'returns'})

    monthly = monthly.sort_values(['brand', 'date'])
    return monthly, df

def add_lags(df, n_lags=3):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}_sales'] = df.groupby('brand')['sales'].shift(lag)
        df[f'lag_{lag}_returns'] = df.groupby('brand')['returns'].shift(lag)
    return df

def assess_data_quality(product_data):
    """Assess quality of historical data"""
    score = 10.0
    issues = []
    
    # Check data length
    if len(product_data) < 12:
        score -= 3
        issues.append("Less than 12 months of data")
    elif len(product_data) < 24:
        score -= 1.5
        issues.append("Less than 24 months of data (seasonality may be unclear)")
    
    # Check for missing values
    missing_pct = product_data[['sales', 'returns']].isnull().sum().sum() / (len(product_data) * 2)
    if missing_pct > 0.05:
        score -= 2
        issues.append(f"High missing values: {missing_pct:.1%}")
    
    # Check for zeros
    zero_sales = (product_data['sales'] == 0).sum() / len(product_data)
    if zero_sales > 0.2:
        score -= 1.5
        issues.append(f"Many zero sales months: {zero_sales:.1%}")
    
    # Check variance
    cv_sales = product_data['sales'].std() / product_data['sales'].mean() if product_data['sales'].mean() > 0 else 0
    if cv_sales > 2:
        score -= 1
        issues.append(f"Very high sales volatility (CV={cv_sales:.2f})")
    
    return max(0, score), issues

# ════════════════════════════════════════════════════════════════
# MODEL TRAINING WITH ENHANCED METRICS
# ════════════════════════════════════════════════════════════════

def train_xgboost_brand(product_data, target_col='sales'):
    if len(product_data) < 10:
        return None, None, None, None, {}

    features = ['month_num', 'shelf_life_days']
    for i in range(1, 4):
        features += [f'lag_{i}_sales', f'lag_{i}_returns']

    available_features = [f for f in features if f in product_data.columns]
    X = product_data[available_features]
    y = product_data[target_col]

    if len(X) < 8:
        return None, None, None, None, {}

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)
    
    # Calculate comprehensive metrics
    y_pred = model.predict(X)
    metrics = {
        'rmse': mean_squared_error(y, y_pred, squared=False),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'mape': np.mean(np.abs((y - y_pred) / (y + 1))) * 100  # +1 to avoid division by zero
    }
    
    return model, available_features, X, y, metrics

def compute_feature_importance(model, feature_names, X_sample=None):
    """Compute feature importance with SHAP if available"""
    if SHAP_AVAILABLE and X_sample is not None and len(X_sample) > 0:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap,
                'method': 'SHAP (mean |impact|)'
            }).sort_values('importance', ascending=False)
            return fi_df, shap_values, explainer
        except Exception:
            pass

    # Fallback to built-in importance
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'method': 'XGBoost Gain'
        }).sort_values('importance', ascending=False)
        return fi_df, None, None
    except Exception:
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [0.0] * len(feature_names),
            'method': 'N/A'
        }).sort_values('importance', ascending=False)
        return fi_df, None, None

# ════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="XAI Sales Forecaster", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0;}
        .sub-header {font-size: 1.2rem; color: #666; margin-top: 0;}
        .metric-card {background: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
        .insight-box {background: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;}
        .warning-box {background: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107;}
        .success-box {background: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">🧠 Explainable AI Sales Forecasting System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Human-AI Collaboration for Intelligent Inventory Management</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("⚙️ Configuration")
        
        uploaded_file = st.file_uploader("📁 Upload Sales Data", type=["xlsx"], help="Upload your Excel file with sales history")
        
        if uploaded_file is None:
            st.info("👆 Upload your data file to begin")
            st.stop()
        
        st.divider()
        
        # Groq API Key
        st.subheader("🤖 AI Assistant")
        groq_api_key = st.text_input("Groq API Key", type="password", help="Get your free API key from console.groq.com")
        enable_ai = st.checkbox("Enable AI Decision Support", value=bool(groq_api_key))
        
        st.divider()
        st.subheader("📊 Forecast Settings")

    # Load data
    with st.spinner("🔄 Loading and analyzing data..."):
        monthly, raw_df = load_and_prepare_data(uploaded_file)

    # Sidebar controls continued
    with st.sidebar:
        available_brands = sorted(monthly['brand'].unique())
        selected_brand = st.selectbox("🏷️ Select Brand", available_brands, help="Choose product brand to analyze")
        
        forecast_months = st.slider("📅 Forecast Horizon (months)", 1, 18, 6)
        lag_count = st.slider("🔢 Lag Features", 1, 6, 3, help="Number of historical periods to use as features")
        
        st.divider()
        st.subheader("🎛️ Advanced Controls")
        
        buffer_mode = st.radio("Safety Stock Strategy", ["Auto (AI-driven)", "Manual Override"])
        
        if buffer_mode == "Manual Override":
            human_buffer = st.slider("Buffer Multiplier", 0.5, 3.0, 1.15, 0.05, 
                                     help="Higher = more safety stock")
        else:
            human_buffer = None
        
        confidence_level = st.select_slider("Confidence Level", 
                                           options=[80, 85, 90, 95, 99], 
                                           value=90,
                                           help="Higher = wider prediction intervals")

    # Filter and prepare data
    product_data = monthly[monthly['brand'] == selected_brand].copy()
    
    if len(product_data) < 12:
        st.error(f"⚠️ Insufficient data for {selected_brand}: only {len(product_data)} months available. Need at least 12 months.")
        st.stop()

    # Assess data quality
    data_quality_score, data_issues = assess_data_quality(product_data)
    
    # Add lags and prepare
    product_data = add_lags(product_data, lag_count)
    product_data = product_data.dropna()
    product_data['month_num'] = product_data['date'].dt.month

    # Train models
    with st.spinner("🎯 Training XGBoost models..."):
        model_sales, features_sales, X_sales, y_sales, metrics_sales = train_xgboost_brand(product_data, 'sales')
        model_returns, features_returns, X_returns, y_returns, metrics_returns = train_xgboost_brand(product_data, 'returns')

    if model_sales is None or model_returns is None:
        st.error("❌ Model training failed - insufficient data after preprocessing")
        st.stop()

    # ═══════════════════════════════════════════════════════════
    # FORECASTING
    # ═══════════════════════════════════════════════════════════
    
    current_row = product_data.iloc[-1].copy()
    forecast_dates, sales_forecast, returns_forecast = [], [], []
    recommended_stock, lower_bound, upper_bound = [], [], []
    working_row = current_row.copy()

    for i in range(forecast_months):
        next_date = working_row['date'] + pd.DateOffset(months=1)
        forecast_dates.append(next_date)
        month_num = next_date.month

        X_next_sales = pd.DataFrame([{
            'month_num': month_num,
            'shelf_life_days': working_row['shelf_life_days'],
            **{f'lag_{j}_sales': working_row.get(f'lag_{j}_sales', 0) for j in range(1, lag_count+1)},
            **{f'lag_{j}_returns': working_row.get(f'lag_{j}_returns', 0) for j in range(1, lag_count+1)}
        }])

        if features_sales:
            for col in features_sales:
                if col not in X_next_sales.columns:
                    X_next_sales[col] = 0
            X_next_sales = X_next_sales[features_sales]

        pred_sales = max(0, model_sales.predict(X_next_sales)[0])
        sales_forecast.append(pred_sales)

        # Uncertainty estimation (simple approach using training RMSE)
        sales_std = metrics_sales['rmse']
        z_score = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
        lower_bound.append(max(0, pred_sales - z_score * sales_std))
        upper_bound.append(pred_sales + z_score * sales_std)

        X_next_returns = X_next_sales.copy()
        if features_returns and 'lag_1_sales' in features_returns:
            X_next_returns['lag_1_sales'] = pred_sales

        if features_returns:
            for col in features_returns:
                if col not in X_next_returns.columns:
                    X_next_returns[col] = 0
            X_next_returns = X_next_returns[features_returns]

        pred_returns = max(0, model_returns.predict(X_next_returns)[0])
        returns_forecast.append(pred_returns)

        # Smart buffer calculation
        if human_buffer is not None:
            buffer = human_buffer
        else:
            shelf_life = working_row['shelf_life_days']
            return_rate = pred_returns / (pred_sales + 0.01)
            if shelf_life <= 7:
                buffer = 1.25 + (return_rate * 0.5)
            elif shelf_life <= 14:
                buffer = 1.15 + (return_rate * 0.3)
            else:
                buffer = 1.05 + (return_rate * 0.2)
        
        stock = pred_sales - pred_returns * buffer
        recommended_stock.append(max(0, stock))

        # Update lags
        for lag in range(lag_count, 1, -1):
            working_row[f'lag_{lag}_sales'] = working_row.get(f'lag_{lag-1}_sales', 0)
            working_row[f'lag_{lag}_returns'] = working_row.get(f'lag_{lag-1}_returns', 0)
        working_row['lag_1_sales'] = pred_sales
        working_row['lag_1_returns'] = pred_returns
        working_row['date'] = next_date

    # ═══════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ═══════════════════════════════════════════════════════════
    
    # Key metrics at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Data Quality", f"{data_quality_score:.1f}/10", 
                 delta="Good" if data_quality_score >= 7 else "Review Needed",
                 delta_color="normal" if data_quality_score >= 7 else "inverse")
    with col2:
        st.metric("🎯 Sales Model R²", f"{metrics_sales['r2']:.3f}",
                 delta="Strong" if metrics_sales['r2'] >= 0.7 else "Moderate")
    with col3:
        st.metric("🔄 Returns Model R²", f"{metrics_returns['r2']:.3f}",
                 delta="Strong" if metrics_returns['r2'] >= 0.7 else "Moderate")
    with col4:
        avg_return_rate = np.mean(returns_forecast) / (np.mean(sales_forecast) + 0.01)
        st.metric("📉 Forecast Return Rate", f"{avg_return_rate:.1%}")

    # Data quality warning
    if data_quality_score < 7:
        st.markdown(f'<div class="warning-box"><b>⚠️ Data Quality Issues Detected:</b><ul>{"".join([f"<li>{issue}</li>" for issue in data_issues])}</ul></div>', 
                   unsafe_allow_html=True)

    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecasts", "🔍 Explainability", "🤖 AI Insights", "📋 Report"])

    with tab1:
        # Sales forecast with confidence intervals
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Sales Forecast with Confidence Intervals", "Returns Forecast"),
            vertical_spacing=0.12
        )

        # Historical sales
        fig.add_trace(go.Scatter(
            x=product_data['date'], y=product_data['sales'],
            mode='lines+markers', name='Historical Sales',
            line=dict(color='blue', width=2)
        ), row=1, col=1)

        # Forecast sales with confidence band
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=sales_forecast,
            mode='lines+markers', name='Forecast Sales',
            line=dict(color='orange', width=2, dash='dash')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself', fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level}% Confidence', showlegend=True
        ), row=1, col=1)

        # Returns
        fig.add_trace(go.Scatter(
            x=product_data['date'], y=product_data['returns'],
            mode='lines+markers', name='Historical Returns',
            line=dict(color='green', width=2)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=forecast_dates, y=returns_forecast,
            mode='lines+markers', name='Forecast Returns',
            line=dict(color='red', width=2, dash='dash')
        ), row=2, col=1)

        fig.update_layout(height=700, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("📊 Detailed Forecast")
        df_forecast = pd.DataFrame({
            'Month': [d.strftime('%Y-%m') for d in forecast_dates],
            'Sales (Forecast)': np.round(sales_forecast, 0),
            'Lower Bound': np.round(lower_bound, 0),
            'Upper Bound': np.round(upper_bound, 0),
            'Returns (Forecast)': np.round(returns_forecast, 0),
            'Return Rate': [f"{r/(s+0.01):.1%}" for s, r in zip(sales_forecast, returns_forecast)],
            'Recommended Stock': np.round(recommended_stock, 0)
        })
        
        st.dataframe(df_forecast, use_container_width=True, hide_index=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forecast Sales", f"{sum(sales_forecast):,.0f}")
        with col2:
            st.metric("Total Forecast Returns", f"{sum(returns_forecast):,.0f}")
        with col3:
            st.metric("Recommended Total Stock", f"{sum(recommended_stock):,.0f}")

    with tab2:
        st.header("🔍 Model Explainability")
        
        # Model performance
        st.subheader("📊 Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sales Model**")
            metrics_df_sales = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R² Score', 'MAPE (%)'],
                'Value': [
                    f"{metrics_sales['rmse']:.2f}",
                    f"{metrics_sales['mae']:.2f}",
                    f"{metrics_sales['r2']:.4f}",
                    f"{metrics_sales['mape']:.2f}"
                ]
            })
            st.dataframe(metrics_df_sales, hide_index=True, use_container_width=True)
            
            # Interpretation
            if metrics_sales['r2'] >= 0.8:
                st.success("✅ Excellent model fit")
            elif metrics_sales['r2'] >= 0.6:
                st.info("ℹ️ Good model fit")
            else:
                st.warning("⚠️ Moderate fit - use predictions cautiously")
        
        with col2:
            st.markdown("**Returns Model**")
            metrics_df_returns = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R² Score', 'MAPE (%)'],
                'Value': [
                    f"{metrics_returns['rmse']:.2f}",
                    f"{metrics_returns['mae']:.2f}",
                    f"{metrics_returns['r2']:.4f}",
                    f"{metrics_returns['mape']:.2f}"
                ]
            })
            st.dataframe(metrics_df_returns, hide_index=True, use_container_width=True)
            
            if metrics_returns['r2'] >= 0.8:
                st.success("✅ Excellent model fit")
            elif metrics_returns['r2'] >= 0.6:
                st.info("ℹ️ Good model fit")
            else:
                st.warning("⚠️ Moderate fit - use predictions cautiously")

        st.divider()

        # Feature importance
        st.subheader("🎯 Feature Importance Analysis")
        
        fi_sales, shap_vals_sales, explainer_sales = compute_feature_importance(
            model_sales, features_sales, X_sales.head(100) if X_sales is not None else None
        )
        fi_returns, shap_vals_returns, explainer_returns = compute_feature_importance(
            model_returns, features_returns, X_returns.head(100) if X_returns is not None else None
        )

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sales Model Features**")
            fig_fi_sales = px.bar(fi_sales, x='importance', y='feature', orientation='h',
