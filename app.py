# app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Optional libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# =========================
# CONFIGURATION
# =========================
SHELF_LIFE_MAP = {
    'Milk': 7, 'Yoghurt': 21, 'Cheese': 60,
    'UHT Milk': 180, 'Fresh Juice': 5, 'Cream': 14
}

MONTH_TO_NUM = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

COLUMN_MAPPING = {
    'Brand': 'brand',
    'Calendar Year': 'year',
    'Net Sales Value CAF': 'sales',
    'Exp Returns Value': 'returns'
}


# =========================
# UTILITIES
# =========================
def detect_month_column(df):
    candidates = [
        'month', 'Month', 'MONTH',
        'English Month', 'Month Name', 'month_name'
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# =========================
# AI INSIGHTS
# =========================
def get_ai_insights(api_key, context):
    if not GROQ_AVAILABLE or not api_key:
        return None

    client = Groq(api_key=api_key)

    prompt = f"""
You are a supply chain AI expert.

Brand: {context['brand']}
Shelf Life: {context['shelf_life']} days
Forecast Horizon: {context['forecast_months']} months
Sales R²: {context['sales_r2']:.3f}
Returns R²: {context['returns_r2']:.3f}
Return Rate: {context['return_rate']:.2%}

Provide:
1. Model trustworthiness
2. Inventory risks
3. Actionable decisions
4. What humans must verify
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=600
    )
    return response.choices[0].message.content


# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # Detect month column safely
    month_col = detect_month_column(df)
    if month_col is None:
        st.error("❌ Month column not found (English Month / Month / numeric)")
        st.stop()

    # Convert month
    if df[month_col].dtype == object:
        df['month_num'] = df[month_col].str.strip().map(MONTH_TO_NUM)
    else:
        df['month_num'] = pd.to_numeric(df[month_col], errors='coerce')

    if df['month_num'].isnull().any():
        bad = df.loc[df['month_num'].isnull(), month_col].unique()
        st.error(f"❌ Invalid month values found: {bad}")
        st.stop()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01'
    )

    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').clip(lower=0)
    df['returns'] = pd.to_numeric(df['returns'], errors='coerce').clip(lower=0)

    df['shelf_life_days'] = df['brand'].map(SHELF_LIFE_MAP).fillna(30)

    monthly = df.groupby(['date', 'brand']).agg({
        'sales': 'sum',
        'returns': 'sum',
        'shelf_life_days': 'first'
    }).reset_index()

    return monthly.sort_values('date')


def add_lags(df, n=3):
    for i in range(1, n + 1):
        df[f'lag_{i}_sales'] = df.groupby('brand')['sales'].shift(i)
        df[f'lag_{i}_returns'] = df.groupby('brand')['returns'].shift(i)
    return df.dropna()


# =========================
# MODEL
# =========================
def train_model(df, target):
    features = [c for c in df.columns if 'lag_' in c or c in ['month_num', 'shelf_life_days']]
    X, y = df[features], df[target]

    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(X, y)
    pred = model.predict(X)

    metrics = {
        "rmse": mean_squared_error(y, pred, squared=False),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred)
    }
    return model, features, metrics


# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config("XAI Sales Forecast", layout="wide")

    st.title("🧠 Explainable AI Sales Forecasting")
    st.caption("Human-in-the-loop inventory decision system")

    uploaded = st.file_uploader("📁 Upload Excel File", type="xlsx")
    if not uploaded:
        st.stop()

    df = load_data(uploaded)
    brands = df['brand'].unique()
    brand = st.selectbox("🏷️ Select Brand", brands)
    horizon = st.slider("📅 Forecast Months", 1, 18, 6)

    df_b = df[df['brand'] == brand].copy()
    df_b['month_num'] = df_b['date'].dt.month
    df_b = add_lags(df_b)

    if len(df_b) < 12:
        st.error("❌ At least 12 months of data required")
        st.stop()

    sales_model, sales_feat, sales_m = train_model(df_b, 'sales')
    ret_model, ret_feat, ret_m = train_model(df_b, 'returns')

    last = df_b.iloc[-1]
    sales_fc, ret_fc, dates = [], [], []
    row = last.copy()

    for i in range(horizon):
        row['month_num'] = (row['month_num'] % 12) + 1
        Xs = pd.DataFrame([row[sales_feat]])
        sr = max(0, sales_model.predict(Xs)[0])
        rr = max(0, ret_model.predict(Xs)[0])

        sales_fc.append(sr)
        ret_fc.append(rr)
        dates.append(row['date'] + pd.DateOffset(months=i + 1))

        row['lag_1_sales'] = sr
        row['lag_1_returns'] = rr

    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🔍 Explainability", "🤖 AI Insights"])

    # Forecast
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_b['date'], y=df_b['sales'], name="Actual Sales"))
        fig.add_trace(go.Scatter(x=dates, y=sales_fc, name="Forecast Sales", line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

    # Explainability
    with tab2:
        st.subheader("Feature Importance (Sales)")
        fi = pd.DataFrame({
            "Feature": sales_feat,
            "Importance": sales_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.plotly_chart(px.bar(fi, x="Importance", y="Feature", orientation="h"),
                        use_container_width=True)

        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(sales_model)
            shap_vals = explainer.shap_values(df_b[sales_feat])
            shap_df = pd.DataFrame(
                np.abs(shap_vals), columns=sales_feat
            ).mean().sort_values(ascending=False)

            st.subheader("SHAP Mean |Impact|")
            st.plotly_chart(px.bar(
                shap_df,
                x=shap_df.values,
                y=shap_df.index,
                orientation="h"
            ), use_container_width=True)

    # AI Insights
    with tab3:
        api_key = st.text_input("🔑 Groq API Key", type="password")
        if st.button("Generate AI Insights"):
            context = {
                "brand": brand,
                "shelf_life": int(last['shelf_life_days']),
                "forecast_months": horizon,
                "sales_r2": sales_m['r2'],
                "returns_r2": ret_m['r2'],
                "return_rate": sum(ret_fc) / (sum(sales_fc) + 1)
            }
            st.markdown(get_ai_insights(api_key, context))


if __name__ == "__main__":
    main()
