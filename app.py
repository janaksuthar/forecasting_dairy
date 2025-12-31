import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="ReturnAI – FMCG Return Reduction",
    layout="wide"
)

st.title("📦 ReturnAI – FMCG AI Return Reduction Platform")
st.caption("AI-powered planning intelligence to reduce expired FMCG returns")

# =========================================
# FILE UPLOAD
# =========================================
st.sidebar.header("📂 Upload FMCG Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("👈 Upload your Excel file to start analysis")
    st.stop()

# =========================================
# LOAD DATA
# =========================================
df = pd.read_excel(uploaded_file)

# -----------------------------------------
# STANDARDIZE COLUMN NAMES
# -----------------------------------------
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

# -----------------------------------------
# COLUMN MAPPING (CRITICAL FIX)
# -----------------------------------------
column_map = {
    "channel_name": "channel",
    "english_month": "month",
    "calendar_year": "year",
    "brand": "brand",
    "prod_name": "product",
    "net_sales_value_caf": "net_sales_value",
    "exp_returns_litres": "exp_returns_litres",
    "gross_sales_lit": "gross_sales_lit",
    "exp_returns_value": "expired_returns_value"
}

df = df.rename(columns=column_map)

required_cols = list(column_map.values())

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# =========================================
# FEATURE ENGINEERING
# =========================================
df["return_ratio_value"] = df["expired_returns_value"] / df["net_sales_value"]
df["return_ratio_volume"] = df["exp_returns_litres"] / df["gross_sales_lit"]

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================================
# SIDEBAR FILTERS
# =========================================
st.sidebar.header("🔍 Filters")

selected_brand = st.sidebar.selectbox(
    "Select Brand",
    sorted(df["brand"].unique())
)

selected_product = st.sidebar.selectbox(
    "Select Product",
    sorted(df["product"].unique())
)

# =========================================
# EXECUTIVE DASHBOARD
# =========================================
st.subheader("📊 Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Net Sales (₹)", f"{df['net_sales_value'].sum():,.0f}")
col2.metric("Expired Returns Value (₹)", f"{df['expired_returns_value'].sum():,.0f}")
col3.metric("Avg Return % (Value)", f"{df['return_ratio_value'].mean()*100:.2f}%")
col4.metric("Avg Return % (Volume)", f"{df['return_ratio_volume'].mean()*100:.2f}%")

st.divider()

# =========================================
# YEARLY TREND
# =========================================
st.subheader("📈 Yearly Sales vs Expired Returns")

yearly = df.groupby("year").sum(numeric_only=True).reset_index()

fig, ax = plt.subplots()
ax.plot(yearly["year"], yearly["net_sales_value"], marker="o", label="Net Sales")
ax.plot(yearly["year"], yearly["expired_returns_value"], marker="o", label="Expired Returns")
ax.set_xlabel("Year")
ax.set_ylabel("₹ Value")
ax.legend()
st.pyplot(fig)

# =========================================
# PRODUCT RISK SEGMENTATION (AI)
# =========================================
st.subheader("🔥 Product Risk Segmentation (AI Clustering)")

product_level = (
    df.groupby("product")
    .agg({
        "net_sales_value": "sum",
        "expired_returns_value": "sum",
        "return_ratio_value": "mean",
        "return_ratio_volume": "mean"
    })
    .reset_index()
)

features = product_level[
    ["net_sales_value", "expired_returns_value", "return_ratio_value", "return_ratio_volume"]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
product_level["risk_cluster"] = kmeans.fit_predict(X_scaled)

risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
product_level["risk_label"] = product_level["risk_cluster"].map(risk_map)

st.dataframe(
    product_level.sort_values("return_ratio_value", ascending=False),
    use_container_width=True
)

# =========================================
# SKU INTELLIGENCE
# =========================================
st.subheader("🧠 SKU Intelligence")

sku_df = df[
    (df["brand"] == selected_brand) &
    (df["product"] == selected_product)
]

col1, col2 = st.columns(2)

with col1:
    st.write("### Net Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(sku_df["year"], sku_df["net_sales_value"], marker="o")
    st.pyplot(fig)

with col2:
    st.write("### Expired Returns Trend")
    fig, ax = plt.subplots()
    ax.plot(sku_df["year"], sku_df["expired_returns_value"], marker="o", color="red")
    st.pyplot(fig)

avg_return = sku_df["return_ratio_value"].mean()

st.metric("AI Return Risk Score", f"{avg_return*100:.2f}%")

# =========================================
# FORECASTING
# =========================================
st.subheader("🔮 Expired Returns Forecast")

X = yearly[["year"]]
y = yearly["expired_returns_value"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

next_year = int(yearly["year"].max() + 1)
forecast = model.predict([[next_year]])[0]

st.metric(
    f"Forecasted Expired Returns for {next_year}",
    f"₹ {forecast:,.0f}"
)

# =========================================
# AI RECOMMENDATIONS
# =========================================
st.subheader("🤖 AI Planning Recommendations")

if avg_return > 0.15:
    st.error("""
🔴 High Return Risk  
• Reduce production by 15–20%  
• Shorten replenishment cycles  
• Avoid blanket promotions  
• Rationalize SKUs
""")

elif avg_return > 0.08:
    st.warning("""
🟡 Moderate Return Risk  
• Improve forecasting accuracy  
• Channel-wise reallocation  
• Shelf-life based planning
""")

else:
    st.success("""
🟢 Low Return Risk  
• Maintain current planning  
• Monitor monthly
""")

# =========================================
# IMPACT SIMULATION
# =========================================
st.subheader("📉 Return Reduction Simulation")

reduction = st.slider("Expected Reduction in Returns (%)", 0, 40, 20)
savings = forecast * (reduction / 100)

st.metric("Projected Savings (₹)", f"{savings:,.0f}")

# =========================================
# FOOTER
# =========================================
st.divider()
st.caption("ReturnAI | FMCG AI Planning Intelligence | Production-Ready MVP")
