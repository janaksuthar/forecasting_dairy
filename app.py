import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------
# CONFIG
# ------------------------------------
st.set_page_config(
    page_title="ReturnAI – FMCG Return Reduction",
    layout="wide"
)

st.title("📦 ReturnAI – FMCG AI Return Reduction Platform")
st.caption("AI-powered intelligence to prevent expired & returned FMCG products")

# ------------------------------------
# LOAD DATA
# ------------------------------------
@st.cache_data
st.sidebar.header("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales & Returns Excel File",
    type=["xlsx"]
)
def load_data():
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.sidebar.selectbox("Select Sheet", xls.sheet_names)
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    return df

df = load_data()

# ------------------------------------
# DATA CLEANING & FEATURE ENGINEERING
# ------------------------------------
df.columns = df.columns.str.lower().str.replace(" ", "_")

df["return_ratio"] = df["expired_returns_value"] / df["net_sales_value"]
df["wastage_ratio"] = df["wastage"] / df["net_sales_value"]

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# ------------------------------------
# SIDEBAR
# ------------------------------------
st.sidebar.header("🔍 Controls")
selected_product = st.sidebar.selectbox(
    "Select Product",
    df["product"].unique()
)

# ------------------------------------
# EXECUTIVE DASHBOARD
# ------------------------------------
st.subheader("📊 Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Net Sales (₹)", f"{df['net_sales_value'].sum():,.0f}")
col2.metric("Expired Returns (₹)", f"{df['expired_returns_value'].sum():,.0f}")
col3.metric("Avg Return %", f"{df['return_ratio'].mean()*100:.2f}%")
col4.metric("Total Wastage (₹)", f"{df['wastage'].sum():,.0f}")

st.divider()

# ------------------------------------
# SALES vs RETURNS TREND
# ------------------------------------
st.subheader("📈 Sales vs Expired Returns Trend")

fig, ax = plt.subplots()
ax.plot(df["year"], df["net_sales_value"], label="Net Sales", marker="o")
ax.plot(df["year"], df["expired_returns_value"], label="Expired Returns", marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("₹ Value")
ax.legend()
st.pyplot(fig)

# ------------------------------------
# PRODUCT RISK HEATMAP (CLUSTERING)
# ------------------------------------
st.subheader("🔥 Product Risk Segmentation (AI)")

cluster_features = df[[
    "net_sales_value",
    "expired_returns_value",
    "return_ratio",
    "wastage_ratio"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42)
df["risk_cluster"] = kmeans.fit_predict(X_scaled)

risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
df["risk_label"] = df["risk_cluster"].map(risk_map)

st.dataframe(
    df[["product", "year", "return_ratio", "risk_label"]]
    .sort_values("return_ratio", ascending=False),
    use_container_width=True
)

# ------------------------------------
# SKU INTELLIGENCE
# ------------------------------------
st.subheader("🧠 SKU Intelligence")

sku_df = df[df["product"] == selected_product]

col1, col2 = st.columns(2)

with col1:
    st.write("### Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(sku_df["year"], sku_df["net_sales_value"], marker="o")
    st.pyplot(fig)

with col2:
    st.write("### Return Trend")
    fig, ax = plt.subplots()
    ax.plot(sku_df["year"], sku_df["expired_returns_value"], marker="o", color="red")
    st.pyplot(fig)

avg_return = sku_df["return_ratio"].mean()

st.metric(
    "AI Return Risk Score",
    f"{avg_return*100:.1f}%",
    help="Higher value indicates higher expiry/return risk"
)

# ------------------------------------
# FORECASTING (SIMPLE REGRESSION)
# ------------------------------------
st.subheader("🔮 Forecast & Risk Projection")

X = df[["year"]]
y = df["expired_returns_value"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

next_year = df["year"].max() + 1
forecast = model.predict([[next_year]])[0]

st.metric(
    f"Forecasted Expired Returns for {next_year}",
    f"₹ {forecast:,.0f}"
)

# ------------------------------------
# AI RECOMMENDATION ENGINE
# ------------------------------------
st.subheader("🤖 AI Recommendations")

if avg_return > 0.15:
    st.error(
        f"""
        🔴 **High Return Risk Detected**
        
        **Recommended Actions:**
        - Reduce production by 15–20%
        - Shorten replenishment cycles
        - Avoid blanket promotions
        - Push selective SKU rationalization
        
        **Estimated Saving:** ₹ {(forecast * 0.25):,.0f}
        """
    )

elif avg_return > 0.08:
    st.warning(
        f"""
        🟡 **Moderate Return Risk**
        
        **Recommended Actions:**
        - Tighten demand forecasting
        - Regional reallocation
        - Improve shelf-life planning
        
        **Estimated Saving:** ₹ {(forecast * 0.15):,.0f}
        """
    )

else:
    st.success(
        """
        🟢 **Low Return Risk**
        
        **Recommended Actions:**
        - Maintain current planning
        - Monitor periodically
        """
    )

# ------------------------------------
# PERFORMANCE TRACKING
# ------------------------------------
st.subheader("📉 Impact Simulation")

reduction = st.slider("Expected Reduction in Returns (%)", 0, 40, 20)
savings = forecast * (reduction / 100)

st.metric("Projected Savings", f"₹ {savings:,.0f}")

st.caption("Simulation based on AI-driven planning adjustments")

# ------------------------------------
# FOOTER
# ------------------------------------
st.divider()
st.caption("ReturnAI | FMCG AI Planning Intelligence | Pilot-Ready MVP")
