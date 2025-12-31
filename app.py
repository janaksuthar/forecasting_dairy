import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ReturnAI – FMCG Return Reduction",
    layout="wide"
)

st.title("📦 ReturnAI – FMCG AI Return Reduction Platform")
st.caption("AI-powered planning intelligence to reduce expiry & returns")

# ======================================================
# SIDEBAR – FILE UPLOAD
# ======================================================
st.sidebar.header("📂 Upload FMCG Sales & Return Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file (Sheet2)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("👈 Upload your Excel file to start analysis")
    st.stop()

# ======================================================
# DATA LOADING & CLEANING (PIVOT → AI READY)
# ======================================================
@st.cache_data
def load_and_clean_data(file):

    raw = pd.read_excel(file, sheet_name="Sheet2", header=None)

    # Extract years
    years = raw.iloc[2, 1::3].values

    products = raw.iloc[4:, 0].values

    records = []

    for i, product in enumerate(products):
        row = raw.iloc[4 + i, 1:].values

        for j, year in enumerate(years):
            idx = j * 3
            records.append({
                "product": str(product),
                "year": int(year),
                "net_sales_value": float(row[idx]),
                "wastage": float(row[idx + 1]),
                "expired_returns_value": float(row[idx + 2])
            })

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Feature engineering
    df["return_ratio"] = df["expired_returns_value"] / df["net_sales_value"]
    df["wastage_ratio"] = df["wastage"] / df["net_sales_value"]

    return df


df = load_and_clean_data(uploaded_file)

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.header("🔍 Filters")

selected_product = st.sidebar.selectbox(
    "Select Product",
    sorted(df["product"].unique())
)

# ======================================================
# EXECUTIVE DASHBOARD
# ======================================================
st.subheader("📊 Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Net Sales (₹)", f"{df['net_sales_value'].sum():,.0f}")
col2.metric("Expired Returns (₹)", f"{df['expired_returns_value'].sum():,.0f}")
col3.metric("Avg Return %", f"{df['return_ratio'].mean()*100:.2f}%")
col4.metric("Total Wastage (₹)", f"{df['wastage'].sum():,.0f}")

st.divider()

# ======================================================
# SALES vs RETURNS TREND
# ======================================================
st.subheader("📈 Sales vs Expired Returns Trend")

trend_df = df.groupby("year").sum(numeric_only=True).reset_index()

fig, ax = plt.subplots()
ax.plot(trend_df["year"], trend_df["net_sales_value"], marker="o", label="Net Sales")
ax.plot(trend_df["year"], trend_df["expired_returns_value"], marker="o", label="Expired Returns")
ax.set_xlabel("Year")
ax.set_ylabel("₹ Value")
ax.legend()
st.pyplot(fig)

# ======================================================
# PRODUCT RISK SEGMENTATION (CLUSTERING)
# ======================================================
st.subheader("🔥 Product Risk Segmentation (AI Clustering)")

cluster_features = df[[
    "net_sales_value",
    "expired_returns_value",
    "return_ratio",
    "wastage_ratio"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["risk_cluster"] = kmeans.fit_predict(X_scaled)

risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
df["risk_label"] = df["risk_cluster"].map(risk_map)

st.dataframe(
    df[["product", "year", "return_ratio", "risk_label"]]
    .sort_values("return_ratio", ascending=False),
    use_container_width=True
)

# ======================================================
# SKU INTELLIGENCE
# ======================================================
st.subheader("🧠 SKU Intelligence")

sku_df = df[df["product"] == selected_product]

col1, col2 = st.columns(2)

with col1:
    st.write("### Net Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(sku_df["year"], sku_df["net_sales_value"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("₹ Sales")
    st.pyplot(fig)

with col2:
    st.write("### Expired Returns Trend")
    fig, ax = plt.subplots()
    ax.plot(sku_df["year"], sku_df["expired_returns_value"], marker="o", color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel("₹ Returns")
    st.pyplot(fig)

avg_return = sku_df["return_ratio"].mean()

st.metric(
    "AI Return Risk Score",
    f"{avg_return*100:.1f}%",
    help="Higher value indicates higher expiry/return risk"
)

# ======================================================
# FORECASTING (RANDOM FOREST)
# ======================================================
st.subheader("🔮 Forecast: Expired Returns")

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

# ======================================================
# AI RECOMMENDATION ENGINE
# ======================================================
st.subheader("🤖 AI Recommendations")

if avg_return > 0.15:
    st.error(f"""
    🔴 **High Return Risk Detected**

    **Actions:**
    - Reduce production by 15–20%
    - Shorter replenishment cycles
    - Targeted promotions only
    - SKU rationalization

    **Potential Saving:** ₹ {(forecast * 0.25):,.0f}
    """)

elif avg_return > 0.08:
    st.warning(f"""
    🟡 **Moderate Return Risk**

    **Actions:**
    - Improve demand forecasting
    - Regional reallocation
    - Shelf-life planning

    **Potential Saving:** ₹ {(forecast * 0.15):,.0f}
    """)

else:
    st.success("""
    🟢 **Low Return Risk**

    **Actions:**
    - Maintain current planning
    - Monitor periodically
    """)

# ======================================================
# IMPACT SIMULATION
# ======================================================
st.subheader("📉 Impact Simulation")

reduction = st.slider("Expected Reduction in Returns (%)", 0, 40, 20)
savings = forecast * (reduction / 100)

st.metric("Projected Savings (₹)", f"{savings:,.0f}")

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption("ReturnAI | FMCG AI Planning Intelligence | Pilot-Ready MVP")
