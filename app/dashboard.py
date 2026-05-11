import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fraud Decision System Dashboard",
    layout="wide",
)

st.title("Fraud Decision System Dashboard")
st.caption(
    "Decision-centric fraud detection: static threshold vs risk-zone decision system"
)

# =========================
# SIDEBAR CONTROLS
# =========================

st.sidebar.header("Simulation Settings")

limit = st.sidebar.selectbox(
    "Transaction limit",
    [1000, 3000, 10000],
    index=1,  # default 3000 to avoid timeout
)

investigation_cost = st.sidebar.number_input(
    "Investigation cost",
    min_value=1.0,
    value=10.0,
    step=1.0,
)

ranking_policy = st.sidebar.selectbox(
    "Ranking policy",
    ["risk_zone", "score", "benefit", "hybrid"],
    index=0,
)

risk_zone_floor = st.sidebar.slider(
    "Risk-zone floor",
    min_value=0.0,
    max_value=0.5,
    value=0.3,
    step=0.05,
)

budget_multiplier = st.sidebar.slider(
    "Budget multiplier",
    min_value=1.0,
    max_value=1.5,
    value=1.4,
    step=0.1,
)


# =========================
# API HELPER
# =========================

@st.cache_data(ttl=300)
def get_json(endpoint: str, params: dict):
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        params=params,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# =========================
# LOAD DATA FROM API
# =========================

comparison_params = {
    "limit": limit,
    "investigation_cost": investigation_cost,
    "ranking_policy": ranking_policy,
    "risk_zone_floor": risk_zone_floor,
    "budget_multiplier": budget_multiplier,
}

curve_params = {
    "limit": limit,
    "investigation_cost": investigation_cost,
    "ranking_policy": ranking_policy,
    "risk_zone_floor": risk_zone_floor,
}

try:
    with st.spinner("Loading comparison and operating curve from FastAPI..."):
        comparison = get_json("/comparison", comparison_params)
        operating_curve = get_json("/operating_curve", curve_params)

except Exception as e:
    st.error(
        "Could not connect to the FastAPI backend. "
        "Make sure it is running with: py -m uvicorn app.api.main:app --reload"
    )
    st.exception(e)
    st.stop()


static = comparison["static"]
decision = comparison["decision_system"]


# =========================
# KPI CARDS
# =========================

st.subheader("Static Threshold vs Decision System")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Recall",
    f"{decision['recall']:.3f}",
    delta=f"{comparison['recall_diff']:.3f}",
)

col2.metric(
    "Precision",
    f"{decision['precision']:.3f}",
    delta=f"{comparison['precision_diff']:.3f}",
)

col3.metric(
    "Total Cost",
    f"{decision['total_cost']:,.0f}",
    delta=f"{comparison['cost_diff']:,.0f}",
    delta_color="inverse",
)

col4.metric(
    "Alerts",
    f"{decision['alerts']}",
    delta=f"{decision['alerts'] - static['alerts']}",
)


# =========================
# COMPARISON TABLE
# =========================

comparison_df = pd.DataFrame(
    [
        {
            "system": "Static Threshold",
            "alerts": static["alerts"],
            "frauds_caught": static["frauds_caught"],
            "recall": static["recall"],
            "precision": static["precision"],
            "total_cost": static["total_cost"],
        },
        {
            "system": "Decision System",
            "alerts": decision["alerts"],
            "frauds_caught": decision["frauds_caught"],
            "recall": decision["recall"],
            "precision": decision["precision"],
            "total_cost": decision["total_cost"],
        },
    ]
)

st.dataframe(comparison_df, use_container_width=True)


# =========================
# OPERATING CURVE
# =========================

st.subheader("Operating Curve")

curve_df = pd.DataFrame(operating_curve["operating_curve"])

st.dataframe(curve_df, use_container_width=True)

best_row = curve_df.loc[curve_df["total_cost"].idxmin()]

st.success(
    f"Best operating point: multiplier={best_row['budget_multiplier']} | "
    f"recall={best_row['recall']:.3f} | "
    f"cost={best_row['total_cost']:,.0f}"
)

fig1, ax1 = plt.subplots()
ax1.plot(curve_df["budget_multiplier"], curve_df["recall"], marker="o")
ax1.set_xlabel("Alert Budget Multiplier")
ax1.set_ylabel("Recall")
ax1.set_title("Recall vs Alert Capacity")
ax1.grid(True)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(curve_df["budget_multiplier"], curve_df["total_cost"], marker="o")
ax2.set_xlabel("Alert Budget Multiplier")
ax2.set_ylabel("Total Operational Cost")
ax2.set_title("Cost vs Alert Capacity")
ax2.grid(True)
st.pyplot(fig2)


# =========================
# ANALYST QUEUE
# =========================

st.subheader("Analyst Queue")

st.info(
    "Top flagged transactions prioritized for analyst review. "
    "Severity is assigned from model confidence, while rank_score controls analyst ordering."
)

if limit <= 3000:
    try:
        with st.spinner("Loading analyst queue..."):
            alerts = get_json(
                "/alerts",
                {
                    "limit": limit,
                    "investigation_cost": investigation_cost,
                    "ranking_policy": ranking_policy,
                    "risk_zone_floor": risk_zone_floor,
                    "budget_multiplier": budget_multiplier,
                },
            )

        alerts_df = pd.DataFrame(alerts)

        if alerts_df.empty:
            st.warning("No alerts generated.")
        else:
            high_count = int((alerts_df["severity"] == "HIGH").sum())
            medium_count = int((alerts_df["severity"] == "MEDIUM").sum())
            low_count = int((alerts_df["severity"] == "LOW").sum())

            c1, c2, c3 = st.columns(3)

            c1.metric("High severity", high_count)
            c2.metric("Medium severity", medium_count)
            c3.metric("Low severity", low_count)

            st.dataframe(
                alerts_df[
                    [
                        "transaction_id",
                        "step",
                        "type",
                        "amount",
                        "fraud_score",
                        "rank_score",
                        "expected_benefit",
                        "severity",
                        "reason",
                        "isFraud",
                    ]
                ],
                use_container_width=True,
            )

            csv = alerts_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Analyst Queue CSV",
                data=csv,
                file_name="analyst_queue.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.warning("Could not load analyst queue.")
        st.exception(e)
else:
    st.warning("Analyst queue disabled for limit=10000 to keep dashboard responsive.")