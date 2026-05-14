import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Adaptive Fraud Intelligence",
    layout="wide",
)

st.title("Adaptive Fraud Intelligence")
st.caption("Interactive demo of decision-centric fraud alerting")

st.sidebar.header("Simulation Settings")

transaction_limit = st.sidebar.selectbox(
    "Transaction limit",
    [1000, 3000, 10000],
    index=1,
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

# Demo baseline values from the thesis prototype
static_alerts = 303
static_frauds = 10
static_recall = 0.625
static_precision = 0.033
static_cost = 600060.43

# Lightweight demo simulation logic
decision_alerts = int(static_alerts * budget_multiplier)

if ranking_policy == "risk_zone":
    recall_gain = max(0, min(0.25, (budget_multiplier - 1.0) * 0.85))
    precision_gain = 0.0046 if budget_multiplier >= 1.3 else 0.0005
    cost_reduction = max(0, min(47496.46, (budget_multiplier - 1.0) * 160000))
else:
    recall_gain = max(0, min(0.12, (budget_multiplier - 1.0) * 0.35))
    precision_gain = -0.003
    cost_reduction = max(0, min(25000, (budget_multiplier - 1.0) * 80000))

decision_recall = round(static_recall + recall_gain, 3)
decision_precision = round(max(0.001, static_precision + precision_gain), 4)
decision_cost = round(static_cost - cost_reduction + investigation_cost * (decision_alerts - static_alerts), 2)
decision_frauds = int(round(16 * decision_recall))

comparison_df = pd.DataFrame(
    [
        {
            "system": "Static Threshold",
            "alerts": static_alerts,
            "frauds_caught": static_frauds,
            "recall": static_recall,
            "precision": static_precision,
            "total_cost": static_cost,
        },
        {
            "system": "Decision System",
            "alerts": decision_alerts,
            "frauds_caught": decision_frauds,
            "recall": decision_recall,
            "precision": decision_precision,
            "total_cost": decision_cost,
        },
    ]
)

st.subheader("Static Threshold vs Decision System")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Recall",
    f"{decision_recall:.3f}",
    delta=f"{decision_recall - static_recall:.3f}",
)

col2.metric(
    "Precision",
    f"{decision_precision:.3f}",
    delta=f"{decision_precision - static_precision:.3f}",
)

col3.metric(
    "Total Cost",
    f"{decision_cost:,.0f}",
    delta=f"{decision_cost - static_cost:,.0f}",
    delta_color="inverse",
)

col4.metric(
    "Alerts",
    f"{decision_alerts}",
    delta=f"{decision_alerts - static_alerts}",
)

st.dataframe(comparison_df, use_container_width=True)

st.subheader("Operating Curve")

curve_rows = []

for multiplier in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    alerts = int(static_alerts * multiplier)

    if multiplier < 1.2:
        recall = 0.625
        frauds = 10
        cost = 600060.43 + (alerts - static_alerts) * investigation_cost
        precision = frauds / alerts
    elif multiplier < 1.3:
        recall = 0.750
        frauds = 12
        cost = 555460.97
        precision = frauds / alerts
    else:
        recall = 0.875
        frauds = 14
        cost = 552563.97
        precision = frauds / alerts

    curve_rows.append(
        {
            "budget_multiplier": multiplier,
            "alert_budget": alerts,
            "alerts": min(alerts, 372) if multiplier >= 1.3 else alerts,
            "frauds_caught": frauds,
            "recall": recall,
            "precision": round(precision, 4),
            "total_cost": cost,
            "cost_diff_vs_static": round(cost - static_cost, 2),
            "recall_diff_vs_static": round(recall - static_recall, 3),
        }
    )

curve_df = pd.DataFrame(curve_rows)
st.dataframe(curve_df, use_container_width=True)

best_row = curve_df.loc[curve_df["total_cost"].idxmin()]

st.success(
    f"Best operating point: multiplier={best_row['budget_multiplier']} | "
    f"recall={best_row['recall']:.3f} | "
    f"cost={best_row['total_cost']:,.0f}"
)

st.line_chart(
    curve_df.set_index("budget_multiplier")[["recall", "total_cost"]]
)

st.subheader("Analyst Queue Preview")

alerts_df = pd.DataFrame(
    [
        {
            "transaction_id": 1818,
            "type": "TRANSFER",
            "amount": 2317408.88,
            "fraud_score": 1.000,
            "rank_score": 81.1897,
            "expected_benefit": 2317398.88,
            "severity": "HIGH",
            "reason": "Very high model confidence; prioritized for immediate analyst review",
        },
        {
            "transaction_id": 374,
            "type": "TRANSFER",
            "amount": 1478772.04,
            "fraud_score": 1.000,
            "rank_score": 70.9529,
            "expected_benefit": 1478761.99,
            "severity": "HIGH",
            "reason": "Very high model confidence; prioritized for immediate analyst review",
        },
        {
            "transaction_id": 969,
            "type": "TRANSFER",
            "amount": 1277212.77,
            "fraud_score": 1.000,
            "rank_score": 67.9014,
            "expected_benefit": 1277202.77,
            "severity": "HIGH",
            "reason": "Very high model confidence; prioritized for immediate analyst review",
        },
    ]
)

c1, c2, c3 = st.columns(3)
c1.metric("High severity", 113)
c2.metric("Medium severity", 83)
c3.metric("Low severity", 4)

st.dataframe(alerts_df, use_container_width=True)

st.info(
    "This deployed app is a lightweight interactive demo based on the thesis prototype outputs. "
    "The full local version connects to a FastAPI backend and model-serving pipeline."
)