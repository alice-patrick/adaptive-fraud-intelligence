import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Adaptive Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Adaptive Fraud Intelligence")
st.caption(
    "Interactive demo of decision-centric fraud detection with adaptive alerting, "
    "cost-aware ranking, and analyst queue prioritization."
)

st.markdown(
    """
This demo shows how fraud detection can move beyond static thresholds by adding
an operational decision layer that considers investigation capacity, alert cost,
ranking policy, and analyst workflow.

**GitHub Repository:**  
https://github.com/alice-patrick/adaptive-fraud-intelligence
"""
)

st.divider()

st.sidebar.title("Simulation Settings")

st.sidebar.markdown(
    """
Adjust the operational parameters below to simulate how alert capacity and
ranking policy affect recall, cost, and analyst workload.
"""
)

transaction_limit = st.sidebar.selectbox(
    "Transaction limit",
    [1000, 3000, 10000],
    index=1,
)

investigation_cost = st.sidebar.number_input(
    "Investigation cost per alert",
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
    "Alert budget multiplier",
    min_value=1.0,
    max_value=1.5,
    value=1.4,
    step=0.1,
)

st.sidebar.info(
    "This deployed version is a lightweight interactive demo. "
    "The full local version connects to a FastAPI backend and model-serving pipeline."
)

static_alerts = 303
static_frauds = 10
static_recall = 0.625
static_precision = 0.033
static_cost = 600060.43

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
decision_cost = round(
    static_cost
    - cost_reduction
    + investigation_cost * (decision_alerts - static_alerts),
    2,
)
decision_frauds = int(round(16 * decision_recall))

comparison_df = pd.DataFrame(
    [
        {
            "System": "Static Threshold",
            "Alerts": static_alerts,
            "Frauds Caught": static_frauds,
            "Recall": static_recall,
            "Precision": static_precision,
            "Total Cost": static_cost,
        },
        {
            "System": "Decision System",
            "Alerts": decision_alerts,
            "Frauds Caught": decision_frauds,
            "Recall": decision_recall,
            "Precision": decision_precision,
            "Total Cost": decision_cost,
        },
    ]
)

overview_tab, curve_tab, queue_tab, design_tab = st.tabs(
    ["Overview", "Operating Curve", "Analyst Queue", "System Design"]
)

with overview_tab:
    st.subheader("Static Threshold vs Adaptive Decision System")

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

    st.success(
        "The adaptive decision system improves fraud recall while keeping alert volume "
        "within an operationally controlled budget."
    )

    st.markdown(
        """
### What this demonstrates

- Fraud detection is not only a prediction problem.
- Model scores must be converted into operational decisions.
- Alert budgets, investigation cost, and analyst workload matter.
- A decision layer can outperform static thresholding under realistic constraints.
"""
    )

with curve_tab:
    st.subheader("Operating Curve Analysis")

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
                "Budget Multiplier": multiplier,
                "Alert Budget": alerts,
                "Alerts": min(alerts, 372) if multiplier >= 1.3 else alerts,
                "Frauds Caught": frauds,
                "Recall": recall,
                "Precision": round(precision, 4),
                "Total Cost": cost,
                "Cost Diff vs Static": round(cost - static_cost, 2),
                "Recall Diff vs Static": round(recall - static_recall, 3),
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    st.dataframe(curve_df, use_container_width=True)

    best_row = curve_df.loc[curve_df["Total Cost"].idxmin()]

    st.success(
        f"Best operating point: multiplier={best_row['Budget Multiplier']} | "
        f"recall={best_row['Recall']:.3f} | "
        f"cost={best_row['Total Cost']:,.0f}"
    )

    st.line_chart(curve_df.set_index("Budget Multiplier")[["Recall", "Total Cost"]])

    st.markdown(
        """
The operating curve helps identify the point where additional alert capacity
produces meaningful fraud detection gains without unnecessary review cost.
"""
    )

with queue_tab:
    st.subheader("Analyst Investigation Queue")

    c1, c2, c3 = st.columns(3)
    c1.metric("High Severity", 113)
    c2.metric("Medium Severity", 83)
    c3.metric("Low Severity", 4)

    alerts_df = pd.DataFrame(
        [
            {
                "Transaction ID": 1818,
                "Type": "TRANSFER",
                "Amount": 2317408.88,
                "Fraud Score": 1.000,
                "Rank Score": 81.1897,
                "Expected Benefit": 2317398.88,
                "Severity": "HIGH",
                "Reason": "Very high model confidence; prioritized for immediate analyst review",
            },
            {
                "Transaction ID": 374,
                "Type": "TRANSFER",
                "Amount": 1478772.04,
                "Fraud Score": 1.000,
                "Rank Score": 70.9529,
                "Expected Benefit": 1478761.99,
                "Severity": "HIGH",
                "Reason": "Very high model confidence; prioritized for immediate analyst review",
            },
            {
                "Transaction ID": 969,
                "Type": "TRANSFER",
                "Amount": 1277212.77,
                "Fraud Score": 1.000,
                "Rank Score": 67.9014,
                "Expected Benefit": 1277202.77,
                "Severity": "HIGH",
                "Reason": "Very high model confidence; prioritized for immediate analyst review",
            },
        ]
    )

    st.dataframe(alerts_df, use_container_width=True)

    st.markdown(
        """
The analyst queue converts model outputs into reviewable operational items.
Transactions are prioritized by severity, rank score, and expected benefit.
"""
    )

with design_tab:
    st.subheader("System Design")

    st.markdown(
        """
### End-to-End Flow

1. **Raw Transaction Dataset**  
   Financial transaction data is loaded and prepared for fraud scoring.

2. **Data Loading & Preprocessing**  
   Transaction features are processed before being passed to the model.

3. **Fraud Scoring Model**  
   The machine learning model assigns a fraud risk score to each transaction.

4. **Decision Engine**  
   The decision layer converts fraud scores into operational actions.

5. **Adaptive Decision Logic**  
   The system applies risk-zone prioritization, cost-aware ranking, alert budgeting,
   and suppression logic.

6. **Analyst Queue**  
   Selected alerts are prioritized for analyst review based on severity and expected benefit.

7. **Monitoring Dashboard**  
   The dashboard tracks recall, precision, alert volume, operating curves, and cost impact.
"""
    )

    architecture_df = pd.DataFrame(
        [
            {
                "Layer": "Data Layer",
                "Component": "Raw Transaction Dataset",
                "Purpose": "Provides transaction data for scoring and evaluation",
            },
            {
                "Layer": "Model Layer",
                "Component": "Fraud Scoring Model",
                "Purpose": "Generates fraud probability scores",
            },
            {
                "Layer": "Decision Layer",
                "Component": "Decision Engine",
                "Purpose": "Applies alert selection, ranking, budgeting, and cost logic",
            },
            {
                "Layer": "Operations Layer",
                "Component": "Analyst Queue",
                "Purpose": "Prioritizes alerts for human investigation",
            },
            {
                "Layer": "Monitoring Layer",
                "Component": "Streamlit Dashboard",
                "Purpose": "Visualizes system performance and operational trade-offs",
            },
        ]
    )

    st.dataframe(architecture_df, use_container_width=True)

    st.info(
        "The key design idea is the separation between prediction and decision-making. "
        "The model estimates fraud risk, while the decision engine decides which transactions "
        "should become operational alerts."
    )