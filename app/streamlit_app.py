import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Adaptive Fraud Intelligence",
    layout="wide",
)

st.title("Adaptive Fraud Intelligence")
st.caption("Decision-centric fraud detection with adaptive alerting and cost-aware ranking")

st.subheader("Main Result")

df = pd.DataFrame(
    [
        {
            "System": "Static Threshold",
            "Recall": 0.625,
            "Precision": 0.033,
            "Alerts": 303,
            "Total Cost": 600060,
        },
        {
            "System": "Decision System",
            "Recall": 0.875,
            "Precision": 0.038,
            "Alerts": 372,
            "Total Cost": 552563,
        },
    ]
)

st.dataframe(df, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recall Improvement", "+0.250")
col2.metric("Cost Reduction", "47,497")
col3.metric("Alert Increase", "+69")
col4.metric("Best System", "Decision System")

st.subheader("Dashboard Preview")

st.image("docs/images/dashboard_overview.png", caption="Static threshold vs decision system")
st.image("docs/images/operating_curve.png", caption="Operating curve analysis")
st.image("docs/images/analyst_queue.png", caption="Analyst investigation queue")

st.subheader("System Summary")

st.markdown(
    """
This project separates the predictive layer from the decision layer.

The machine learning model produces fraud risk scores.  
The decision engine converts those scores into operational actions using:

- adaptive alert budgets
- cost-aware ranking
- risk-zone prioritization
- analyst queue generation
- monitoring metrics
"""
)