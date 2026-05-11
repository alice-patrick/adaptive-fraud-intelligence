import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection App", page_icon="🚨", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title("Fraud Detection Prediction App")
st.markdown("Enter the transaction details to predict whether the transaction is fraudulent or legitimate.")
st.divider()

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0, step=100.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0, step=100.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
            st.write(f"Fraud probability: {probability:.2%}")

        st.subheader(f"Prediction: {int(prediction)}")

        if int(prediction) == 1:
            st.error("The transaction is predicted to be fraudulent.")
        else:
            st.success("The transaction is predicted to be legitimate.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input data:")
        st.dataframe(input_data)