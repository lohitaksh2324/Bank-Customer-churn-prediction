import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

REQUIRED_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

@st.cache_resource
def load_pipeline():
    return joblib.load("churn_pipeline.pkl")

pipeline = load_pipeline()

st.title("Bank Customer Churn Predictor")
st.write(
    "Upload a CSV to get churn predictions. Required columns "
    f"(other columns like CustomerId/Surname/RowNumber/Exited are fine, they're ignored):"
)
st.code(", ".join(REQUIRED_COLUMNS), language=None)

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Couldn't read that file as a CSV: {e}")
        st.stop()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        st.stop()

    with st.spinner("Predicting..."):
        X = df[REQUIRED_COLUMNS]
        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)[:, 1]

    result = df.copy()
    result["Churn_Prediction"] = preds
    result["Churn_Probability"] = probs.round(4)

    st.success(f"Done — {int(preds.sum())} of {len(preds)} customers predicted to churn.")
    st.dataframe(result, use_container_width=True)

    st.download_button(
        "Download results as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )
