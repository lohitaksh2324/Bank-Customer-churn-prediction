import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

REQUIRED_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

# Expected format for each feature: dtype family + allowed values / ranges.
# These come directly from what the fitted pipeline was trained on (see the
# OneHotEncoder categories for Geography/Gender, and sane domain ranges for
# the rest).
NUMERIC_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"
]
BINARY_COLUMNS = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_ALLOWED = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
}


@st.cache_resource
def load_pipeline():
    return joblib.load("churn_pipeline.pkl")


pipeline = load_pipeline()

st.title("Bank Customer Churn Predictor")
st.write(
    "Upload a CSV to get churn predictions. Required columns "
    "(other columns like CustomerId/Surname/RowNumber are fine, they're ignored):"
)
st.code(", ".join(REQUIRED_COLUMNS), language=None)

with st.expander("Expected format for each column"):
    st.markdown(
        """
| Column | Expected format |
|---|---|
| CreditScore | numeric |
| Geography | one of: France, Germany, Spain |
| Gender | one of: Female, Male |
| Age | numeric |
| Tenure | numeric |
| Balance | numeric |
| NumOfProducts | numeric |
| HasCrCard | 0 or 1 |
| IsActiveMember | 0 or 1 |
| EstimatedSalary | numeric |

If you also include an **`Exited`** column (0/1, the true churn label),
the app will additionally show accuracy and a confusion matrix.
"""
    )

file = st.file_uploader("Upload CSV", type=["csv"])


def validate_format(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable format problems found in df. Empty = OK."""
    problems = []

    # 1. Missing columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {', '.join(missing)}")
        # Can't check formats of columns that don't exist
        return problems

    # 2. Numeric columns must actually be numeric
    for col in NUMERIC_COLUMNS:
        non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        if non_numeric.any():
            bad_rows = df.index[non_numeric].tolist()[:5]
            problems.append(
                f"Column '{col}' has non-numeric value(s), e.g. row(s) {bad_rows}"
            )
        if df[col].isna().any():
            problems.append(f"Column '{col}' has missing value(s)")

    # 3. Binary columns must be 0/1
    for col in BINARY_COLUMNS:
        vals = pd.to_numeric(df[col], errors="coerce")
        invalid = ~vals.isin([0, 1])
        if invalid.any():
            bad_vals = df.loc[invalid, col].unique().tolist()[:5]
            problems.append(
                f"Column '{col}' must be 0 or 1, found value(s): {bad_vals}"
            )

    # 4. Categorical columns must match known categories
    for col, allowed in CATEGORICAL_ALLOWED.items():
        bad_vals = set(df[col].dropna().unique()) - set(allowed)
        if bad_vals:
            problems.append(
                f"Column '{col}' has unexpected value(s) {sorted(bad_vals)}; "
                f"expected one of {allowed}"
            )

    return problems


if file is not None:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Couldn't read that file as a CSV: {e}")
        st.stop()

    problems = validate_format(df)
    if problems:
        st.error("This file doesn't match the expected format:")
        for p in problems:
            st.write(f"- {p}")
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

    # If ground-truth labels are present, show accuracy + confusion matrix
    if "Exited" in df.columns and df["Exited"].notna().all():
        y_true = pd.to_numeric(df["Exited"], errors="coerce")
        if y_true.isna().any() or not y_true.isin([0, 1]).all():
            st.warning(
                "Found an 'Exited' column but it isn't clean 0/1 values, "
                "so accuracy/confusion matrix can't be computed."
            )
        else:
            y_true = y_true.astype(int)
            acc = accuracy_score(y_true, preds)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

            st.subheader("Model Evaluation (using your 'Exited' column as ground truth)")
            st.metric("Accuracy", f"{acc:.2%}")

            c1, c2 = st.columns(2)
            with c1:
                st.write("**True Positives (TP)**", tp)
                st.write("**True Negatives (TN)**", tn)
            with c2:
                st.write("**False Positives (FP)**", fp)
                st.write("**False Negatives (FN)**", fn)

            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual: No Churn (0)", "Actual: Churn (1)"],
                columns=["Predicted: No Churn (0)", "Predicted: Churn (1)"],
            )
            st.write("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)
