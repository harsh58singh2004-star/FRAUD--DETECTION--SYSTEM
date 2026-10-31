from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
import streamlit as st

from config import (
    DATA_PATH,
    MAX_CONTENT_LENGTH,
    MERCHANT_CATEGORIES,
    METRICS_PATH,
    PAYMENT_METHODS,
    TRANSACTION_TYPES,
)
from model import load_model, predict_one, prepare_features, validate_dataframe


st.set_page_config(
    page_title="FraudGuard | Transaction Intelligence",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e8ebf2;}
    .eyebrow {color: #6878e8; font-size: .75rem; font-weight: 700;
              letter-spacing: .12em; text-transform: uppercase;}
    .disclaimer {color: #697386; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


def read_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not read model evaluation metrics: {exc}")
        return {}


def show_dashboard() -> None:
    st.markdown('<div class="eyebrow">Transaction monitoring</div>', unsafe_allow_html=True)
    st.title("Fraud intelligence overview")
    st.write("Machine-learning assisted screening across the supplied transaction dataset.")

    try:
        data = pd.read_csv(DATA_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not load the transaction dataset at {DATA_PATH}: {exc}")
        return

    required = {"is_fraud", "transaction_type", "transaction_hour"}
    missing = required.difference(data.columns)
    if missing:
        st.error("The dashboard dataset is missing required columns: " + ", ".join(sorted(missing)))
        return

    total = len(data)
    fraud = int(pd.to_numeric(data["is_fraud"], errors="coerce").fillna(0).sum())
    legitimate = total - fraud
    fraud_rate = fraud / total * 100 if total else 0

    stat_columns = st.columns(4)
    stat_columns[0].metric("Total transactions", f"{total:,}")
    stat_columns[1].metric("Flagged fraud cases", f"{fraud:,}")
    stat_columns[2].metric("Legitimate transactions", f"{legitimate:,}")
    stat_columns[3].metric("Dataset fraud rate", f"{fraud_rate:.2f}%")

    metrics = read_metrics()
    st.subheader("Model performance")
    st.caption("Held-out test set · classification threshold 0.50")
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "accuracy": "Accuracy",
    }
    metric_columns = st.columns(len(labels))
    for column, (key, label) in zip(metric_columns, labels.items()):
        value = metrics.get(key)
        column.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "—")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Fraud by transaction type")
        by_type = (
            data.groupby("transaction_type")["is_fraud"].mean().mul(100).round(2)
        )
        st.bar_chart(by_type, y_label="Fraud rate (%)")
    with chart_columns[1]:
        st.subheader("Hourly fraud pattern")
        by_hour = (
            data.groupby("transaction_hour")["is_fraud"].mean().mul(100).round(2)
            .reindex(range(24), fill_value=0)
        )
        st.line_chart(by_hour, y_label="Fraud rate (%)")

    st.subheader("Model snapshot")
    snapshot_columns = st.columns(3)
    snapshot_columns[0].write("**Algorithm**\n\nLogistic Regression")
    snapshot_columns[1].write("**Imbalance handling**\n\nSMOTE on training data")
    snapshot_columns[2].write(
        f"**Training / test rows**\n\n{metrics.get('train_rows', '—')} / "
        f"{metrics.get('test_rows', '—')}"
    )

    st.subheader("Recent dataset records")
    st.caption("Latest rows in the CSV, not live transactions.")
    display_columns = [
        column for column in (
            "transaction_id", "amount", "transaction_type",
            "merchant_category", "transaction_hour", "is_fraud",
        ) if column in data.columns
    ]
    recent = data.loc[:, display_columns].tail(8).iloc[::-1].copy()
    if "is_fraud" in recent:
        recent["is_fraud"] = recent["is_fraud"].map(
            {0: "LEGITIMATE", 1: "FRAUD"}
        ).fillna("UNKNOWN")
    st.dataframe(recent, use_container_width=True, hide_index=True)


def show_single_prediction(model) -> None:
    st.markdown('<div class="eyebrow">Scoring</div>', unsafe_allow_html=True)
    st.title("Score a transaction")
    st.write(
        "Enter transaction attributes to estimate the model's fraud probability. "
        "Use realistic values for a meaningful demonstration."
    )

    with st.form("single_prediction"):
        first_row = st.columns(3)
        amount = first_row[0].number_input(
            "Transaction amount (₹)", min_value=0.0, value=1250.50, step=100.0
        )
        customer_age = first_row[1].number_input(
            "Customer age", min_value=15, max_value=100, value=34
        )
        transaction_hour = first_row[2].selectbox(
            "Transaction hour", options=list(range(24)), format_func=lambda hour: f"{hour:02d}:00"
        )

        second_row = st.columns(3)
        transaction_type = second_row[0].selectbox("Transaction type", TRANSACTION_TYPES)
        merchant_category = second_row[1].selectbox(
            "Merchant category", MERCHANT_CATEGORIES
        )
        payment_method = second_row[2].selectbox("Payment method", PAYMENT_METHODS)
        submitted = st.form_submit_button("Analyze transaction", type="primary")

    if submitted:
        transaction = {
            "amount": amount,
            "customer_age": customer_age,
            "transaction_hour": transaction_hour,
            "transaction_type": transaction_type,
            "merchant_category": merchant_category,
            "payment_method": payment_method,
        }
        try:
            result = predict_one(model, transaction)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Could not score transaction: {exc}")
            return

        result_columns = st.columns(3)
        result_columns[0].metric("Prediction", result["prediction"])
        result_columns[1].metric("Risk classification", result["risk_level"])
        result_columns[2].metric(
            "Fraud probability", f"{result['fraud_probability']:.1%}"
        )
        st.progress(result["fraud_probability"])
        st.caption(
            f"Classification threshold: {result['threshold']:.2f}. "
            "This score is an estimate from a synthetic academic dataset, not "
            "proof of fraud or a real financial decision."
        )

    with st.expander("How scoring works"):
        st.markdown(
            """
            1. Inputs are validated, cleaned, and transformed into model features.
            2. The trained classifier estimates the probability of fraud.
            3. Treat the output as a screening signal, not a final verdict.
            """
        )


def show_batch_analysis(model) -> None:
    st.markdown('<div class="eyebrow">Batch analysis</div>', unsafe_allow_html=True)
    st.title("Analyze transaction batches")
    st.write(
        "Upload a CSV file to score every row. The output includes fraud "
        "probability, predicted class, and risk band."
    )
    st.caption(
        "Required columns: amount, transaction_type, merchant_category, "
        "customer_age, payment_method, transaction_hour. Optional: "
        "transaction_id, is_fraud."
    )

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is None:
        return
    if uploaded_file.size > MAX_CONTENT_LENGTH:
        st.error("Upload too large. Maximum file size is 10 MB.")
        return

    try:
        frame = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        validate_dataframe(frame)
        features = prepare_features(frame)
        probabilities = model.predict_proba(features)[:, 1]
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
        st.error(f"CSV processing failed: {exc}")
        return

    output = frame.copy()
    output["fraud_probability"] = probabilities.round(4)
    output["prediction"] = pd.Series(
        (probabilities >= 0.5).astype(int), index=output.index
    ).map({0: "LEGITIMATE", 1: "FRAUD"})
    output["risk_level"] = pd.cut(
        probabilities,
        [-0.001, 0.35, 0.70, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)

    st.success(f"Scored {len(output):,} transactions.")
    st.subheader("Prediction preview")
    st.caption("Showing up to 100 records; download the full output below.")
    st.dataframe(output.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download complete results",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    with st.sidebar:
        st.title("◈ FraudGuard")
        st.caption("TRANSACTION INTELLIGENCE")
        page = st.radio(
            "Navigation",
            ["Overview", "Single prediction", "Batch analysis"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("MODEL ONLINE")
        st.caption("Logistic Regression · SMOTE")

    st.markdown(
        '<div class="eyebrow">RISK MONITORING</div>', unsafe_allow_html=True
    )
    try:
        model = get_model()
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not load the fraud detection model: {exc}")
        st.stop()

    if page == "Overview":
        show_dashboard()
    elif page == "Single prediction":
        show_single_prediction(model)
    else:
        show_batch_analysis(model)

    st.divider()
    st.markdown(
        '<p class="disclaimer">FraudGuard · Academic demonstration system · '
        "Not a production banking decision engine</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
