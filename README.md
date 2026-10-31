# Fraud Detection System

A Streamlit application for exploring the transaction dataset, scoring an
individual transaction, and running batch predictions from a CSV file.

## Run the application

Install the project dependencies and start Streamlit from the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app trains and saves the model on first launch if no trained model is
available. Use the sidebar to switch between the overview, single prediction,
and batch analysis pages.

## Batch CSV format

CSV uploads may be up to 10 MB. The following columns are required:

`amount, transaction_type, merchant_category, customer_age, payment_method, transaction_hour`

`transaction_id` and `is_fraud` are optional. Labels in `is_fraud` are not used
to generate predictions. The output CSV includes fraud probability, prediction,
and risk level.

This project is an academic demonstration and is not intended for production
financial decision-making.
