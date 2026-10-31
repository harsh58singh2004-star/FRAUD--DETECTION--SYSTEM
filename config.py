from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "transactions.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_CONTENT_LENGTH = 10 * 1024 * 1024
SECRET_KEY = "change-this-before-deployment"

NUMERIC_FEATURES = ["amount", "customer_age", "transaction_hour"]
CATEGORICAL_FEATURES = ["transaction_type", "merchant_category", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"
TRANSACTION_TYPES = ["purchase", "transfer", "withdrawal", "payment", "refund"]
MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "entertainment", "fuel", "healthcare", "restaurant", "online_services"]
PAYMENT_METHODS = ["credit_card", "debit_card", "upi", "net_banking", "wallet"]
