"""Training and inference utilities for the Financial Fraud Detection System."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from config import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES, METRICS_PATH,
                    MODEL_PATH, NUMERIC_FEATURES, TARGET)

HIGH_AMOUNT_THRESHOLD = 5000.0
ENGINEERED_NUMERIC = ["log_amount", "is_night", "is_high_amount", "amount_per_age"]
ENGINEERED_CATEGORICAL = ["age_band", "amount_band"]
MODEL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

class DataValidationError(ValueError):
    pass

def validate_dataframe(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataValidationError("The uploaded dataset is empty.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    if require_target:
        if TARGET not in df.columns:
            raise DataValidationError(f"Training data must contain '{TARGET}'.")
        labels = set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique())
        if not labels.issubset({0, 1}) or len(labels) != 2:
            raise DataValidationError("is_fraud must contain both binary labels 0 and 1.")
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").str.strip().str.lower().replace("", pd.NA)
        mode = out[col].mode(dropna=True)
        out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "unknown").astype(str)
    out["amount"] = out["amount"].clip(lower=0)
    out["customer_age"] = out["customer_age"].clip(15, 100)
    out["transaction_hour"] = out["transaction_hour"].clip(0, 23).round().astype(int)
    if TARGET in out:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    hour = out["transaction_hour"].astype(int)
    out["is_night"] = ((hour >= 23) | (hour <= 5)).astype(int)
    out["is_high_amount"] = (out["amount"] >= HIGH_AMOUNT_THRESHOLD).astype(int)
    out["amount_per_age"] = out["amount"] / out["customer_age"].clip(lower=1)
    out["age_band"] = pd.cut(out["customer_age"], [0,25,35,50,65,200], labels=["under25","25_35","35_50","50_65","65plus"]).astype(str)
    out["amount_band"] = pd.cut(out["amount"], [-0.01,100,500,2000,5000,np.inf], labels=["micro","small","medium","large","very_large"]).astype(str)
    return out

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_dataframe(df)
    return add_features(clean_dataframe(df))[MODEL_NUMERIC + MODEL_CATEGORICAL]

def build_pipeline() -> ImbPipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    encoder = ColumnTransformer([("num", numeric, MODEL_NUMERIC), ("cat", categorical, MODEL_CATEGORICAL)])
    return ImbPipeline([("encode", encoder), ("smote", SMOTE(random_state=42, k_neighbors=5)),
                        ("clf", LogisticRegression(C=1.0, max_iter=1500, class_weight=None, random_state=42))])

def train_model(data_path: Path = DATA_PATH) -> dict:
    raw = pd.read_csv(data_path)
    validate_dataframe(raw, require_target=True)
    cleaned = clean_dataframe(raw)
    X = add_features(cleaned)[MODEL_NUMERIC + MODEL_CATEGORICAL]
    y = cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_pipeline()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0,1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "rows": int(len(raw)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "fraud_rows": int(y.sum()), "fraud_rate": float(y.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": 0.5, "model": "Logistic Regression + SMOTE"
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics

def load_model():
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)

def predict_one(model, transaction: dict) -> dict:
    frame = prepare_features(pd.DataFrame([transaction]))
    probability = float(model.predict_proba(frame)[0, 1])
    threshold = 0.5
    risk = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.35 else "LOW"
    return {"fraud_probability": probability, "prediction": "FRAUD" if probability >= threshold else "LEGITIMATE",
            "risk_level": risk, "threshold": threshold}
