from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
from config import (DATA_PATH, METRICS_PATH, UPLOAD_DIR, MAX_CONTENT_LENGTH,
                    TRANSACTION_TYPES, MERCHANT_CATEGORIES, PAYMENT_METHODS)
from model import DataValidationError, load_model, predict_one, prepare_features, validate_dataframe

app = Flask(__name__)
app.config.from_object("config")
app.secret_key = app.config["SECRET_KEY"]
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(exist_ok=True)
model = load_model()

def get_metrics():
    if METRICS_PATH.exists(): return json.loads(METRICS_PATH.read_text())
    return {}

@app.route("/")
def dashboard():
    metrics = get_metrics()
    try:
        data = pd.read_csv(DATA_PATH)
        total = len(data); fraud = int(data["is_fraud"].sum()); legitimate = total - fraud
        fraud_rate = fraud / total * 100 if total else 0
        by_type = (data.groupby("transaction_type")["is_fraud"].mean() * 100).round(2).to_dict()
        by_hour = (data.groupby("transaction_hour")["is_fraud"].mean() * 100).round(2).to_dict()
        recent = data.tail(8).iloc[::-1].fillna("").to_dict("records")
    except Exception:
        total=fraud=legitimate=0; fraud_rate=0; by_type={}; by_hour={}; recent=[]
    return render_template("dashboard.html", metrics=metrics, total=total, fraud=fraud,
        legitimate=legitimate, fraud_rate=round(fraud_rate,2), by_type=by_type,
        by_hour=by_hour, recent=recent)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            txn = {"amount": request.form.get("amount"), "customer_age": request.form.get("customer_age"),
                "transaction_hour": request.form.get("transaction_hour"),
                "transaction_type": request.form.get("transaction_type"),
                "merchant_category": request.form.get("merchant_category"),
                "payment_method": request.form.get("payment_method")}
            result = predict_one(model, txn)
        except Exception as exc:
            flash(f"Could not score transaction: {exc}", "error")
    return render_template("predict.html", result=result, types=TRANSACTION_TYPES,
        merchants=MERCHANT_CATEGORIES, methods=PAYMENT_METHODS)

@app.route("/batch", methods=["GET", "POST"])
def batch():
    results = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Choose a CSV file first.", "error")
        elif Path(secure_filename(file.filename)).suffix.lower() != ".csv":
            flash("Only CSV files are supported.", "error")
        else:
            try:
                frame = pd.read_csv(file)
                validate_dataframe(frame)
                features = prepare_features(frame)
                probs = model.predict_proba(features)[:,1]
                output = frame.copy()
                output["fraud_probability"] = probs.round(4)
                output["prediction"] = (probs >= 0.5).astype(int)
                output["prediction"] = output["prediction"].map({0:"LEGITIMATE",1:"FRAUD"})
                output["risk_level"] = pd.cut(probs, [-0.001,0.35,0.70,1.001], labels=["LOW","MEDIUM","HIGH"]).astype(str)
                results = output.head(100).to_dict("records")
                out_path = UPLOAD_DIR / "batch_predictions.csv"
                output.to_csv(out_path, index=False)
                flash(f"Scored {len(output):,} transactions. Download the complete result below.", "success")
            except Exception as exc:
                flash(f"CSV processing failed: {exc}", "error")
    return render_template("batch.html", results=results)

@app.route("/download-results")
def download_results():
    from flask import send_file
    path = UPLOAD_DIR / "batch_predictions.csv"
    if not path.exists():
        flash("No batch result is available yet.", "error"); return redirect(url_for("batch"))
    return send_file(path, as_attachment=True, download_name="fraud_predictions.csv")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_one(model, payload)
        return jsonify({"success": True, **result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/health")
def health():
    return jsonify({"status":"ok", "model_loaded": model is not None})

@app.errorhandler(413)
def too_large(_):
    return render_template("error.html", message="Upload too large. Maximum file size is 10 MB."), 413

if __name__ == "__main__":
    app.run(debug=True)
