"""
SmartLender - Flask Web Application
------------------------------------
Serves a form for loan applicant details and returns a real-time
approval prediction using the trained model.

Usage:
    python app.py
Then visit http://127.0.0.1:5000
"""
import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_model import FEATURE_COLS, CATEGORICAL_COLS

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
ENCODERS_PATH = "model/encoders.pkl"

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)


def build_feature_row(form) -> pd.DataFrame:
    """Turn submitted form data into a single-row DataFrame matching training features."""
    dependents = form.get("Dependents", "0")
    dependents = 3 if dependents == "3+" else int(dependents)

    row = {
        "Gender": form.get("Gender"),
        "Married": form.get("Married"),
        "Dependents": dependents,
        "Education": form.get("Education"),
        "Self_Employed": form.get("Self_Employed"),
        "ApplicantIncome": float(form.get("ApplicantIncome", 0)),
        "CoapplicantIncome": float(form.get("CoapplicantIncome", 0)),
        "LoanAmount": float(form.get("LoanAmount", 0)),
        "Loan_Amount_Term": float(form.get("Loan_Amount_Term", 360)),
        "Credit_History": float(form.get("Credit_History", 1)),
        "Property_Area": form.get("Property_Area"),
    }
    df = pd.DataFrame([row])

    # Apply the same label encoders used during training
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    return df[FEATURE_COLS]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    X = build_feature_row(request.form)
    pred_encoded = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(max(model.predict_proba(X)[0]) * 100, 1)

    status_encoder = encoders["Loan_Status"]
    pred_label = status_encoder.inverse_transform([pred_encoded])[0]
    approved = pred_label == "Y"

    return render_template("result.html", approved=approved, confidence=proba)


if __name__ == "__main__":
    app.run(debug=True)
