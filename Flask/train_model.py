"""
SmartLender - Model Training Script
------------------------------------
Loads the loan applicant dataset, cleans/encodes it, trains four classifiers
(Decision Tree, Random Forest, KNN, XGBoost), evaluates them, and saves the
best-performing model + preprocessing artifacts for use by the Flask app.

Usage:
    python train_model.py
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed - run `pip install xgboost` to include it (see requirements.txt)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

CATEGORICAL_COLS = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
FEATURE_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                 "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                 "Loan_Amount_Term", "Credit_History", "Property_Area"]
TARGET_COL = "Loan_Status"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Dependents: "3+" -> 3, then treat as numeric
    df["Dependents"] = df["Dependents"].replace("3+", 3)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    # Fill missing categorical values with the mode
    for col in CATEGORICAL_COLS + ["Dependents"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fill missing numeric values with the median
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_features(df: pd.DataFrame, encoders: dict | None = None):
    """Label-encode categorical columns. If encoders is None, fit new ones."""
    df = df.copy()
    fit_new = encoders is None
    if fit_new:
        encoders = {}

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        if col not in df.columns:
            continue
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df)} rows after cleaning.")

    df_encoded, encoders = encode_features(df)

    X = df_encoded[FEATURE_COLS]
    y = df_encoded[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )

    results = {}
    best_name, best_model, best_test_acc = None, None, -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"train_accuracy": round(train_acc * 100, 2),
                          "test_accuracy": round(test_acc * 100, 2)}
        print(f"{name:15s} | train: {train_acc*100:6.2f}% | test: {test_acc*100:6.2f}%")

        if test_acc > best_test_acc:
            best_name, best_model, best_test_acc = name, model, test_acc

    print(f"\nBest model: {best_name} (test accuracy {best_test_acc*100:.2f}%)")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results, "feature_cols": FEATURE_COLS}, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved encoders -> {ENCODERS_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
