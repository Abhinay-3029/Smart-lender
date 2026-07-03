"""
Generates a SMALL SYNTHETIC sample of loan applicant data, matching the exact
column schema of the real 'Loan Prediction Problem Dataset' (Kaggle), purely
so you can test-run the training pipeline before plugging in the real data.

DO NOT use this synthetic data for your actual project results/accuracy
numbers -- it's just a schema-compatible stand-in for local testing.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "..", "Dataset", "loan_data.csv")

rng = np.random.default_rng(42)
n = 200

genders = rng.choice(["Male", "Female"], n, p=[0.8, 0.2])
married = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.56, 0.17, 0.17, 0.10])
education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
self_employed = rng.choice(["Yes", "No"], n, p=[0.14, 0.86])
applicant_income = rng.integers(1500, 20000, n)
coapplicant_income = rng.integers(0, 8000, n)
loan_amount = rng.integers(20, 400, n)
loan_term = rng.choice([360, 180, 120, 60], n, p=[0.85, 0.08, 0.04, 0.03])
credit_history = rng.choice([1.0, 0.0], n, p=[0.84, 0.16])
property_area = rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.33, 0.38, 0.29])

# Rough approval logic so the sample data isn't pure noise
approve_prob = (
    0.55
    + 0.25 * (credit_history == 1.0)
    + 0.05 * (education == "Graduate")
    - 0.10 * (applicant_income + coapplicant_income < 3000)
)
approve_prob = np.clip(approve_prob, 0.05, 0.95)
loan_status = np.where(rng.random(n) < approve_prob, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{100000+i}" for i in range(n)],
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

# sprinkle a few missing values, like the real dataset has
for col in ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    idx = rng.choice(n, size=int(n * 0.03) + 1, replace=False)
    df.loc[idx, col] = np.nan

df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows to {OUT_PATH}")
