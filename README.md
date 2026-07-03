# 🏦 SmartLender — starter pipeline

A working end-to-end starting point: data cleaning → train 4 models →
auto-pick the best → Flask app for real-time predictions.

## 1. Set up

```bash
pip install -r requirements.txt
```

## 2. Get the real dataset

This code expects the classic **Loan Prediction Problem Dataset** (614 rows) —
it's the same schema your project README describes (Gender, Married,
Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,
LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).

Download it from Kaggle:
**https://www.kaggle.com/datasets/ninzaami/loan-predication**

Save the CSV as `Dataset/loan_data.csv` (replacing the synthetic sample that's
there now — that sample only exists so this pipeline runs out of the box for
testing; don't report its accuracy numbers as your project results).

## 3. Train the models

```bash
cd Training
python train_model.py
```

This trains Decision Tree, Random Forest, KNN, and XGBoost, prints train/test
accuracy for each, and saves the best one straight into `Flask/model.pkl`
(along with `Flask/encoders.pkl` for consistent preprocessing and
`Flask/metrics.json` for your report/documentation).

## 4. Run the web app

```bash
cd Flask
python app.py
```

Visit **https://smart-loan-lender.onrender.com** — fill in the form, get an instant
Approved / Not Approved prediction with confidence score.

## Project structure

```
SmartLender/
├── Dataset/
│   └── loan_data.csv          <- replace with the real Kaggle dataset
├── Training/
│   ├── make_sample_data.py    <- generates the placeholder sample
│   └── train_model.py         <- trains all 4 models, saves the best into Flask/
├── Flask/
│   ├── app.py
│   ├── train_model.py         <- copy of Training/train_model.py (app.py imports FEATURE_COLS/CATEGORICAL_COLS from it)
│   ├── model.pkl              <- best trained model
│   ├── encoders.pkl           <- label encoders
│   ├── metrics.json           <- accuracy comparison across all 4 models
│   ├── templates/
│   │   ├── index.html         <- application form
│   │   └── result.html        <- prediction result
│   └── static/
│       └── style.css
├── wsgi.py                    <- entry point for deployment (gunicorn wsgi:app)
├── Procfile
└── requirements.txt
```

## 5. Deploy

`requirements.txt` already includes `gunicorn`. From the repo root:

```bash
gunicorn wsgi:app
```

`wsgi.py` points at the app inside `Flask/`, so this works the same way
locally and on a host like Render/Railway (which will run this same
command from the `Procfile`).


