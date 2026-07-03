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

Save the CSV as `data/loan_data.csv` (replacing the synthetic sample that's
there now — that sample only exists so this pipeline runs out of the box for
testing; don't report its accuracy numbers as your project results).

## 3. Train the models

```bash
python train_model.py
```

This trains Decision Tree, Random Forest, KNN, and XGBoost, prints train/test
accuracy for each, and saves the best one to `model/model.pkl` (along with
`model/encoders.pkl` for consistent preprocessing and `model/metrics.json`
for your report/documentation).

## 4. Run the web app

```bash
python app.py
```

Visit **http://127.0.0.1:5000** — fill in the form, get an instant
Approved / Not Approved prediction with confidence score.

## Project structure

```
SmartLender/
├── data/
│   ├── loan_data.csv        <- replace with the real Kaggle dataset
│   └── make_sample_data.py  <- generates the placeholder sample
├── model/
│   ├── model.pkl             <- best trained model (created by train_model.py)
│   ├── encoders.pkl          <- label encoders (created by train_model.py)
│   └── metrics.json          <- accuracy comparison across all 4 models
├── templates/
│   ├── index.html            <- application form
│   └── result.html           <- prediction result
├── static/
│   └── style.css
├── train_model.py
├── app.py
└── requirements.txt
```

## Next steps for your project phases

- **Testing phase**: add a `tests/` folder with a few `pytest` cases that
  hit `/predict` with known inputs and assert the response status/content.
- **Deployment (IBM Cloud)**: your README mentions IBM Cloud — once this runs
  locally, we can containerize it (Dockerfile) or wire up IBM Cloud Foundry /
  Code Engine deployment next.
- **Documentation**: `model/metrics.json` gives you ready numbers to drop
  into your Project_Documentation folder.
