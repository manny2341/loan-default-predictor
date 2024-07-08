# 💳 Loan Default Predictor

A machine learning web app that predicts whether a loan applicant is likely to default on their loan. Compares four classification models using ROC-AUC and 5-fold cross-validation, and shows which financial factors drive default risk through feature importance analysis.

## Demo

Enter applicant details → AI evaluates credit risk → Instant prediction with default probability and Low / Medium / High risk rating.

## Results

| Model | Accuracy | ROC-AUC | CV AUC |
|-------|----------|---------|--------|
| Logistic Regression | ~85% | ~93% | ~92% |
| Random Forest | ~99% | ~99% | ~99% |
| Gradient Boosting | ~99% | ~99% | ~99% |
| **XGBoost ⭐** | **~99%** | **~99%** | **~99%** |

## Features

- 4 models compared: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- ROC-AUC evaluation — the right metric for imbalanced credit risk data
- 5-fold cross-validation to verify model generalises beyond training data
- Feature importance chart showing exactly what drives default risk
- Interactive prediction form — enter any applicant's details and get a risk score
- Dataset auto-downloads on first run — no manual setup needed

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Scikit-Learn, XGBoost |
| Preprocessing | StandardScaler |
| Evaluation | ROC-AUC, 5-Fold Cross-Validation |
| Web Framework | Flask |
| Dataset | Credit Risk / Lending Data |
| Language | Python |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/loan-default-predictor.git
cd loan-default-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the app**
```bash
python3 app.py
```
The dataset downloads automatically on first run.

**4. Open in browser**
```
http://127.0.0.1:5015
```

## How It Works

1. Applicant financial data entered via the prediction form
2. All features scaled with **StandardScaler**
3. **XGBoost** model predicts the probability of default
4. Risk level classified as **Low** (<35%), **Medium** (35–60%), or **High** (>60%)
5. Feature importance from Random Forest shows what matters most (e.g. debt-to-income ratio, interest rate)

## Why ROC-AUC Over Accuracy?

Credit risk datasets are imbalanced — most applicants don't default. A model that always predicts "no default" still scores high accuracy but is useless. ROC-AUC measures how well the model separates defaulters from non-defaulters across all thresholds, making it the correct metric for this problem.

## Project Structure

```
loan-default-predictor/
├── app.py               # Flask server, model training, feature importance, prediction API
├── dataset/
│   └── loan.csv         # Auto-downloaded on first run
├── templates/
│   └── index.html       # Model comparison, feature importance chart, prediction form
├── static/
│   └── style.css        # Dark theme styling
└── requirements.txt
```

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Diabetes Classifier | Model comparison + feature scaling demo | [diabetes-classifier](https://github.com/manny2341/diabetes-classifier) |
| Heart Attack Predictor | End-to-end classification pipeline | [heart-attack-predictor](https://github.com/manny2341/heart-attack-predictor) |
| Medical Cost Predictor | Regression — what drives healthcare costs | [medical-cost-predictor](https://github.com/manny2341/medical-cost-predictor) |
| Stock Price Predictor | LSTM — 5,884 tickers + crypto | [stock-price-predictor](https://github.com/manny2341/stock-price-predictor) |

## Author

[@manny2341](https://github.com/manny2341)
