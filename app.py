import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
RESULTS_PATH = "results.pkl"

FEATURE_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "person_home_ownership",
    "loan_intent", "loan_grade", "cb_person_default_on_file"
]

FEATURE_LABELS = {
    "person_age": "Age",
    "person_income": "Annual Income ($)",
    "person_emp_length": "Employment Length (years)",
    "loan_amnt": "Loan Amount ($)",
    "loan_int_rate": "Interest Rate (%)",
    "loan_percent_income": "Loan % of Income",
    "cb_person_cred_hist_length": "Credit History Length (years)",
    "person_home_ownership": "Home Ownership (0=RENT, 1=OWN, 2=MORTGAGE)",
    "loan_intent": "Loan Intent (0=PERSONAL,1=EDUCATION,2=MEDICAL,3=VENTURE,4=HOMEIMPROVE,5=DEBTCONSOLIDATION)",
    "loan_grade": "Loan Grade (0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G)",
    "cb_person_default_on_file": "Previous Default (0=No, 1=Yes)"
}


def load_data():
    path = "dataset/loan.csv"
    if not os.path.exists(path):
        urls = [
            "https://raw.githubusercontent.com/dsrscientist/dataset1/master/loan_prediction.csv",
            "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
        ]
        for url in urls:
            try:
                df = pd.read_csv(url)
                if len(df) > 100:
                    df.to_csv(path, index=False)
                    print(f"Dataset downloaded from {url}")
                    return df, "simple"
            except Exception:
                continue

        # Use Credit Risk dataset from GitHub
        url = "https://raw.githubusercontent.com/monicazng/credit-risk-classification/main/Resources/lending_data.csv"
        try:
            df = pd.read_csv(url)
            df.to_csv(path, index=False)
            return df, "lending"
        except Exception:
            pass

        return None, None

    df = pd.read_csv(path)
    return df, "loaded"


def preprocess_credit_risk(df):
    """Handle the lending_data.csv format."""
    # lending_data.csv has: loan_size, interest_rate, borrower_income, debt_to_income,
    # num_of_accounts, derogatory_marks, total_debt, loan_status
    feature_cols = ["loan_size", "interest_rate", "borrower_income",
                    "debt_to_income", "num_of_accounts", "derogatory_marks", "total_debt"]
    available = [c for c in feature_cols if c in df.columns]
    target_col = None
    for c in ["loan_status", "default", "Loan_Status"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None or not available:
        return None, None, None
    X = df[available]
    y = df[target_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X, y, available


def preprocess_simple(df):
    """Handle simpler loan CSV formats."""
    # Try to find target column
    target_candidates = ["loan_status", "Loan_Status", "default", "Default", "target"]
    target_col = None
    for c in target_candidates:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        return None, None, None

    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        if col != target_col:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    if df[target_col].dtype == object:
        df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

    feature_cols = [c for c in df.columns if c != target_col]
    df = df.dropna()
    return df[feature_cols], df[target_col], feature_cols


def train_models():
    df, fmt = load_data()
    if df is None:
        return None

    if fmt == "lending":
        X, y, feature_cols = preprocess_credit_risk(df)
    else:
        X, y, feature_cols = preprocess_simple(df)

    if X is None:
        return None

    X = X.fillna(X.median(numeric_only=True))
    y = pd.Series(y).fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    results = []
    best_auc = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
        cv = round(cross_val_score(model, X_train_sc, y_train, cv=5, scoring="roc_auc").mean() * 100, 2)
        results.append({"model": name, "accuracy": acc, "auc": auc, "cv_auc": cv})
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    # Feature importance from best tree model
    rf = models.get("Random Forest")
    feat_imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)

    with open(MODEL_PATH, "wb") as f: pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    default_rate = round(float(pd.Series(y).mean()) * 100, 1)

    all_results = {
        "model_results": sorted(results, key=lambda x: x["auc"], reverse=True),
        "feature_importance": feat_imp[:10],
        "feature_cols": feature_cols,
        "best_model": best_name,
        "dataset_size": len(df),
        "default_rate": default_rate
    }
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_results, f)

    print("Training complete.")
    for r in results:
        print(f"  {r['model']}: Acc={r['accuracy']}% AUC={r['auc']}%")
    return all_results


if os.path.exists(MODEL_PATH) and os.path.exists(RESULTS_PATH):
    print("Loading cached model...")
    with open(RESULTS_PATH, "rb") as f:
        RESULTS = pickle.load(f)
else:
    print("Training models...")
    RESULTS = train_models()

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html", results=RESULTS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    feature_cols = RESULTS["feature_cols"]
    try:
        features = [float(data.get(f, 0)) for f in feature_cols]
        X = np.array(features).reshape(1, -1)
        X_sc = SCALER.transform(X)
        pred = int(MODEL.predict(X_sc)[0])
        prob = round(float(MODEL.predict_proba(X_sc)[0][1]) * 100, 1)
        return jsonify({
            "prediction": pred,
            "label": "Default Risk" if pred == 1 else "Low Risk",
            "probability": prob,
            "risk": "High" if prob >= 60 else "Medium" if prob >= 35 else "Low"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5015)
