import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Dashboard", page_icon="📊", layout="wide")
st.title("Breast Cancer Classification Dashboard")
st.caption("Evaluate five pre-trained classification models on an uploaded labelled test dataset.")

MODEL_DIR = "model"
TARGET = "diagnosis"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_artifacts():
    feature_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            "model/feature_names.joblib is missing. Run train_models.py first "
            "and upload the complete model/ folder to GitHub."
        )
    features = joblib.load(feature_path)
    models = {}
    for display_name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing trained model: {path}")
        models[display_name] = joblib.load(path)
    return features, models

def prepare_test_data(df, feature_names):
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == "id"], errors="ignore")

    if TARGET not in df.columns:
        raise ValueError("The uploaded CSV must contain the target column 'diagnosis' (M/B or 0/1).")

    y_raw = df.pop(TARGET)
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.strip().str.upper().map({"M": 1, "B": 0})
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target values must be M/B or 0/1, where 1 = Malignant and 0 = Benign.")

    missing = [col for col in feature_names if col not in df.columns]
    extra = [col for col in df.columns if col not in feature_names]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    if extra:
        st.warning("Extra columns will be ignored: " + ", ".join(extra))

    X = df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Some feature values are missing or non-numeric. Please clean the CSV and upload again.")
    return X, y.astype(int)

try:
    feature_names, models = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("Expected repository structure: app.py, requirements.txt, and model/ containing all five .joblib files plus feature_names.joblib.")
    st.stop()

uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
st.caption("The CSV must contain all 30 input features and a 'diagnosis' target column. It is evaluated only; models are not retrained.")

if uploaded_file is None:
    st.info("Upload test_data.csv to view model evaluation results.")
    with st.expander("Required columns"):
        st.write(", ".join(feature_names + [TARGET]))
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
    X_test, y_test = prepare_test_data(test_df, feature_names)
except Exception as exc:
    st.error(f"Could not read/evaluate this file: {exc}")
    st.stop()

selected_model = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))
model = models[selected_model]

y_pred = model.predict(X_test)
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]
else:
    y_score = model.decision_function(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_score) if y_test.nunique() == 2 else float("nan"),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Evaluation results — {selected_model}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Benign", "Malignant"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.caption(f"Evaluated {len(X_test)} rows using the saved model. No model training was performed in this app.")
