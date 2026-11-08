"""
Breast Cancer Wisconsin (Diagnostic) - Classification with 6 evaluation metrics.
Trains Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM
on the same dataset / same train-test split and saves each model as .joblib.

Usage:  python train_models.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = "data.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # drop id and the empty trailing column created by the trailing comma in the CSV header
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == "id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # 1 = Malignant, 0 = Benign
    return df


def get_models():
    # Scale-sensitive models are wrapped in a Pipeline so the scaler is saved with the model.
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": scaled(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    df = load_data()
    X, y = df.drop(columns="diagnosis"), df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results, cms = {}, {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        cms[name] = confusion_matrix(y_test, model.predict(X_test))
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        print(f"{name:25s}", {k: round(v, 4) for k, v in results[name].items()})

    res_df = pd.DataFrame(results).T.round(4)
    res_df.index.name = "ML Model Name"
    res_df.to_csv("model_comparison.csv")
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(res_df.to_dict(orient="index"), f, indent=2)
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

    print("\n", res_df.to_markdown())

    # Confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=120)


if __name__ == "__main__":
    main()
