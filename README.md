# Breast Cancer Classification – ML Model Comparison

## a. Problem Statement
Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis greatly improves survival. The goal of this project is to build and compare **five machine learning classifiers** that predict whether a breast tumour is **Malignant (M)** or **Benign (B)** from measurements of cell nuclei taken from a digitized fine-needle aspirate (FNA) image. All five models are trained and evaluated on the **same dataset and the same train/test split**, using six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), to find the best model for this problem.

## b. Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (`data.csv`)
- **Source:** UCI Machine Learning Repository / Kaggle
- **Type:** Binary classification
- **Instances:** 569 (357 Benign, 212 Malignant)
- **Features:** 30 numeric features (after dropping the `id` column and an empty trailing column)
- **Target:** `diagnosis` – `M` (Malignant → 1) or `B` (Benign → 0)
- **Missing values:** None in the 30 feature columns

The 30 features are the **mean**, **standard error (se)** and **worst (largest)** values of 10 characteristics computed for each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

**Preprocessing:** dropped `id` and the empty column; encoded target (M=1, B=0); stratified 80/20 train-test split (`random_state=42`); `StandardScaler` applied (inside a pipeline) for Logistic Regression, KNN and SVM.

## c. GitHub Repository Link: https://github.com/ankit1aiml/ML-Assignment-Ankit/tree/main

## d. Models Used
1. Logistic Regression
2. Decision Tree Classifier (`max_depth=5`)
3. K-Nearest Neighbor Classifier (`k=5`)
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (RBF kernel)

### Comparison Table (test set, 114 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong baseline: 96.5% accuracy and the highest AUC (0.996), meaning it ranks malignant vs. benign cases almost perfectly. The classes are close to linearly separable once features are scaled. |
| Decision Tree | Weakest model on every metric (accuracy 92.1%, recall 83.3%, MCC 0.83). A single tree with limited depth is less stable and misses more malignant cases; it does offer easy interpretability. |
| KNN | Good performance (95.6% accuracy, AUC 0.982) after scaling. Slightly lower recall than LR/SVM and slower at prediction time since it stores all training data. |
| Naive Bayes | Perfect precision (no false alarms) and a high AUC (0.993), but recall is low (83.3%), so it misses several malignant tumours. The independence assumption is violated because many features (radius, perimeter, area) are strongly correlated. |
| Support Vector Machine | Best overall: highest accuracy (97.4%), F1 (0.963) and MCC (0.944), with perfect precision and 92.9% recall. The RBF kernel captures the slightly non-linear boundary well. |

**Overall winner for this dataset: Support Vector Machine (RBF kernel)** – it has the best Accuracy, Precision, F1 and MCC, with Logistic Regression a very close second (best AUC). Because MCC accounts for all four confusion-matrix cells, it is a reliable single measure for comparison. Note that the test set is small (114 samples), so differences of 1–2 cases can shift the ranking; cross-validation would give a more robust estimate. In a medical setting, recall (catching malignant cases) is especially important, and SVM/Logistic Regression are tied best on it.

## Repository Structure
```
├── data.csv                 # dataset
├── train_models.py          # trains, evaluates and saves all 5 models
├── requirements.txt
├── README.md
├── model_comparison.csv     # metrics table
├── confusion_matrices.png
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── feature_names.joblib
    └── metrics.json
```

## How to Run
```bash
pip install -r requirements.txt
python train_models.py
```

Load a saved model:
```python
import joblib
model = joblib.load("model/support_vector_machine.joblib")
pred = model.predict(X_new)   # X_new: DataFrame with the 30 feature columns
```
