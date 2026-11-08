# Breast Cancer Classification – Machine Learning Model Comparison

## 1. Problem Statement

Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis is important for improving patient outcomes.

The objective of this project is to develop and compare five machine learning classification algorithms to predict whether a breast tumour is **Malignant (M)** or **Benign (B)** using measurements of cell nuclei obtained from digitized fine-needle aspirate (FNA) images.

The five classifiers are evaluated on the same dataset and train-test split using six performance metrics: Accuracy, AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Type:** Binary classification
- **Instances:** 569 (357 Benign, 212 Malignant)
- **Features:** 30 numerical features
- **Target variable:** `diagnosis` – Malignant (M = 1), Benign (B = 0)
- **Missing values:** None in the 30 feature columns

The 30 features represent the mean, standard error, and worst (largest) values of 10 characteristics of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

### Data Preprocessing

- Dropped the `id` column and empty trailing column.
- Encoded the target variable: M = 1 and B = 0.
- Applied an 80:20 stratified train-test split using `random_state=42`.
- Applied `StandardScaler` within pipelines for Logistic Regression, KNN, and SVM.

## 3. Project Links

- **GitHub Repository:** [https://github.com/ankit1aiml/ML-Assignment-Ankit/tree/main]
- **Live Streamlit Application:** [https://ml-assignment-ankit-idid9xl9pmcub5cndi2ijl.streamlit.app/]

## 4. Machine Learning Models

The following five classifiers were implemented and evaluated:

1. Logistic Regression (`max_iter=1000`)
2. Decision Tree Classifier (`max_depth=5`)
3. K-Nearest Neighbors (`n_neighbors=5`)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

All models were trained using the same training dataset and evaluated on the same held-out test dataset of 114 samples.

## 5. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

## 6. Observations on Model Performance

| Model | Observations |
|---|---|
| Logistic Regression | Strong baseline performance, with 96.49% accuracy and the highest AUC of 0.9960. This indicates excellent discrimination between the two classes. |
| Decision Tree | Lowest performance among the five models, with 92.11% accuracy and 83.33% recall. The limited tree depth may restrict its ability to capture more complex patterns. It offers relatively straightforward interpretability. |
| KNN | Achieved 95.61% accuracy and 0.9823 AUC. Feature scaling helped make distance-based comparisons meaningful. Its recall was slightly lower than Logistic Regression and SVM. |
| Gaussian Naive Bayes | Achieved perfect precision and an AUC of 0.9934, but recall was 83.33%. Its conditional independence assumption may be restrictive for correlated features such as radius, perimeter, and area. |
| Support Vector Machine | Best overall performance, achieving the highest accuracy (97.37%), F1-score (0.9630), and MCC (0.9442), along with perfect precision. The RBF kernel can capture nonlinear decision boundaries. |

## 7. Overall Model Selection

**Support Vector Machine (RBF kernel) is the best-performing model overall** based on the evaluated test-set metrics.

It achieved:

- Accuracy: 97.37%
- Precision: 1.0000
- Recall: 0.9286
- F1-Score: 0.9630
- MCC: 0.9442

Logistic Regression was a close competitor and achieved the highest AUC of 0.9960. SVM and Logistic Regression also achieved the highest recall of 0.9286.

MCC was considered an important comparison metric because it accounts for all four confusion-matrix outcomes. However, the test set contains only 114 samples, so small differences in the number of predictions can affect the rankings. Cross-validation would provide a more robust estimate of generalization performance.

These results represent an educational classification experiment and should not be interpreted as clinical validation or a substitute for medical diagnosis.

## 8. Repository Structure

```text
ML-Assignment-Ankit/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data.csv
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── feature_names.joblib
    └── metrics.json
```

## 9. How to Run the Project Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the five models:

```bash
python train_models.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## 10. Streamlit Application Usage

The deployed application allows users to:

1. Upload a test CSV file containing the required input features and target column (`diagnosis`).
2. Select one of the five trained machine learning models.
3. Evaluate the selected model on the uploaded dataset.
4. View Accuracy, AUC, Precision, Recall, F1-Score, and MCC.
5. View the confusion matrix and classification report.

The application loads the previously trained and saved model artifacts. It does not retrain models when a user uploads a test dataset.

## 11. Saved Model Artifacts

All five trained models are saved in the `model/` directory in Joblib format. The feature-name artifact and metrics JSON file are also saved to support model loading and application functionality.

The training script can be executed again to reproduce the model artifacts and evaluation results.
