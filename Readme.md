# Machine Learning Assignment 2 - Model Evaluation & Deployment

## a. Problem Statement
Implement, evaluate, and compare 5 classification models on a dataset and deploy an interactive evaluation dashboard on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset**: UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- **Instances**: 569 samples (>= 500 requirement met)
- **Features**: 30 numeric predictive features (>= 12 requirement met)
- **Target**: Binary Classification (0 = Malignant, 1 = Benign)

## c. GitHub Repository Link
`https://github.com/YOUR_USERNAME/ml_assignment_2`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9974 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9474 | 0.9427 | 0.9583 | 0.9583 | 0.9583 | 0.8872 |
| **KNN** | 0.9649 | 0.9914 | 0.9722 | 0.9722 | 0.9722 | 0.9248 |
| **Naive Bayes** | 0.9386 | 0.9881 | 0.9452 | 0.9583 | 0.9517 | 0.8690 |
| **Support Vector Machine** | 0.9737 | 0.9954 | 0.9726 | 0.9861 | 0.9793 | 0.9439 |

### Observations & Winner

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved optimal overall accuracy (98.25%) and top AUC (0.9974) post feature scaling. |
| **Decision Tree** | Faster execution but slightly lower generalization accuracy (94.74%). |
| **KNN** | Strong spatial boundary performance (96.49%) with $k=5$. |
| **Naive Bayes** | Fast probabilistic baseline (93.86%), slightly impacted by feature correlation. |
| **Support Vector Machine** | Excellent generalization (97.37%) using RBF kernel margin optimization. |
| **Overall Winner for your dataset?** | **Logistic Regression** achieved the highest overall score across Accuracy, AUC, and MCC. |