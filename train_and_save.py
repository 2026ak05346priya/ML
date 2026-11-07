import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Load Dataset (Breast Cancer Dataset: 569 instances, 30 features - meets criteria >=500 instances, >=12 features)
data = load_breast_cancer(as_frame=True)
df = data.frame

# Rename target column for clarity
X = df.drop(columns=['target'])
y = df['target']

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing / Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test dataset and scaler artifact
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.values
test_df.to_csv("test_data.csv", index=False)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# 4. Initialize the 5 Classification Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# 5. Model Evaluation and Saving
results = []
for name, model in models.items():
    # Fit scaled features for scale-sensitive models, unscaled for Decision Tree
    if name in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Save trained model to disk
    file_name = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"model/{file_name}")
    
    # Calculate required 6 evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

# Print Results Table to terminal
results_df = pd.DataFrame(results)
print("\n--- Model Evaluation Results ---")
print(results_df.to_string(index=False))