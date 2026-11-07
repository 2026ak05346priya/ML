import streamlit as st
import pandas as pd
#import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
st.set_page_config(page_title="ML Classifier Dashboard", layout="wide")

st.title("📊 ML Model Evaluation Dashboard")
st.write("Upload test data CSV to evaluate performance across trained classifiers.")

# Sidebar Controls
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("2. Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

@st.cache_resource
def load_model(name):
    filename = name.lower().replace(" ", "_") + ".pkl"
    return joblib.load(f"model/{filename}")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Test Dataset Preview")
    st.dataframe(df_test.head())
    
    target_col = st.text_input("Enter Target Column Name", value="target")
    
    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]
        
        try:
            scaler = load_scaler()
            model = load_model(model_choice)
            
            # Predict using pre-loaded models (without retraining)
            if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
                X_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_scaled)
                y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.markdown(f"### 📈 Performance Metrics: **{model_choice}**")
            
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else auc)
            c3.metric("Precision", f"{prec:.4f}")
            c4.metric("Recall", f"{rec:.4f}")
            c5.metric("F1 Score", f"{f1:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)
                
            with col2:
                st.subheader("📝 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
        except Exception as e:
            st.error(f"Error loading model or generating predictions: {e}")
    else:
        st.warning(f"Column '{target_col}' not found in uploaded file.")
else:
    st.info("Please upload `test_data.csv` using the sidebar to view metrics.")
