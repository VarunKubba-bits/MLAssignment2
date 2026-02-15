import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Classification Model Demo")

# ---------------------------------------------------
# Download File
# ---------------------------------------------------

st.sidebar.header("Download Credit Card CSV Dataset")
# Download button
with open("model_metrics.csv", "rb") as f:
    st.sidebar.download_button(
        label="Download Credit Card CSV",
        data=f,
        file_name="model_metrics.csv",
        mime="text/csv"
    )


uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

if uploaded_file:
    
    uploaded_df = pd.read_csv(uploaded_file)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Separate features and target
    X = uploaded_df.iloc[:, :-1]
    y = uploaded_df.iloc[:, -1]

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    expected_features = scaler.n_features_in_

    if X.shape[1] != expected_features:
        st.error(f"Expected {expected_features} features, but got {X.shape[1]}")
        st.stop()

    X_scaled = scaler.transform(X.values)
    preds = model.predict(X_scaled)
    
    st.subheader("Classification Report")

    st.subheader("Model Comparison Table")

    metrics_path = os.path.join(MODEL_DIR, "model_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    
    st.dataframe(metrics_df)

    X = uploaded_df.iloc[:, :-1]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_scaled = scaler.transform(X.values)
    preds = model.predict(X_scaled)

    y_true = uploaded_df.iloc[:, -1].astype(int).values
    y_pred = preds.astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)
    
    st.text(classification_report(y, preds))

    # ---------------------------------------------------
    # Evaluation Metrics
    # ---------------------------------------------------
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
