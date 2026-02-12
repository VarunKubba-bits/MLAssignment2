import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Classification Model Demo")

uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    model = joblib.load(f"model/{model_name}.pkl")
    scaler = joblib.load("model/scaler.pkl")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = scaler.transform(X)

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    metrics_df = pd.read_csv("model/model_metrics.csv")
    st.subheader("Saved Model Metrics")
    st.dataframe(metrics_df)
