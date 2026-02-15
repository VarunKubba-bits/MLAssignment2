# ML Assignment 2 — Classification Models & Streamlit Deployment

## Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard metrics, and deploy an interactive Streamlit web application to demonstrate predictions.

This project demonstrates an end-to-end ML workflow:
- Data preprocessing
- Model training
- Model evaluation
- Model comparison
- Streamlit deployment

---

## Dataset Description

Dataset used: **Credit Card Fraud Detection Dataset (2023)**  
Source: Kaggle

This dataset contains anonymized transaction features used to detect fraudulent credit card transactions.

**Dataset characteristics**
- Binary classification problem
- Number of features: 29
- Target variable: `Class`
- Instances: Large-scale transaction dataset
- Features are numerical and preprocessed

The dataset satisfies the assignment requirement of:
- Minimum 12 features
- Minimum 500 instances

---

## Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

All models were trained on the same dataset and evaluated using the same test split.

---

## Evaluation Metrics

The following metrics were computed for each model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.965215 | 0.993396 | 0.977125 | 0.952875 | 0.964847 | 0.930720 |
| Decision Tree | 0.998004 | 0.998002 | 0.996970 | 0.999052 | 0.998010 | 0.996010 |
| kNN | 0.954795 | 0.987353 | 0.962523 | 0.946627 | 0.954509 | 0.909717 |
| Naive Bayes | 0.918040 | 0.974501 | 0.975286 | 0.858151 | 0.912977 | 0.842236 |
| Random Forest (Ensemble) | 0.999886 | 0.999981 | 0.999772 | 1.000000 | 0.999886 | 0.999771 |
| XGBoost (Ensemble) | 0.999129 | 0.999968 | 0.998283 | 0.999982 | 0.999132 | 0.998260 |

*(Values populated from comparison_model_metrics.csv)*

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Provides strong baseline performance with high AUC but slightly lower accuracy compared to ensemble models. |
| Decision Tree | Performs very well on the dataset and captures non-linear relationships effectively. |
| kNN | Shows good performance but slightly lower accuracy due to reduced training samples for memory efficiency. |
| Naive Bayes | Fast and simple model but performance is lower due to independence assumptions. |
| Random Forest (Ensemble) | Achieves the best overall performance with near-perfect accuracy and MCC. |
| XGBoost (Ensemble) | Provides extremely high performance close to Random Forest with strong generalization. |

---

## Streamlit Application

The deployed Streamlit app provides:

- CSV dataset upload (test data)
- Model selection dropdown
- Prediction generation
- Classification report display
- Confusion matrix visualization
- Model comparison metrics table

---

## Project Structure

project/
│-- app.py
│-- train_models.py
│-- requirements.txt
│-- README.md
|-- model_metrics.csv
|-- comparison_model_metrics.csv
│-- model/
│ ├── *.pkl
│ ├── scaler.pkl

---

## How to Run Locally

Install dependencies:
pip install -r requirements.txt

Train models:
python train_models.py

Run Streamlit app:
streamlit run app.py

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.
