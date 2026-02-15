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
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| XGBoost (Ensemble) | | | | | | |

*(Values populated from comparison_model_metrics.csv)*

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Performs well on linearly separable patterns and provides stable baseline performance. |
| Decision Tree | Captures non-linear relationships but may overfit without pruning. |
| kNN | Performance depends on sample size and distance metric; memory-intensive for large datasets. |
| Naive Bayes | Fast and simple but assumes feature independence. |
| Random Forest (Ensemble) | Strong performance due to averaging multiple trees and reducing variance. |
| XGBoost (Ensemble) | Achieves the best performance due to boosting and regularization. |

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
