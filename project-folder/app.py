import pandas as pd
import numpy as np
import joblib
import kagglehub
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# Load dataset
# Original download path (read-only)
kaggle_download_root = kagglehub.dataset_download('nelgiriyewithana/credit-card-fraud-detection-dataset-2023')
print(f"Kaggle dataset downloaded to: {kaggle_download_root}")
print('Data source import complete.')

file_path = os.path.join(kaggle_download_root, "creditcard_2023.csv")
data = pd.read_csv(file_path, sep=',')

# Correctly separate features (X) and target (y) from the DataFrame
X = data.drop(['id', 'Class'], axis=1) # Drop 'id' and 'Class' to get features
y = data['Class'] # 'Class' is the target variable

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create 'models' directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(scaler, "models/scaler.pkl")

# Models
models = {
    "logistic": LogisticRegression(max_iter=500),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    }

    results.append(metrics)
    joblib.dump(model, f"models/{name}.pkl")

df = pd.DataFrame(results)
print(df)
df.to_csv("models/model_metrics.csv", index=False)
