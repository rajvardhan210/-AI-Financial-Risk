import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "clean_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATASET AND CREATE BASELINE FEATURES
# ==================================================

print("=" * 60)
print("AI FINANCIAL RISK INTELLIGENCE PLATFORM")
print("DAY 8: XGBOOST")
print("=" * 60)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

print("\nCreating engineered features...")
df["TransactionAmt_log"] = np.log1p(
    df["TransactionAmt"].clip(lower=0)
)
df["TransactionHour"] = (
    (df["TransactionDT"] // 3600) % 24
).astype(int)

selected_features = [
    "TransactionAmt",
    "TransactionAmt_log",
    "TransactionDT",
    "TransactionHour",
    "ProductCD",
    "card1",
    "card2",
]

X = df[selected_features].copy()
y = df["isFraud"]

print("Selected features:")
print(selected_features)
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ==================================================
# 3. TRAIN/TEST SPLIT
# ==================================================

print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==================================================
# 4. TRAIN-ONLY CLASS IMBALANCE WEIGHT
# ==================================================

negative_training_examples = int((y_train == 0).sum())
positive_training_examples = int((y_train == 1).sum())
scale_pos_weight = (
    negative_training_examples / positive_training_examples
)
print("\nNegative training examples:", negative_training_examples)
print("Positive training examples:", positive_training_examples)
print("scale_pos_weight:", scale_pos_weight)


# ==================================================
# 5. PREPROCESSING
# ==================================================

numerical_features = [
    "TransactionAmt",
    "TransactionAmt_log",
    "TransactionDT",
    "TransactionHour",
    "card1",
    "card2",
]
categorical_features = ["ProductCD"]

numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ==================================================
# 6. MODEL PIPELINE AND TRAINING
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Model training completed successfully!")


# ==================================================
# 7. EVALUATION ON HELD-OUT TEST DATA
# ==================================================

print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(
    "\nMatrix layout: "
    "[[True Negatives, False Positives], "
    "[False Negatives, True Positives]]"
)

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

metrics = {
    "model": "XGBoost",
    "features": selected_features,
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "negative_training_examples": negative_training_examples,
    "positive_training_examples": positive_training_examples,
    "scale_pos_weight": float(scale_pos_weight),
    "fraud_precision": float(report_dict["1"]["precision"]),
    "fraud_recall": float(report_dict["1"]["recall"]),
    "fraud_f1_score": float(report_dict["1"]["f1-score"]),
    "accuracy": float(accuracy),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
}

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Fraud precision: {metrics['fraud_precision']:.4f}")
print(f"Fraud recall: {metrics['fraud_recall']:.4f}")
print(f"Fraud F1-score: {metrics['fraud_f1_score']:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average precision: {avg_precision:.4f}")


# ==================================================
# 8. SAVE THE FITTED PIPELINE AND METRICS
# ==================================================

model_path = MODELS_DIR / "xgboost_model.joblib"
joblib.dump(model, model_path)
print("\nFitted preprocessing/model pipeline saved to:")
print(model_path)

metrics_path = MODELS_DIR / "xgboost_metrics.json"
with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)
print("Evaluation metrics saved to:")
print(metrics_path)
