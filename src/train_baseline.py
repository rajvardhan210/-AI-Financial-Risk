import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "clean_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("=" * 60)
print("AI FINANCIAL RISK INTELLIGENCE PLATFORM")
print("DAY 6: LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Dataset loaded successfully!")


# ==================================================
# 3. FEATURE ENGINEERING
# ==================================================

print("\nCreating engineered features...")

# Log transformation reduces the right skew of transaction amounts
df["TransactionAmt_log"] = np.log1p(
    df["TransactionAmt"].clip(lower=0)
)

# Derive a repeating 24-hour interval from TransactionDT.
# This is a relative interval, not necessarily the actual local clock hour.
df["TransactionHour"] = (
    (df["TransactionDT"] // 3600) % 24
).astype(int)

# Features used by the baseline model
selected_features = [
    "TransactionAmt",
    "TransactionAmt_log",
    "TransactionDT",
    "TransactionHour",
    "ProductCD",
    "card1",
    "card2"
]

# Input features
X = df[selected_features].copy()

# Target variable: 0 = legitimate, 1 = fraud
y = df["isFraud"]

print("Selected features:")
print(selected_features)

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nFraud percentage:")
print(round(y.mean() * 100, 2), "%")


# ==================================================
# 4. TRAIN/TEST SPLIT
# ==================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print(
    "Training fraud percentage:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Testing fraud percentage:",
    round(y_test.mean() * 100, 2),
    "%"
)


# ==================================================
# 5. DEFINE FEATURE TYPES
# ==================================================

numerical_features = [
    "TransactionAmt",
    "TransactionAmt_log",
    "TransactionDT",
    "TransactionHour",
    "card1",
    "card2"
]

categorical_features = [
    "ProductCD"
]


# ==================================================
# 6. PREPROCESSING PIPELINE
# ==================================================

# Numerical preprocessing:
# 1. Fill missing values with the median.
# 2. Standardize numerical features.

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing:
# 1. Fill missing values with the most frequent category.
# 2. Convert categories into numerical columns.

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Combine numerical and categorical preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("\nPreprocessing pipeline created successfully!")


# ==================================================
# 7. BUILD LOGISTIC REGRESSION MODEL
# ==================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ==================================================
# 8. TRAIN MODEL
# ==================================================

print("\nTraining Logistic Regression model...")
print("Please wait; the dataset is large.")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ==================================================
# 9. MAKE PREDICTIONS
# ==================================================

print("\nEvaluating model...")

# Predict class labels
y_pred = model.predict(X_test)

# Predict probability of fraud
y_proba = model.predict_proba(X_test)[:, 1]


# ==================================================
# 10. EVALUATE MODEL
# ==================================================

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print(
    "\nMatrix layout: "
    "[[True Negatives, False Positives], "
    "[False Negatives, True Positives]]"
)


# Classification report
report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# Additional metrics
roc_auc = roc_auc_score(
    y_test,
    y_proba
)

avg_precision = average_precision_score(
    y_test,
    y_proba
)

print("\nAdditional Evaluation Metrics:")

print(f"ROC-AUC: {roc_auc:.4f}")

print(
    f"Average Precision (AP): {avg_precision:.4f}"
)


# ==================================================
# 11. PREPARE METRICS FOR SAVING
# ==================================================

metrics = {
    "model": "Logistic Regression",
    "features": selected_features,
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "fraud_precision": float(
        report_dict["1"]["precision"]
    ),
    "fraud_recall": float(
        report_dict["1"]["recall"]
    ),
    "fraud_f1_score": float(
        report_dict["1"]["f1-score"]
    ),
    "accuracy": float(
        report_dict["accuracy"]
    ),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision)
}


# ==================================================
# 12. SAVE TRAINED MODEL
# ==================================================

MODEL_PATH = (
    MODELS_DIR / "logistic_regression_baseline.joblib"
)

joblib.dump(model, MODEL_PATH)

print("\nTrained model saved successfully:")
print(MODEL_PATH)


# ==================================================
# 13. SAVE EVALUATION METRICS
# ==================================================

METRICS_PATH = (
    MODELS_DIR / "baseline_metrics.json"
)

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nEvaluation metrics saved successfully:")
print(METRICS_PATH)


# ==================================================
# 14. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DAY 6 BASELINE SUMMARY")
print("=" * 60)

print("Algorithm: Logistic Regression")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    f"Fraud Precision: "
    f"{metrics['fraud_precision']:.4f}"
)

print(
    f"Fraud Recall: "
    f"{metrics['fraud_recall']:.4f}"
)

print(
    f"Fraud F1-score: "
    f"{metrics['fraud_f1_score']:.4f}"
)

print(f"ROC-AUC: {roc_auc:.4f}")

print(f"Average Precision: {avg_precision:.4f}")

print("\nSaved files:")
print("- models/logistic_regression_baseline.joblib")
print("- models/baseline_metrics.json")

print("\nDay 6 model training and evaluation completed!")