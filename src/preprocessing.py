import pandas as pd
from pathlib import Path


# -----------------------------
# 1. File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "train_transaction.csv"
CLEAN_DATA = BASE_DIR / "data" / "clean_dataset.csv"


# -----------------------------
# 2. Load raw dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(RAW_DATA)

print("Original shape:", df.shape)


# -----------------------------
# 3. Remove duplicate rows
# -----------------------------

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# -----------------------------
# 4. Remove columns with >90% missing values
# -----------------------------

missing_percent = df.isnull().mean() * 100

high_missing_cols = missing_percent[missing_percent > 90].index

print("Columns removed (>90% missing):", len(high_missing_cols))

df = df.drop(columns=high_missing_cols)


# -----------------------------
# 5. Fill remaining missing values
# -----------------------------

numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(exclude="number").columns


# Numerical → median
for col in numeric_cols:
    if col != "isFraud":
        df[col] = df[col].fillna(df[col].median())


# Categorical → mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])


# -----------------------------
# 6. Save cleaned dataset
# -----------------------------

df.to_csv(CLEAN_DATA, index=False)

print("Clean dataset saved to:", CLEAN_DATA)
print("Final shape:", df.shape)

print("\nRemaining missing values:",
      df.isnull().sum().sum())