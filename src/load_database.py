import pandas as pd
import sqlite3
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
CSV_PATH = BASE_DIR / "data" / "clean_dataset.csv"
DB_PATH = BASE_DIR / "data" / "fraud.db"

print("Loading cleaned dataset...")

# Connect to SQLite database
connection = sqlite3.connect(DB_PATH)

# Read the CSV in chunks
chunk_size = 50_000

first_chunk = True
total_rows = 0

for chunk in pd.read_csv(CSV_PATH, chunksize=chunk_size):

    if first_chunk:
        chunk.to_sql(
            "transactions",
            connection,
            if_exists="replace",
            index=False
        )
        first_chunk = False

    else:
        chunk.to_sql(
            "transactions",
            connection,
            if_exists="append",
            index=False
        )

    total_rows += len(chunk)

    print(f"Loaded {total_rows:,} rows")

connection.close()

print("\nDatabase created successfully!")
print("Database:", DB_PATH)
print("Total rows:", total_rows)