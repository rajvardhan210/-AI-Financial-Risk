import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "fraud.db"


def get_fraud_transactions(limit=5):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT TransactionID, TransactionAmt, ProductCD
        FROM transactions
        WHERE isFraud = 1
        ORDER BY TransactionAmt DESC
        LIMIT ?;
    """, (limit,))

    transactions = cursor.fetchall()

    connection.close()

    return transactions


# Get top 10 fraud transactions
fraud_transactions = get_fraud_transactions(10)

for transaction in fraud_transactions:
    print(transaction)