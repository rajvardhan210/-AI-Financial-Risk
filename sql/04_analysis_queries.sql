SELECT isFraud, COUNT(*)
FROM transactions
GROUP BY isFraud;
SELECT ProductCD, COUNT(*) AS fraud_count
FROM transactions
WHERE isFraud = 1
GROUP BY ProductCD;
SELECT isFraud, AVG(TransactionAmt) AS average_amount
FROM transactions
GROUP BY isFraud;
SELECT TransactionID, TransactionAmt, ProductCD
FROM transactions
WHERE isFraud = 1
ORDER BY TransactionAmt DESC
LIMIT 1;