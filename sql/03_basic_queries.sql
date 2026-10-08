SELECT COUNT(*)
FROM transactions
WHERE isFraud = 1
AND TransactionAmt > 500;