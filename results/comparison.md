# Aavaran results

All setups are scored on the same held-out test set: 1,272,524 transactions, 1,642 of them fraud.
Precision, recall and F1 use a 0.5 probability threshold; PR-AUC does not depend on a threshold.

| Setup | Precision | Recall | F1 | PR-AUC | Recall TRANSFER | Recall CASH_OUT | False alarms |
|---|---|---|---|---|---|---|---|
| Bank 1 alone | 0.998 | 0.540 | 0.701 | 0.733 | 1.000 | 0.083 | 2 |
| Bank 2 alone | 0.994 | 0.497 | 0.663 | 0.616 | 0.000 | 0.991 | 5 |
| Bank 3 alone | 0.999 | 0.512 | 0.677 | 0.735 | 1.000 | 0.026 | 1 |
| Bank 4 alone | 0.981 | 0.497 | 0.660 | 0.554 | 0.000 | 0.991 | 16 |
| Bank 5 alone | 0.979 | 0.993 | 0.986 | 0.994 | 0.999 | 0.987 | 35 |
| Aavaran federated | 0.998 | 0.972 | 0.985 | 0.992 | 0.985 | 0.959 | 4 |
| Aavaran + DP, ε = 10 | 0.989 | 0.968 | 0.978 | 0.978 | 0.958 | 0.977 | 18 |
| Aavaran + DP, ε = 5 | 0.989 | 0.967 | 0.978 | 0.978 | 0.958 | 0.976 | 18 |
| Aavaran + DP, ε = 1 | 0.990 | 0.947 | 0.968 | 0.976 | 0.949 | 0.945 | 16 |
| Centralised (upper limit) | 0.999 | 0.996 | 0.997 | 0.996 | 1.000 | 0.991 | 2 |

Banks 1 and 3 never saw CASH_OUT fraud; Banks 2 and 4 never saw TRANSFER fraud.

![f1_by_setup](charts/f1_by_setup.png)
![unseen_fraud_recall](charts/unseen_fraud_recall.png)
![rounds_pr_auc](charts/rounds_pr_auc.png)
![privacy_tradeoff](charts/privacy_tradeoff.png)
