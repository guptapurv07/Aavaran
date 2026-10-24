# Aavaran results

## Main benchmark: random test set

Test set: 1,272,524 transactions, 1,642 of them fraud (0.13%), held out from every bank. Each value is the mean ± standard deviation over 3 training seeds (same data split). Precision, recall and F1 use a 0.5 probability threshold; PR-AUC does not depend on a threshold.

| Setup | Precision | Recall | F1 | PR-AUC | Recall TRANSFER | Recall CASH_OUT | False alarms |
|---|---|---|---|---|---|---|---|
| Bank 1 alone | 0.994 ± 0.001 | 0.542 ± 0.043 | 0.701 ± 0.036 | 0.780 ± 0.081 | 1.000 ± 0.000 | 0.085 ± 0.086 | 5 ± 1 |
| Bank 2 alone | 0.991 ± 0.002 | 0.573 ± 0.127 | 0.721 ± 0.098 | 0.943 ± 0.028 | 0.153 ± 0.254 | 0.991 ± 0.000 | 8 ± 1 |
| Bank 3 alone | 0.994 ± 0.002 | 0.540 ± 0.045 | 0.700 ± 0.038 | 0.808 ± 0.043 | 1.000 ± 0.000 | 0.083 ± 0.091 | 5 ± 2 |
| Bank 4 alone | 0.990 ± 0.002 | 0.576 ± 0.070 | 0.727 ± 0.057 | 0.932 ± 0.018 | 0.158 ± 0.140 | 0.991 ± 0.000 | 9 ± 2 |
| Bank 5 alone | 0.965 ± 0.002 | 0.986 ± 0.002 | 0.975 ± 0.001 | 0.992 ± 0.001 | 0.985 ± 0.004 | 0.987 ± 0.000 | 58 ± 4 |
| Aavaran federated | 0.995 ± 0.002 | 0.988 ± 0.003 | 0.992 ± 0.002 | 0.994 ± 0.001 | 0.991 ± 0.006 | 0.986 ± 0.001 | 8 ± 3 |
| Aavaran + DP, ε = 10 | 0.989 ± 0.002 | 0.968 ± 0.001 | 0.978 ± 0.001 | 0.978 ± 0.000 | 0.959 ± 0.001 | 0.977 ± 0.001 | 18 ± 3 |
| Aavaran + DP, ε = 5 | 0.988 ± 0.002 | 0.967 ± 0.001 | 0.977 ± 0.001 | 0.978 ± 0.000 | 0.958 ± 0.000 | 0.976 ± 0.001 | 19 ± 3 |
| Aavaran + DP, ε = 1 | 0.989 ± 0.001 | 0.950 ± 0.007 | 0.969 ± 0.003 | 0.976 ± 0.001 | 0.948 ± 0.003 | 0.953 ± 0.012 | 18 ± 2 |
| Centralised (upper limit) | 0.999 ± 0.000 | 0.996 ± 0.000 | 0.998 ± 0.000 | 0.996 ± 0.000 | 1.000 ± 0.000 | 0.991 ± 0.000 | 1 ± 0 |

Average F1 of the 5 banks training alone: 0.765.

Banks 1 and 3 never saw CASH_OUT fraud; Banks 2 and 4 never saw TRANSFER fraud.

![f1_by_setup](charts/f1_by_setup.png)
![unseen_fraud_recall](charts/unseen_fraud_recall.png)
![rounds_pr_auc](charts/rounds_pr_auc.png)
![privacy_tradeoff](charts/privacy_tradeoff.png)

## Out-of-time check: test on later hours

Models train on PaySim's first 355 hours and are tested on the rest: 1,248,736 transactions, 4,250 of them fraud (0.34%). PaySim's normal transaction volume falls sharply in later hours while fraud stays steady, so the test fraud rate is 4.4 times the training rate, a real shift between training and test data.

| Setup | Precision | Recall | F1 | PR-AUC | Recall TRANSFER | Recall CASH_OUT | False alarms |
|---|---|---|---|---|---|---|---|
| Bank 1 alone | 0.996 | 0.507 | 0.672 | 0.811 | 0.994 | 0.021 | 8 |
| Bank 2 alone | 0.996 | 0.500 | 0.666 | 0.526 | 0.000 | 1.000 | 8 |
| Bank 3 alone | 0.996 | 0.731 | 0.843 | 0.973 | 0.978 | 0.483 | 13 |
| Bank 4 alone | 0.995 | 0.497 | 0.663 | 0.879 | 0.000 | 0.994 | 11 |
| Bank 5 alone | 0.994 | 0.970 | 0.982 | 0.984 | 0.947 | 0.993 | 24 |
| Aavaran federated | 0.997 | 0.971 | 0.984 | 0.994 | 0.947 | 0.994 | 13 |
| Aavaran + DP, ε = 10 | 0.995 | 0.944 | 0.969 | 0.977 | 0.934 | 0.955 | 19 |
| Aavaran + DP, ε = 5 | 0.996 | 0.936 | 0.965 | 0.976 | 0.928 | 0.944 | 16 |
| Aavaran + DP, ε = 1 | 0.998 | 0.888 | 0.940 | 0.971 | 0.890 | 0.886 | 6 |
| Centralised (upper limit) | 1.000 | 0.997 | 0.998 | 1.000 | 0.999 | 0.995 | 2 |

Average F1 of the 5 banks training alone: 0.765.
