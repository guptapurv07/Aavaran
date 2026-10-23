"""Metrics used by every experiment. Accuracy is left out on purpose: fraud is
~0.13% of transactions, so a model that flags nothing is already 99.87% accurate."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from config import THRESHOLD, TX_TYPES

FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]


def compute_metrics(y: np.ndarray, prob: np.ndarray, types: np.ndarray, threshold: float = THRESHOLD) -> dict:
    pred = prob >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "false_alarms": int((pred & (y == 0)).sum()),
    }
    # Recall per fraud type shows whether a bank catches fraud it rarely saw.
    for name in FRAUD_TYPES:
        fraud_of_type = (y == 1) & (types == TX_TYPES.index(name))
        metrics[f"recall_{name}"] = float(pred[fraud_of_type].mean())
    return metrics


def format_metrics(m: dict) -> str:
    return (
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
        f"PR-AUC={m['pr_auc']:.3f} | recall TRANSFER={m['recall_TRANSFER']:.3f} "
        f"CASH_OUT={m['recall_CASH_OUT']:.3f} | false alarms={m['false_alarms']:,}"
    )
