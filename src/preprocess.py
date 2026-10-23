"""Stage 2: clean PaySim, engineer features and split it into 5 simulated banks.

Outputs (in data/processed/):
  bank_<i>.npz     training and local validation data for bank i (0-based)
  test.npz         common held-out test set
  test_stream.parquet  raw test transactions, tagged with a bank, for Kafka replay
  scaler.json      feature means/stds built from per-bank aggregate statistics
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    FRAUD_SPLIT,
    LEGIT_SPLIT,
    NUM_BANKS,
    PROCESSED_DIR,
    RAW_CSV,
    RESULTS_DIR,
    SEED,
    TEST_FRACTION,
    TX_TYPES,
    VAL_FRACTION,
)

RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]
# Sender-side features only: a bank knows its own customer's balances, not the
# balances of a receiving account held at another bank. (PaySim's destination
# balances are also a known simulator artefact for fraud rows.)
CONTINUOUS = ["log_amount", "log_old_bal_orig", "log_new_bal_orig", "err_bal_orig"]
FEATURES = CONTINUOUS + ["orig_drained", "hour_sin", "hour_cos"] + [f"type_{t}" for t in TX_TYPES]


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Turn raw PaySim columns into the unscaled model features (float32).

    Shared with the Kafka scorer so live transactions get exactly the same features.
    """
    amount = df["amount"].to_numpy(np.float64)
    old_orig = df["oldbalanceOrg"].to_numpy(np.float64)
    new_orig = df["newbalanceOrig"].to_numpy(np.float64)
    hour = df["step"].to_numpy() % 24
    type_codes = pd.Categorical(df["type"], categories=TX_TYPES).codes

    cols = [
        np.log1p(amount),
        np.log1p(old_orig),
        np.log1p(new_orig),
        # Money that does not add up on the sender's books after the transaction.
        signed_log(old_orig - amount - new_orig),
        ((old_orig > 0) & (new_orig == 0)).astype(np.float64),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ]
    one_hot = np.eye(len(TX_TYPES))[type_codes]
    return np.column_stack(cols + [one_hot]).astype(np.float32)


def assign_banks(type_codes: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give each row a bank id, following the FRAUD_SPLIT / LEGIT_SPLIT shares."""
    bank = np.full(len(labels), -1, dtype=np.int8)
    for code, tx_type in enumerate(TX_TYPES):
        for label, split in ((0, LEGIT_SPLIT), (1, FRAUD_SPLIT)):
            idx = np.flatnonzero((type_codes == code) & (labels == label))
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            cuts = np.round(np.cumsum(split[tx_type])[:-1] * len(idx)).astype(int)
            for b, part in enumerate(np.split(idx, cuts)):
                bank[part] = b
    assert (bank >= 0).all()
    return bank


def federated_scaler(X: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the continuous features from per-bank (count, sum, sum of squares).

    Each bank only shares three aggregate numbers per feature, never rows, and the
    result is exactly the mean/std of all banks' training data combined.
    """
    n_cont = len(CONTINUOUS)
    count, total, total_sq = 0, np.zeros(n_cont), np.zeros(n_cont)
    for b in range(NUM_BANKS):
        Xb = X[bank == b, :n_cont].astype(np.float64)
        count += len(Xb)
        total += Xb.sum(axis=0)
        total_sq += (Xb**2).sum(axis=0)
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return mean, std


def scale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, : len(mean)] = (X[:, : len(mean)] - mean) / std
    return X


def main() -> None:
    rng = np.random.default_rng(SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, engine="pyarrow", usecols=RAW_COLUMNS)
    print(f"  {len(df):,} transactions, {df.isFraud.sum():,} fraud")

    X = build_features(df)
    y = df["isFraud"].to_numpy(np.int8)
    t = pd.Categorical(df["type"], categories=TX_TYPES).codes.astype(np.int8)

    # Common test set, stratified so it has the same type/fraud mix as the full data.
    strata = t * 2 + y
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_FRACTION, stratify=strata, random_state=SEED
    )
    train_idx.sort()
    test_idx.sort()

    bank = assign_banks(t[train_idx], y[train_idx], rng)
    mean, std = federated_scaler(X[train_idx], bank)
    X = scale(X, mean, std)

    summary = {"features": FEATURES, "banks": []}
    for b in range(NUM_BANKS):
        rows = train_idx[bank == b]
        tr, va = train_test_split(
            rows, test_size=VAL_FRACTION, stratify=strata[rows], random_state=SEED
        )
        np.savez(
            PROCESSED_DIR / f"bank_{b}.npz",
            X_train=X[tr], y_train=y[tr], t_train=t[tr],
            X_val=X[va], y_val=y[va], t_val=t[va],
        )
        fraud = y[rows] == 1
        summary["banks"].append({
            "bank": b + 1,
            "rows": int(len(rows)),
            "fraud": int(fraud.sum()),
            "fraud_TRANSFER": int((fraud & (t[rows] == TX_TYPES.index("TRANSFER"))).sum()),
            "fraud_CASH_OUT": int((fraud & (t[rows] == TX_TYPES.index("CASH_OUT"))).sum()),
        })

    np.savez(PROCESSED_DIR / "test.npz", X=X[test_idx], y=y[test_idx], t=t[test_idx])

    # Raw test transactions for the Kafka producer, tagged with the bank that receives them.
    stream = df.iloc[test_idx].copy()
    stream["bank"] = assign_banks(t[test_idx], y[test_idx], rng) + 1
    stream.sort_values("step", kind="stable").to_parquet(PROCESSED_DIR / "test_stream.parquet", index=False)

    with open(PROCESSED_DIR / "scaler.json", "w") as f:
        json.dump({"continuous": CONTINUOUS, "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    summary["test"] = {"rows": int(len(test_idx)), "fraud": int(y[test_idx].sum())}
    with open(RESULTS_DIR / "data_split.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Bank':<6}{'Rows':>12}{'Fraud':>8}{'TRANSFER fraud':>16}{'CASH_OUT fraud':>16}")
    for s in summary["banks"]:
        print(f"{s['bank']:<6}{s['rows']:>12,}{s['fraud']:>8,}{s['fraud_TRANSFER']:>16,}{s['fraud_CASH_OUT']:>16,}")
    print(f"{'Test':<6}{len(test_idx):>12,}{int(y[test_idx].sum()):>8,}")
    print(f"\nSaved bank files, test set and scaler to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
