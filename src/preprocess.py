"""Stage 2: clean PaySim, engineer features and split it into 5 simulated banks.

  python src/preprocess.py                 # random test set (main benchmark)
  python src/preprocess.py --split time    # latest 20% of transactions as the test set

Outputs (in data/processed/<split>/):
  bank_<i>.npz     training and local validation data for bank i (0-based)
  test.npz         common held-out test set
  test_stream.parquet  raw test transactions, tagged with a bank, for Kafka replay
  scaler.json      feature means/stds built from noised per-bank aggregate statistics
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DP_SCALER_NOISE,
    FRAUD_SPLIT,
    LEGIT_SPLIT,
    NUM_BANKS,
    RAW_CSV,
    RESULTS_DIR,
    SCALER_BOUND,
    SEED,
    SPLITS,
    TEST_FRACTION,
    TX_TYPES,
    VAL_FRACTION,
    processed_dir,
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


def random_split(strata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random test set with the same type/fraud mix as the full data."""
    train_idx, test_idx = train_test_split(
        np.arange(len(strata)), test_size=TEST_FRACTION, stratify=strata, random_state=SEED
    )
    return np.sort(train_idx), np.sort(test_idx)


def time_split(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Everything after the hour by which (1 - TEST_FRACTION) of transactions happened is the test set."""
    hours, counts = np.unique(steps, return_counts=True)
    cut = hours[np.searchsorted(np.cumsum(counts) / len(steps), 1 - TEST_FRACTION)]
    return np.flatnonzero(steps <= cut), np.flatnonzero(steps > cut)


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


def federated_scaler(
    X: np.ndarray, bank: np.ndarray, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the continuous features from per-bank (count, sum, sum of squares).

    Each bank shares only its row count and two noised sums per feature, never rows.
    Values are clipped to [-SCALER_BOUND, SCALER_BOUND], so one transaction changes
    the sums by at most `sensitivity`, and Gaussian noise of `noise * sensitivity`
    makes the release differentially private (accounted for in privacy.py).
    """
    n_cont = len(CONTINUOUS)
    sensitivity = np.sqrt(n_cont * SCALER_BOUND**2 + n_cont * SCALER_BOUND**4)
    count, total, total_sq = 0, np.zeros(n_cont), np.zeros(n_cont)
    for b in range(NUM_BANKS):
        Xb = np.clip(X[bank == b, :n_cont].astype(np.float64), -SCALER_BOUND, SCALER_BOUND)
        count += len(Xb)
        total += Xb.sum(axis=0) + rng.normal(0, noise * sensitivity, n_cont)
        total_sq += (Xb**2).sum(axis=0) + rng.normal(0, noise * sensitivity, n_cont)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-6))
    return mean, std


def scale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, : len(mean)] = (X[:, : len(mean)] - mean) / std
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="random")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    out = processed_dir(args.split)
    out.mkdir(parents=True, exist_ok=True)
    summary_dir = RESULTS_DIR / "runs" / args.split
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, engine="pyarrow", usecols=RAW_COLUMNS)
    print(f"  {len(df):,} transactions, {df.isFraud.sum():,} fraud")

    X = build_features(df)
    y = df["isFraud"].to_numpy(np.int8)
    t = pd.Categorical(df["type"], categories=TX_TYPES).codes.astype(np.int8)
    strata = t * 2 + y

    if args.split == "time":
        train_idx, test_idx = time_split(df["step"].to_numpy())
    else:
        train_idx, test_idx = random_split(strata)

    bank = assign_banks(t[train_idx], y[train_idx], rng)
    mean, std = federated_scaler(X[train_idx], bank, DP_SCALER_NOISE, rng)
    exact_mean, exact_std = federated_scaler(X[train_idx], bank, 0.0, np.random.default_rng())  # for reporting only
    X = scale(X, mean, std)

    summary = {"split": args.split, "features": FEATURES, "banks": []}
    for b in range(NUM_BANKS):
        rows = train_idx[bank == b]
        tr, va = train_test_split(
            rows, test_size=VAL_FRACTION, stratify=strata[rows], random_state=SEED
        )
        np.savez(
            out / f"bank_{b}.npz",
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

    np.savez(out / "test.npz", X=X[test_idx], y=y[test_idx], t=t[test_idx])

    # Raw test transactions for the Kafka producer, tagged with the bank that receives them.
    stream = df.iloc[test_idx].copy()
    stream["bank"] = assign_banks(t[test_idx], y[test_idx], rng) + 1
    stream.sort_values("step", kind="stable").to_parquet(out / "test_stream.parquet", index=False)

    with open(out / "scaler.json", "w") as f:
        json.dump({"continuous": CONTINUOUS, "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    summary["test"] = {
        "rows": int(len(test_idx)),
        "fraud": int(y[test_idx].sum()),
        "first_hour": int(df["step"].iloc[test_idx].min()),
    }
    with open(summary_dir / "data_split.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Bank':<6}{'Rows':>12}{'Fraud':>8}{'TRANSFER fraud':>16}{'CASH_OUT fraud':>16}")
    for s in summary["banks"]:
        print(f"{s['bank']:<6}{s['rows']:>12,}{s['fraud']:>8,}{s['fraud_TRANSFER']:>16,}{s['fraud_CASH_OUT']:>16,}")
    print(f"{'Test':<6}{len(test_idx):>12,}{int(y[test_idx].sum()):>8,}  (from hour {summary['test']['first_hour']})")
    print(f"\nPrivacy noise on scaling statistics: means off by at most {np.abs(mean - exact_mean).max():.4f}, "
          f"standard deviations by at most {np.abs(std / exact_std - 1).max():.2%}")
    print(f"Saved bank files, test set and scaler to {out}")


if __name__ == "__main__":
    main()
