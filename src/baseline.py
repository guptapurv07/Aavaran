"""Stage 3: the two non-federated reference points.

  isolated     each bank trains only on its own data (today's reality)
  centralised  all banks' data pooled (the upper limit, not allowed in real life)

Both get TOTAL_EPOCHS passes over their data, the same as the federated run
(NUM_ROUNDS x LOCAL_EPOCHS), and are scored on the common test set.
"""

import argparse
import json
import time

import numpy as np
import torch

from config import MODELS_DIR, NUM_BANKS, RESULTS_DIR, SEED, TOTAL_EPOCHS
from evaluate import compute_metrics, format_metrics
from model import FraudNet, load_bank, load_test, predict_proba, train_model


def fit_and_score(X: np.ndarray, y: np.ndarray, epochs: int, seed: int, name: str) -> dict:
    torch.manual_seed(seed)
    model = FraudNet()
    start = time.time()
    train_model(model, X, y, epochs=epochs, seed=seed)
    test = load_test()
    metrics = compute_metrics(test["y"], predict_proba(model, test["X"]), test["t"])
    metrics["train_rows"] = int(len(y))
    metrics["train_seconds"] = round(time.time() - start, 1)
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    print(f"{name:<22} {format_metrics(metrics)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    args = parser.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {"epochs": args.epochs, "isolated": {}}
    for b in range(NUM_BANKS):
        d = load_bank(b)
        results["isolated"][f"bank_{b + 1}"] = fit_and_score(
            d["X_train"], d["y_train"], args.epochs, SEED + b, f"isolated_bank_{b + 1}"
        )

    banks = [load_bank(b) for b in range(NUM_BANKS)]
    X_all = np.concatenate([d["X_train"] for d in banks])
    y_all = np.concatenate([d["y_train"] for d in banks])
    results["centralised"] = fit_and_score(X_all, y_all, args.epochs, SEED, "centralised")

    with open(RESULTS_DIR / "baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'baseline.json'}")


if __name__ == "__main__":
    main()
