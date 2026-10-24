"""Stage 3: the two non-federated reference points.

  isolated     each bank trains only on its own data (today's reality)
  centralised  all banks' data pooled (the upper limit, not allowed in real life)

Both get TOTAL_EPOCHS passes over their data, the same as the federated run
(NUM_ROUNDS x LOCAL_EPOCHS), with the same learning-rate schedule, and are
scored on the common test set.
"""

import argparse
import json
import time

import numpy as np
import torch

from config import LR_DECAY, NUM_BANKS, SEED, SPLITS, TOTAL_EPOCHS, model_dir, run_dir
from evaluate import compute_metrics, format_metrics
from model import FraudNet, load_bank, load_test, predict_proba, train_model


def fit_and_score(
    X: np.ndarray, y: np.ndarray, epochs: int, seed: int, lr_decay: float, split: str, out_seed: int, name: str
) -> dict:
    torch.manual_seed(seed)
    model = FraudNet()
    start = time.time()
    train_model(model, X, y, epochs=epochs, seed=seed, lr_decay=lr_decay)
    test = load_test(split)
    metrics = compute_metrics(test["y"], predict_proba(model, test["X"]), test["t"])
    metrics["train_rows"] = int(len(y))
    metrics["train_seconds"] = round(time.time() - start, 1)
    torch.save(model.state_dict(), model_dir(split, out_seed) / f"{name}.pt")
    print(f"{name:<22} {format_metrics(metrics)}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--lr-decay", type=float, default=LR_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split", choices=SPLITS, default="random")
    args = parser.parse_args()
    model_dir(args.split, args.seed).mkdir(parents=True, exist_ok=True)
    run_dir(args.split, args.seed).mkdir(parents=True, exist_ok=True)

    def fit(X, y, seed, name):
        return fit_and_score(X, y, args.epochs, seed, args.lr_decay, args.split, args.seed, name)

    results = {"split": args.split, "seed": args.seed, "epochs": args.epochs, "lr_decay": args.lr_decay,
               "isolated": {}}
    for b in range(NUM_BANKS):
        d = load_bank(b, args.split)
        results["isolated"][f"bank_{b + 1}"] = fit(d["X_train"], d["y_train"], args.seed + b, f"isolated_bank_{b + 1}")

    banks = [load_bank(b, args.split) for b in range(NUM_BANKS)]
    X_all = np.concatenate([d["X_train"] for d in banks])
    y_all = np.concatenate([d["y_train"] for d in banks])
    results["centralised"] = fit(X_all, y_all, args.seed, "centralised")

    with open(run_dir(args.split, args.seed) / "baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {run_dir(args.split, args.seed) / 'baseline.json'}")


if __name__ == "__main__":
    main()
