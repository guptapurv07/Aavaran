"""Stage 4: the Flower ServerApp (the central aggregator).

Each round it sends the global model to every bank, combines their updated
weights with FedAvg (an average weighted by how many transactions each bank
trained on), and scores the new global model on the common test set.
"""

import json

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from config import LEARNING_RATE, NUM_BANKS, model_dir, run_dir
from evaluate import compute_metrics, format_metrics
from model import FraudNet, load_test, predict_proba


def aggregate_train_metrics(records: list[RecordDict], weight_key: str) -> MetricRecord:
    """Example-weighted mean training loss (no-DP runs), and the largest privacy budget any bank has spent."""
    metrics = [next(iter(r.metric_records.values())) for r in records]
    out = MetricRecord()
    if "train_loss" in metrics[0]:
        total = sum(m[weight_key] for m in metrics)
        out["train_loss"] = sum(m["train_loss"] * m[weight_key] for m in metrics) / total
    if "epsilon" in metrics[0]:
        out["epsilon_max"] = max(m["epsilon"] for m in metrics)
    return out


def build_server_app(
    num_rounds: int, local_epochs: int, dp_epsilon: float, lr_decay: float, split: str, seed: int, run_name: str
) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        torch.manual_seed(seed)
        model = FraudNet()
        test = load_test(split)
        history = []

        def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            model.load_state_dict(arrays.to_torch_state_dict())
            metrics = compute_metrics(test["y"], predict_proba(model, test["X"]), test["t"])
            history.append({"round": server_round, **metrics})
            print(f"[{split}/seed {seed}/{run_name}] round {server_round:>2}: {format_metrics(metrics)}", flush=True)
            return MetricRecord(metrics)

        strategy = FedAvg(
            # Under DP, banks report no validation scores: those would leak outside the budget.
            fraction_evaluate=0.0 if dp_epsilon > 0 else 1.0,
            min_train_nodes=NUM_BANKS,
            min_evaluate_nodes=NUM_BANKS,
            min_available_nodes=NUM_BANKS,
            train_metrics_aggr_fn=aggregate_train_metrics,
        )
        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(model.state_dict()),
            num_rounds=num_rounds,
            train_config=ConfigRecord({
                "lr": LEARNING_RATE,
                "lr-decay": lr_decay,
                "local-epochs": local_epochs,
                "num-rounds": num_rounds,
                "dp-epsilon": dp_epsilon,
                "split": split,
                "seed": seed,
            }),
            evaluate_config=ConfigRecord({"split": split}),
            evaluate_fn=global_evaluate,
        )

        model_dir(split, seed).mkdir(parents=True, exist_ok=True)
        torch.save(result.arrays.to_torch_state_dict(), model_dir(split, seed) / f"{run_name}.pt")
        for entry in history[1:]:
            r = entry["round"]
            entry.update(dict(result.train_metrics_clientapp.get(r, {})))
            entry.update(dict(result.evaluate_metrics_clientapp.get(r, {})))
        run_dir(split, seed).mkdir(parents=True, exist_ok=True)
        with open(run_dir(split, seed) / f"{run_name}.json", "w") as f:
            json.dump({
                "run": run_name,
                "split": split,
                "seed": seed,
                "rounds": num_rounds,
                "local_epochs": local_epochs,
                "lr_decay": lr_decay,
                "dp_epsilon": dp_epsilon or None,
                "final": history[-1],
                "history": history,
            }, f, indent=2)

    return app
