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

from config import LEARNING_RATE, MODELS_DIR, NUM_BANKS, RESULTS_DIR, SEED
from evaluate import compute_metrics, format_metrics
from model import FraudNet, load_test, predict_proba


def aggregate_train_metrics(records: list[RecordDict], weight_key: str) -> MetricRecord:
    """Example-weighted mean training loss, plus the largest privacy budget any bank has spent."""
    metrics = [next(iter(r.metric_records.values())) for r in records]
    total = sum(m[weight_key] for m in metrics)
    out = MetricRecord({"train_loss": sum(m["train_loss"] * m[weight_key] for m in metrics) / total})
    if "epsilon" in metrics[0]:
        out["epsilon_max"] = max(m["epsilon"] for m in metrics)
    return out


def build_server_app(num_rounds: int, local_epochs: int, dp_epsilon: float, run_name: str) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        torch.manual_seed(SEED)
        model = FraudNet()
        test = load_test()
        history = []

        def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            model.load_state_dict(arrays.to_torch_state_dict())
            metrics = compute_metrics(test["y"], predict_proba(model, test["X"]), test["t"])
            history.append({"round": server_round, **metrics})
            print(f"[{run_name}] round {server_round:>2}: {format_metrics(metrics)}", flush=True)
            return MetricRecord(metrics)

        strategy = FedAvg(
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
                "local-epochs": local_epochs,
                "num-rounds": num_rounds,
                "dp-epsilon": dp_epsilon,
            }),
            evaluate_fn=global_evaluate,
        )

        MODELS_DIR.mkdir(exist_ok=True)
        torch.save(result.arrays.to_torch_state_dict(), MODELS_DIR / f"{run_name}.pt")
        for entry in history[1:]:
            r = entry["round"]
            entry.update(dict(result.train_metrics_clientapp.get(r, {})))
            entry.update(dict(result.evaluate_metrics_clientapp.get(r, {})))
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / f"{run_name}.json", "w") as f:
            json.dump({
                "run": run_name,
                "rounds": num_rounds,
                "local_epochs": local_epochs,
                "dp_epsilon": dp_epsilon or None,
                "final": history[-1],
                "history": history,
            }, f, indent=2)

    return app
