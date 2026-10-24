"""Stage 4: the Flower ClientApp, one per bank.

A bank receives the current global model, trains it on its own transactions
(optionally with differential privacy) and sends back only the updated weights
and a few summary numbers. Its raw data never leaves this function. Under
differential privacy it sends no scores at all: only the weights, its row
count and the privacy budget spent so far.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import average_precision_score

from model import FraudNet, load_bank, predict_proba, train_model
from privacy import epsilon_spent, noise_for_budget, train_dp

app = ClientApp()


def _global_model(msg: Message) -> FraudNet:
    model = FraudNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


@app.train()
def train(msg: Message, context: Context) -> Message:
    bank = int(context.node_config["partition-id"])
    cfg = msg.content["config"]
    server_round, local_epochs, seed = int(cfg["server-round"]), int(cfg["local-epochs"]), int(cfg["seed"])
    data = load_bank(bank, str(cfg["split"]))
    X, y = data["X_train"], data["y_train"]

    # Continue the same per-epoch learning-rate schedule the baselines use.
    decay = float(cfg["lr-decay"])
    lr = float(cfg["lr"]) * decay ** ((server_round - 1) * local_epochs)

    torch.manual_seed(seed + 100 * server_round + bank)
    model = _global_model(msg)
    metrics = {"num-examples": len(y)}

    dp_epsilon = float(cfg["dp-epsilon"])
    if dp_epsilon > 0:
        sigma = noise_for_budget(dp_epsilon, len(y), int(cfg["num-rounds"]) * local_epochs)
        train_dp(model, X, y, local_epochs, sigma, lr=lr, lr_decay=decay)
        metrics["epsilon"] = epsilon_spent(sigma, len(y), server_round * local_epochs)
    else:
        metrics["train_loss"] = train_model(
            model, X, y, local_epochs, lr=lr, seed=seed + server_round, lr_decay=decay
        )

    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Score the global model on this bank's own validation data (runs without DP only)."""
    data = load_bank(int(context.node_config["partition-id"]), str(msg.content["config"]["split"]))
    prob = predict_proba(_global_model(msg), data["X_val"])
    metrics = MetricRecord({
        "num-examples": len(data["y_val"]),
        "val_pr_auc": float(average_precision_score(data["y_val"], prob)),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
