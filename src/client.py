"""Stage 4: the Flower ClientApp, one per bank.

A bank receives the current global model, trains it on its own transactions
(optionally with differential privacy) and sends back only the updated weights
and a few summary numbers. Its raw data never leaves this function.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import average_precision_score

from config import SEED
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
    server_round, local_epochs = int(cfg["server-round"]), int(cfg["local-epochs"])
    data = load_bank(bank)
    X, y = data["X_train"], data["y_train"]

    torch.manual_seed(SEED + 100 * server_round + bank)
    model = _global_model(msg)
    metrics = {"num-examples": len(y)}

    dp_epsilon = float(cfg["dp-epsilon"])
    if dp_epsilon > 0:
        sigma = noise_for_budget(dp_epsilon, len(y), int(cfg["num-rounds"]) * local_epochs)
        loss = train_dp(model, X, y, local_epochs, sigma, lr=float(cfg["lr"]))
        metrics["epsilon"] = epsilon_spent(sigma, len(y), server_round * local_epochs)
    else:
        loss = train_model(model, X, y, local_epochs, lr=float(cfg["lr"]), seed=SEED + server_round)
    metrics["train_loss"] = loss

    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Score the global model on this bank's own validation data."""
    data = load_bank(int(context.node_config["partition-id"]))
    prob = predict_proba(_global_model(msg), data["X_val"])
    metrics = MetricRecord({
        "num-examples": len(data["y_val"]),
        "val_pr_auc": float(average_precision_score(data["y_val"], prob)),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
