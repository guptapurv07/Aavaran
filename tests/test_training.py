import numpy as np
import pytest
import torch
from flwr.app import MetricRecord, RecordDict
from sklearn.metrics import average_precision_score

from config import TX_TYPES
from evaluate import compute_metrics
from model import FraudNet, predict_proba, train_model
from privacy import train_dp
from server import aggregate_train_metrics


def toy_data(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 2.0).astype(np.int8)  # about 8% positives, learnable
    return X, y


def test_training_learns_a_simple_pattern():
    X, y = toy_data()
    torch.manual_seed(0)
    model = FraudNet()
    train_model(model, X, y, epochs=5, batch_size=256, seed=0, lr_decay=0.9)
    assert average_precision_score(y, predict_proba(model, X)) > 0.9


def test_dp_training_updates_the_plain_model():
    X, y = toy_data()
    torch.manual_seed(0)
    model = FraudNet()
    before = [p.detach().clone() for p in model.parameters()]
    train_dp(model, X, y, epochs=1, noise_multiplier=1.0)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert set(model.state_dict()) == set(FraudNet().state_dict())  # no Opacus wrapper left behind


def test_metrics_on_a_known_example():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3])
    transfer, cash_out = TX_TYPES.index("TRANSFER"), TX_TYPES.index("CASH_OUT")
    types = np.array([transfer] * 4 + [cash_out] * 2)
    m = compute_metrics(y, prob, types)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarms"] == 1
    assert m["recall_TRANSFER"] == 1.0 and m["recall_CASH_OUT"] == 0.5


def reply(**metrics) -> RecordDict:
    return RecordDict({"metrics": MetricRecord(metrics)})


def test_server_weights_training_loss_by_bank_size():
    out = aggregate_train_metrics(
        [reply(**{"num-examples": 100, "train_loss": 1.0}), reply(**{"num-examples": 300, "train_loss": 3.0})],
        "num-examples",
    )
    assert out["train_loss"] == pytest.approx(2.5)


def test_server_reports_only_the_largest_budget_under_dp():
    out = aggregate_train_metrics(
        [reply(**{"num-examples": 100, "epsilon": 0.8}), reply(**{"num-examples": 300, "epsilon": 0.9})],
        "num-examples",
    )
    assert dict(out) == {"epsilon_max": 0.9}
