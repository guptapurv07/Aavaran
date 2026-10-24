import json

import numpy as np
import pandas as pd

from config import TX_TYPES
from preprocess import FEATURES, build_features, scale


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "step": [1, 13, 25, 400],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"],
        "amount": [9839.64, 181.0, 181.0, 5000.0],
        "nameOrig": ["C1", "C2", "C3", "C4"],
        "oldbalanceOrg": [170136.0, 181.0, 181.0, 100.0],
        "newbalanceOrig": [160296.36, 0.0, 0.0, 5100.0],
        "nameDest": ["M1", "C5", "C6", "C7"],
        "oldbalanceDest": [0.0, 0.0, 21182.0, 0.0],
        "newbalanceDest": [0.0, 0.0, 0.0, 0.0],
        "isFraud": [0, 1, 1, 0],
    })


def test_features_have_the_expected_shape_and_meaning():
    X = build_features(sample_transactions())
    col = {name: X[:, i] for i, name in enumerate(FEATURES)}

    assert X.shape == (4, len(FEATURES)) and X.dtype == np.float32
    np.testing.assert_allclose(col["log_amount"], np.log1p([9839.64, 181.0, 181.0, 5000.0]), rtol=1e-6)
    assert col["orig_drained"].tolist() == [0, 1, 1, 0]  # the two frauds empty the sender's account
    assert col["type_TRANSFER"].tolist() == [0, 1, 0, 0]
    one_hot = X[:, [FEATURES.index(f"type_{t}") for t in TX_TYPES]]
    assert (one_hot.sum(axis=1) == 1).all()


def test_live_scoring_builds_exactly_the_training_features():
    """Guards against training/serving skew: a transaction that travels through
    Kafka as JSON must get the same features the model was trained on."""
    df = sample_transactions()
    mean, std = np.array([10.0, 8.0, 7.0, 1.0]), np.array([2.0, 5.0, 5.0, 3.0])
    expected = scale(build_features(df), mean, std)

    messages = [json.dumps({**rec, "bank": 1, "tx_id": i, "sent_at": 0.0})
                for i, rec in enumerate(df.to_dict("records"))]
    live = pd.DataFrame.from_records([json.loads(m) for m in messages])

    np.testing.assert_array_equal(scale(build_features(live), mean, std), expected)
