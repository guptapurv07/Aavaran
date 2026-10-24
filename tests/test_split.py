import numpy as np

from config import FRAUD_SPLIT, NUM_BANKS, TEST_FRACTION, TX_TYPES
from preprocess import assign_banks, time_split

TRANSFER, CASH_OUT = TX_TYPES.index("TRANSFER"), TX_TYPES.index("CASH_OUT")


def synthetic_labels(n: int = 50_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    types = rng.integers(0, len(TX_TYPES), n).astype(np.int8)
    fraud = (rng.random(n) < 0.05) & np.isin(types, [TRANSFER, CASH_OUT])
    return types, fraud.astype(np.int8)


def test_banks_never_see_the_fraud_type_they_are_meant_to_miss():
    types, labels = synthetic_labels()
    bank = assign_banks(types, labels, np.random.default_rng(1))
    for tx_type, shares in FRAUD_SPLIT.items():
        for b, share in enumerate(shares):
            n = ((bank == b) & (types == TX_TYPES.index(tx_type)) & (labels == 1)).sum()
            assert (n == 0) if share == 0 else (n > 0)


def test_every_row_gets_a_bank_in_the_configured_shares():
    types, labels = synthetic_labels()
    bank = assign_banks(types, labels, np.random.default_rng(1))
    assert set(np.unique(bank)) == set(range(NUM_BANKS))
    transfer_fraud = (types == TRANSFER) & (labels == 1)
    shares = np.bincount(bank[transfer_fraud], minlength=NUM_BANKS) / transfer_fraud.sum()
    np.testing.assert_allclose(shares, FRAUD_SPLIT["TRANSFER"], atol=0.01)


def test_time_split_tests_only_on_later_hours():
    steps = np.repeat(np.arange(1, 101), 10)  # 100 hours, 10 transactions each
    train, test = time_split(steps)
    assert steps[train].max() < steps[test].min()
    assert len(train) + len(test) == len(steps)
    assert abs(len(test) / len(steps) - TEST_FRACTION) < 0.02
