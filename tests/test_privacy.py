import numpy as np
import pytest

from config import NUM_BANKS, SCALER_BOUND
from preprocess import CONTINUOUS, federated_scaler
from privacy import epsilon_spent, noise_for_budget

ROWS, EPOCHS = 1_000_000, 20


@pytest.mark.parametrize("epsilon", [1.0, 5.0, 10.0])
def test_each_bank_spends_exactly_its_budget_by_the_last_round(epsilon):
    sigma = noise_for_budget(epsilon, ROWS, EPOCHS)
    spent = epsilon_spent(sigma, ROWS, EPOCHS)
    assert spent <= epsilon
    assert spent == pytest.approx(epsilon, rel=1e-3)


def test_privacy_spent_grows_round_by_round():
    sigma = noise_for_budget(5.0, ROWS, EPOCHS)
    spent = [epsilon_spent(sigma, ROWS, e) for e in (1, 5, 10, 20)]
    assert spent == sorted(spent)


def test_stronger_privacy_needs_more_noise():
    assert noise_for_budget(1.0, ROWS, EPOCHS) > noise_for_budget(10.0, ROWS, EPOCHS)


def scaler_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal([10, 8, 7, 0], [2, 5, 5, 3], size=(20_000, len(CONTINUOUS))).astype(np.float32)
    return X, rng.integers(0, NUM_BANKS, len(X))


def test_scaler_without_noise_equals_the_pooled_statistics():
    X, bank = scaler_inputs()
    mean, std = federated_scaler(X, bank, noise=0.0, rng=np.random.default_rng(0))
    pooled = np.clip(X.astype(np.float64), -SCALER_BOUND, SCALER_BOUND)  # the scaler clips by design
    np.testing.assert_allclose(mean, pooled.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(std, pooled.std(axis=0), rtol=1e-9)


def test_scaler_adds_privacy_noise():
    X, bank = scaler_inputs()
    exact, _ = federated_scaler(X, bank, noise=0.0, rng=np.random.default_rng(0))
    noisy, _ = federated_scaler(X, bank, noise=1.0, rng=np.random.default_rng(0))
    assert not np.allclose(exact, noisy)
