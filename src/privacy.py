"""Stage 5: differential privacy for each bank's local training, using Opacus.

Each bank runs DP-SGD: every transaction's gradient is clipped to DP_MAX_GRAD_NORM
and Gaussian noise is added to each batch update, with batches drawn by Poisson
sampling. The noise level is set once from the bank's privacy budget (epsilon)
for the *whole* federated run, so after the last round each bank has spent
exactly its target epsilon on its own customers, however many rounds there were.

The budget also pays for the one-off release of noised feature-scaling
statistics before training (see preprocess.federated_scaler), so it covers
everything a bank shares except its row count.
"""

from functools import lru_cache

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from torch import nn
from torch.utils.data import DataLoader

from config import (
    DP_BATCH_SIZE, DP_DELTA, DP_MAX_GRAD_NORM, DP_SCALER_NOISE, LEARNING_RATE, POS_WEIGHT,
)
from model import FraudNet, TensorBatches, keep_batch

# Renyi orders to search; going up to 512 keeps the bound tight at larger epsilons.
ALPHAS = [1 + x / 10 for x in range(1, 100)] + list(range(12, 512))


def _epsilon(noise_multiplier: float, sample_rate: float, steps: int) -> float:
    """Total epsilon of the scaling-statistics release plus `steps` DP-SGD steps."""
    accountant = RDPAccountant()
    accountant.history = [(DP_SCALER_NOISE, 1.0, 1), (noise_multiplier, sample_rate, steps)]
    return accountant.get_epsilon(DP_DELTA, alphas=ALPHAS)


@lru_cache(maxsize=None)  # each bank asks every round; the answer never changes within a run
def noise_for_budget(target_epsilon: float, n_rows: int, total_epochs: int) -> float:
    """Smallest noise multiplier whose total spend over total_epochs stays within target_epsilon."""
    sample_rate = DP_BATCH_SIZE / n_rows
    steps = int(total_epochs / sample_rate)
    lo, hi = 0.1, 1000.0  # epsilon falls as the noise rises
    if _epsilon(hi, sample_rate, steps) > target_epsilon:
        raise ValueError(f"epsilon {target_epsilon} is too small to cover the scaling statistics")
    while hi - lo > 1e-4 * hi:
        mid = (lo + hi) / 2
        if _epsilon(mid, sample_rate, steps) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def epsilon_spent(noise_multiplier: float, n_rows: int, epochs_done: int) -> float:
    """Privacy budget a bank has used after epochs_done epochs of DP-SGD."""
    sample_rate = DP_BATCH_SIZE / n_rows
    return _epsilon(noise_multiplier, sample_rate, int(epochs_done / sample_rate))


def train_dp(
    model: FraudNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    noise_multiplier: float,
    lr: float = LEARNING_RATE,
    lr_decay: float = 1.0,
) -> float:
    """DP-SGD training through Opacus's PrivacyEngine. Returns the last epoch's mean loss."""
    loader = DataLoader(TensorBatches(X, y), batch_size=DP_BATCH_SIZE, collate_fn=keep_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    engine = PrivacyEngine(accountant="rdp")
    dp_model, dp_optimizer, dp_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=DP_MAX_GRAD_NORM,
        poisson_sampling=True,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))

    dp_model.train()
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = lr * lr_decay**epoch
        total, seen = 0.0, 0
        for Xb, yb in dp_loader:
            if len(yb) == 0:  # Poisson sampling can, very rarely, draw an empty batch
                continue
            dp_optimizer.zero_grad()
            loss = loss_fn(dp_model(Xb), yb)
            loss.backward()
            dp_optimizer.step()
            total += loss.item() * len(yb)
            seen += len(yb)
    dp_model.remove_hooks()  # `model` itself now holds the trained weights
    return total / max(seen, 1)
