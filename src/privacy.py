"""Stage 5: differential privacy for each bank's local training, using Opacus.

Each bank runs DP-SGD: every transaction's gradient is clipped to DP_MAX_GRAD_NORM
and Gaussian noise is added to each batch update, with batches drawn by Poisson
sampling. The noise level is set once from the bank's privacy budget (epsilon)
for the *whole* federated run, so after the last round each bank has spent
exactly its target epsilon on its own customers, however many rounds there were.
"""

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.utils.data import DataLoader

from config import DP_BATCH_SIZE, DP_DELTA, DP_MAX_GRAD_NORM, LEARNING_RATE, POS_WEIGHT
from model import FraudNet, TensorBatches, keep_batch


def noise_for_budget(target_epsilon: float, n_rows: int, total_epochs: int) -> float:
    """Noise multiplier that spends exactly target_epsilon over total_epochs."""
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=DP_DELTA,
        sample_rate=DP_BATCH_SIZE / n_rows,
        epochs=total_epochs,
        accountant="rdp",
    )


def epsilon_spent(noise_multiplier: float, n_rows: int, epochs_done: int) -> float:
    """Privacy budget a bank has used after epochs_done epochs of DP-SGD."""
    sample_rate = DP_BATCH_SIZE / n_rows
    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, int(epochs_done / sample_rate))]
    return accountant.get_epsilon(DP_DELTA)


def train_dp(
    model: FraudNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    noise_multiplier: float,
    lr: float = LEARNING_RATE,
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
    for _ in range(epochs):
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
