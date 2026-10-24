"""PyTorch fraud model, data loading and the plain (non-private) training loop."""

from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from config import BATCH_SIZE, LEARNING_RATE, POS_WEIGHT, processed_dir
from preprocess import FEATURES


class FraudNet(nn.Module):
    """Small feed-forward network: 3 linear layers with ReLU and dropout.

    Returns logits; apply a sigmoid to get fraud probabilities. No BatchNorm, so
    Opacus can compute per-sample gradients for differential privacy.
    """

    def __init__(self, n_features: int = len(FEATURES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TensorBatches(Dataset):
    """In-memory dataset that returns whole batches at once via __getitems__.

    The default DataLoader path indexes one row at a time, which is very slow for
    a million rows; this keeps Opacus's Poisson-sampling loader fast.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return self.X[idx], self.y[idx]


def keep_batch(batch):
    """collate_fn for TensorBatches: the batch is already stacked."""
    return batch


@lru_cache(maxsize=None)
def load_bank(bank: int, split: str = "random") -> dict[str, np.ndarray]:
    """Training and validation arrays for one bank (0-based id)."""
    with np.load(processed_dir(split) / f"bank_{bank}.npz") as d:
        return {k: d[k] for k in d.files}


@lru_cache(maxsize=None)
def load_test(split: str = "random") -> dict[str, np.ndarray]:
    with np.load(processed_dir(split) / "test.npz") as d:
        return {k: d[k] for k in d.files}


def train_model(
    model: FraudNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
    lr_decay: float = 1.0,
) -> float:
    """Train with a class-weighted loss, multiplying the learning rate by lr_decay
    after every epoch. Returns the mean loss of the last epoch."""
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = lr * lr_decay**epoch
        perm = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for start in range(0, len(yt), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
    return total / len(yt)


@torch.no_grad()
def predict_proba(model: FraudNet, X: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    model.eval()
    Xt = torch.from_numpy(X)
    out = [torch.sigmoid(model(Xt[i : i + batch_size])) for i in range(0, len(Xt), batch_size)]
    return torch.cat(out).numpy()
