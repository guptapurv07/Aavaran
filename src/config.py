"""Shared paths and settings for every Aavaran script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "paysim.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42  # fixes the test set and the bank split
SEEDS = [42, 43, 44]  # training seeds: every experiment is repeated once per seed
NUM_BANKS = 5
TEST_FRACTION = 0.20  # common held-out test set, never seen by any bank
VAL_FRACTION = 0.10  # each bank keeps 10% of its own data for local validation

# How the common test set is chosen:
#   random  a stratified random 20% of all transactions (the main benchmark)
#   time    the latest 20% of transactions by hour, so models are tested on the
#           future, the way fraud teams validate ("out-of-time" testing)
SPLITS = ["random", "time"]


def processed_dir(split: str) -> Path:
    return ROOT / "data" / "processed" / split


def run_dir(split: str, seed: int) -> Path:
    return RESULTS_DIR / "runs" / split / f"seed_{seed}"


def model_dir(split: str, seed: int) -> Path:
    return MODELS_DIR / split / f"seed_{seed}"


TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Non-IID split: share of each (transaction type, label) group that goes to
# each bank. Each row sums to 1. Fraud is split "pathologically" (as in the
# FedAvg paper): Banks 1 and 3 have never seen CASH_OUT fraud, Banks 2 and 4
# have never seen TRANSFER fraud, and Bank 5 is a small bank that sees a little
# of both. Every bank still has plenty of legitimate transactions of each type.
# A softer split (2-3% of the other fraud type) let every bank learn both
# patterns alone; see results/baseline_soft_split_all_features.json.
FRAUD_SPLIT = {
    "TRANSFER": [0.60, 0.00, 0.30, 0.00, 0.10],
    "CASH_OUT": [0.00, 0.60, 0.00, 0.30, 0.10],
}
LEGIT_SPLIT = {
    "CASH_IN": [0.30, 0.15, 0.20, 0.25, 0.10],
    "CASH_OUT": [0.15, 0.35, 0.15, 0.25, 0.10],
    "DEBIT": [0.20, 0.20, 0.20, 0.20, 0.20],
    "PAYMENT": [0.25, 0.15, 0.30, 0.20, 0.10],
    "TRANSFER": [0.35, 0.10, 0.25, 0.20, 0.10],
}

# Training settings, identical for every setup so comparisons are fair.
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
LR_DECAY = 0.9  # learning rate is multiplied by this after every epoch (federated: every round)
# DP-SGD keeps a constant learning rate: with the decay, the noise left DP models
# under-trained after 20 rounds (see results/runs/*/seed_*/federated_dp_eps*_lr_decay.json).
DP_LR_DECAY = 1.0
POS_WEIGHT = 10.0  # weight on the fraud class in the loss (fraud is ~0.13%)
THRESHOLD = 0.5  # probability above which a transaction is flagged
NUM_ROUNDS = 20  # federated rounds
LOCAL_EPOCHS = 1  # epochs each bank trains per round
TOTAL_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS  # isolated/centralised get the same passes

# Real-time scoring (Kafka)
KAFKA_BOOTSTRAP = "localhost:9092"
TRANSACTIONS_TOPIC = "transactions"
ALERTS_TOPIC = "fraud-alerts"

# Differential privacy (Opacus)
DP_EPSILONS = [1.0, 5.0, 10.0]
DP_DELTA = 1e-7  # well below 1 / (rows per bank)
DP_MAX_GRAD_NORM = 1.0
DP_BATCH_SIZE = 4096  # larger batches give DP-SGD a better signal-to-noise ratio

# Feature-scaling statistics are shared once, before training, with Gaussian
# noise so they are covered by each bank's privacy budget too. Every scaled
# feature is clipped to [-SCALER_BOUND, SCALER_BOUND] (log1p of any PaySim
# amount or balance is below 19), and DP_SCALER_NOISE is the noise multiplier.
# Each bank's row count is treated as public, as FedAvg needs it for weighting.
SCALER_BOUND = 21.0
DP_SCALER_NOISE = 50.0
