"""Stages 4 and 5: run Aavaran's federated training as a Flower simulation.

Five banks (Flower SuperNodes) train together for NUM_ROUNDS rounds.

  python src/federated.py                   # plain FedAvg
  python src/federated.py --dp-epsilon 5    # FedAvg + DP-SGD at epsilon = 5
  python src/federated.py --seed 43 --split time
"""

import argparse
import os
from pathlib import Path

from flwr.simulation import run_simulation

from client import app as client_app
from config import DP_LR_DECAY, LOCAL_EPOCHS, LR_DECAY, NUM_BANKS, NUM_ROUNDS, SEED, SPLITS
from server import build_server_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--dp-epsilon", type=float, default=0.0, help="privacy budget per bank; 0 = no DP")
    parser.add_argument("--lr-decay", type=float, help=f"default {LR_DECAY}, or {DP_LR_DECAY} with DP")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split", choices=SPLITS, default="random")
    parser.add_argument("--name", help="result name (default: federated or federated_dp_eps<E>)")
    parser.add_argument("--cpus-per-bank", type=float, default=1.6, help="1.6 lets all 5 banks train at once on 8 cores")
    args = parser.parse_args()

    eps = args.dp_epsilon
    run_name = args.name or (f"federated_dp_eps{eps:g}" if eps > 0 else "federated")
    lr_decay = args.lr_decay if args.lr_decay is not None else (DP_LR_DECAY if eps > 0 else LR_DECAY)

    # Ray workers that run the banks need to import our modules from src/.
    src = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))

    run_simulation(
        server_app=build_server_app(args.rounds, args.local_epochs, eps, lr_decay, args.split, args.seed, run_name),
        client_app=client_app,
        num_supernodes=NUM_BANKS,
        backend_config={"client_resources": {"num_cpus": args.cpus_per_bank, "num_gpus": 0.0}},
    )


if __name__ == "__main__":
    main()
