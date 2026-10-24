"""Run every experiment once per training seed: the isolated and centralised
baselines, federated training without DP, and federated training at each
privacy budget.

  python src/run_experiments.py                          # random split, seeds 42 43 44
  python src/run_experiments.py --split time --seeds 42

Runs whose results already exist are skipped, so an interrupted run can be
resumed. Each run's full log goes to logs/<split>/seed_<seed>/<run>.log.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import DP_EPSILONS, ROOT, SEEDS, SPLITS, run_dir

SRC = Path(__file__).resolve().parent


def jobs(split: str, seed: int):
    common = ["--split", split, "--seed", str(seed)]
    yield "baseline", [sys.executable, str(SRC / "baseline.py"), *common]
    yield "federated", [sys.executable, str(SRC / "federated.py"), *common]
    for eps in DP_EPSILONS:
        yield f"federated_dp_eps{eps:g}", [sys.executable, str(SRC / "federated.py"), *common, "--dp-epsilon", str(eps)]


def summary(path: Path) -> str:
    result = json.loads(path.read_text())
    if "centralised" in result:
        alone = [m["f1"] for m in result["isolated"].values()]
        return f"F1 banks alone {min(alone):.3f}-{max(alone):.3f}, centralised {result['centralised']['f1']:.3f}"
    return f"F1 {result['final']['f1']:.3f}, PR-AUC {result['final']['pr_auc']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="random")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--force", action="store_true", help="rerun even if results exist")
    args = parser.parse_args()

    for seed in args.seeds:
        for name, cmd in jobs(args.split, seed):
            out = run_dir(args.split, seed) / f"{name}.json"
            label = f"[{args.split} seed {seed}] {name}"
            if out.exists() and not args.force:
                print(f"{label}: already done ({summary(out)})", flush=True)
                continue
            log = ROOT / "logs" / args.split / f"seed_{seed}" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            with open(log, "w") as f:
                code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            if code != 0 or not out.exists():
                print(f"{label}: FAILED after {time.time() - start:.0f}s, see {log}", flush=True)
                sys.exit(1)
            print(f"{label}: {summary(out)} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
