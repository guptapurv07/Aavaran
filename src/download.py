"""Stage 1: download the PaySim dataset from Kaggle into data/raw/paysim.csv."""

import shutil
from pathlib import Path

import kagglehub

from config import RAW_CSV


def main() -> None:
    if RAW_CSV.exists():
        print(f"{RAW_CSV} already exists")
        return
    folder = Path(kagglehub.dataset_download("ealaxi/paysim1"))
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(next(folder.glob("*.csv")), RAW_CSV)
    print(f"Saved {RAW_CSV} ({RAW_CSV.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
