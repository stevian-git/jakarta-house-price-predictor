from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import kagglehub

DATASET_ID = "abiyyurasyiq/jakarta-house-price-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "jakarta_house.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> None:
    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size:,} bytes")
    print(f"SHA-256: {sha256(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Jakarta house price dataset from Kaggle."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite the local raw dataset.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not args.force:
        print("Raw dataset already exists, skipping download.")
        describe(RAW_PATH)
        return

    print(f"Downloading Kaggle dataset: {DATASET_ID}")
    cache_dir = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"kagglehub cache: {cache_dir}")

    matches = sorted(cache_dir.rglob("jakarta_house.csv"))
    if not matches:
        raise FileNotFoundError(
            f"jakarta_house.csv not found under kagglehub cache: {cache_dir}"
        )

    source = matches[0]
    if source.resolve() != RAW_PATH.resolve():
        shutil.copy2(source, RAW_PATH)

    print("Raw dataset saved.")
    describe(RAW_PATH)


if __name__ == "__main__":
    main()
