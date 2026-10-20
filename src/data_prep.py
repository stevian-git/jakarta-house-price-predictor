from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "jakarta_house.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "cleaning_report.json"

DATASET_ID = "abiyyurasyiq/jakarta-house-price-dataset"
FEATURES = [
    "city",
    "building_area",
    "bed_rooms",
    "bath_rooms",
    "carport",
]
TARGET = "price"
NUMERIC_COLUMNS = [
    "price",
    "land_area",
    "building_area",
    "bed_rooms",
    "bath_rooms",
    "carport",
]
MIN_PRICE = 100_000_000
MIN_BUILDING_AREA = 10


def _step(report: dict, name: str, before: int, after: int, reason: str) -> None:
    removed = before - after
    report["steps"].append(
        {
            "step": name,
            "rows_before": int(before),
            "rows_after": int(after),
            "rows_removed": int(removed),
            "reason": reason,
        }
    )
    print(f"[{name}] {before:,} -> {after:,} (removed {removed:,})")


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {
        "dataset_id": DATASET_ID,
        "source_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "raw_rows": int(len(raw)),
        "features": FEATURES,
        "target": TARGET,
        "rules": {
            "drop_duplicates": "semua kolom kecuali index",
            "min_price": MIN_PRICE,
            "min_building_area": MIN_BUILDING_AREA,
            "zero_rooms": "bed_rooms/bath_rooms == 0 dianggap missing",
            "carport_zero": "dipertahankan (tidak punya carport)",
            "high_prices": "dipertahankan; target log1p digunakan saat training",
        },
        "steps": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    df = raw.copy()

    for column in ("city", "district"):
        df[column] = df[column].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        before_na = int(df[column].isna().sum())
        df[column] = pd.to_numeric(df[column], errors="coerce")
        coerced = int(df[column].isna().sum()) - before_na
        if coerced:
            print(f"[coerce] {column}: {coerced} nilai tidak numerik -> NaN")

    before = len(df)
    duplicate_subset = [column for column in df.columns if column != "index"]
    df = df.drop_duplicates(subset=duplicate_subset, keep="first")
    _step(report, "drop_duplicate_listings", before, len(df), "duplikat persis selain index")

    before = len(df)
    valid_price = df[TARGET].notna() & (df[TARGET] > 0)
    df = df[valid_price]
    _step(report, "drop_invalid_price", before, len(df), "price missing, nol, atau negatif")

    before = len(df)
    df = df[df[TARGET] >= MIN_PRICE]
    _step(
        report,
        "drop_low_price",
        before,
        len(df),
        f"price < Rp{MIN_PRICE:,.0f} (tidak masuk akal untuk rumah Jakarta)",
    )

    before = len(df)
    valid_area = df["building_area"].notna() & (df["building_area"] >= MIN_BUILDING_AREA)
    df = df[valid_area]
    _step(
        report,
        "drop_invalid_building_area",
        before,
        len(df),
        f"building_area missing atau < {MIN_BUILDING_AREA} m2",
    )

    zero_bed = int((df["bed_rooms"] == 0).sum())
    zero_bath = int((df["bath_rooms"] == 0).sum())
    df["bed_rooms"] = df["bed_rooms"].mask(df["bed_rooms"] == 0)
    df["bath_rooms"] = df["bath_rooms"].mask(df["bath_rooms"] == 0)
    report["zero_rooms_masked"] = {
        "bed_rooms": zero_bed,
        "bath_rooms": zero_bath,
        "note": "diimputasi median di dalam pipeline training",
    }
    print(f"[mask_zero_rooms] bed_rooms={zero_bed}, bath_rooms={zero_bath}")

    df = df.reset_index(drop=True)

    missing = {column: int(df[column].isna().sum()) for column in df.columns}
    report["clean_rows"] = int(len(df))
    report["rows_removed_total"] = int(report["raw_rows"] - len(df))
    report["missing_after_cleaning"] = missing
    report["columns"] = list(df.columns)
    report["price_stats_after_cleaning"] = {
        "min": float(df[TARGET].min()),
        "median": float(df[TARGET].median()),
        "max": float(df[TARGET].max()),
        "mean": float(df[TARGET].mean()),
        "skew": float(df[TARGET].skew()),
    }

    missing_features = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing_features:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_features}")

    return df, report


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset tidak ditemukan: {RAW_PATH}. Jalankan src/fetch_data.py dulu."
        )

    raw = pd.read_csv(RAW_PATH)
    clean, report = clean_data(raw)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n== HASIL CLEANING ==")
    print(f"Rows: {report['raw_rows']:,} -> {report['clean_rows']:,}")
    print(f"Columns: {report['columns']}")
    print(f"Missing setelah cleaning: {report['missing_after_cleaning']}")
    print(f"Price after cleaning:\n{clean[TARGET].describe().to_string()}")
    print(f"\nSaved: {PROCESSED_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
