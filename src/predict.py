from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"

FEATURES = [
    "city",
    "building_area",
    "bed_rooms",
    "bath_rooms",
    "carport",
]

BUILDING_AREA_BOUNDS = (10.0, 10_000.0)
ROOMS_BOUNDS = (1, 20)
CARPORT_BOUNDS = (0, 10)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. Jalankan training terlebih dahulu."
        )
    model = joblib.load(MODEL_PATH)
    trained_features = getattr(model.regressor_, "feature_names_in_", None)
    if trained_features is not None and list(trained_features) != FEATURES:
        raise ValueError(
            "Kontrak fitur model tidak cocok dengan input inference: "
            f"{list(trained_features)} != {FEATURES}."
        )
    return model


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


@lru_cache(maxsize=1)
def get_valid_cities() -> tuple[str, ...]:
    model = load_model()
    try:
        preprocessor = model.regressor_.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cities = tuple(str(city) for city in encoder.categories_[0])
    except (AttributeError, KeyError, IndexError):
        cities = tuple(load_metrics().get("cities", ()))
    if not cities:
        raise ValueError("Daftar kota tidak dapat dibaca dari artefak model.")
    return cities


def _validate_number(name: str, value, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} harus berupa angka.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} harus berupa angka finite.")
    if number < minimum or number > maximum:
        raise ValueError(f"{name} harus antara {minimum:g} dan {maximum:g}.")
    return number


def _validate_integer(name: str, value, minimum: int, maximum: int) -> int:
    number = _validate_number(name, value, minimum, maximum)
    if not number.is_integer():
        raise ValueError(f"{name} harus berupa bilangan bulat.")
    return int(number)


def validate_inputs(
    city: str,
    building_area,
    bed_rooms,
    bath_rooms,
    carport,
) -> dict:
    if not isinstance(city, str) or not city.strip():
        raise ValueError("Kota wajib dipilih.")
    city = city.strip()
    cities = get_valid_cities()
    if city not in cities:
        raise ValueError(
            f"Kota '{city}' tidak didukung model. Pilihan: {', '.join(cities)}."
        )

    return {
        "city": city,
        "building_area": _validate_number(
            "Luas bangunan", building_area, *BUILDING_AREA_BOUNDS
        ),
        "bed_rooms": _validate_integer("Jumlah kamar tidur", bed_rooms, *ROOMS_BOUNDS),
        "bath_rooms": _validate_integer("Jumlah kamar mandi", bath_rooms, *ROOMS_BOUNDS),
        "carport": _validate_integer("Kapasitas carport", carport, *CARPORT_BOUNDS),
    }


def predict_price(
    city: str,
    building_area,
    bed_rooms,
    bath_rooms,
    carport,
) -> float:
    validated = validate_inputs(
        city, building_area, bed_rooms, bath_rooms, carport
    )
    frame = pd.DataFrame([{feature: validated[feature] for feature in FEATURES}])

    model = load_model()
    prediction = float(np.asarray(model.predict(frame)).ravel()[0])

    if not np.isfinite(prediction):
        raise ValueError("Model menghasilkan prediksi non-finite.")
    if prediction < 0:
        raise ValueError("Model menghasilkan prediksi negatif.")
    return prediction


def format_rupiah(amount: float) -> str:
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("Nominal Rupiah harus finite dan nonnegatif.")

    units = (
        (1_000_000_000_000, "Triliun"),
        (1_000_000_000, "Miliar"),
        (1_000_000, "Juta"),
        (1_000, "Ribu"),
    )
    for divisor, label in units:
        if amount >= divisor:
            value = amount / divisor
            text = f"{value:,.2f}".rstrip("0").rstrip(".")
            text = text.replace(",", "X").replace(".", ",").replace("X", ".")
            return f"Rp {text} {label}"

    text = f"{amount:,.0f}".replace(",", ".")
    return f"Rp {text}"


if __name__ == "__main__":
    print("Valid cities:", ", ".join(get_valid_cities()))
    examples = [
        ("Jakarta Selatan", 150, 3, 2, 1),
        ("Jakarta Barat", 200, 4, 3, 2),
        ("Jakarta Pusat", 300, 5, 4, 2),
    ]
    for example in examples:
        price = predict_price(*example)
        print(f"{example} -> {format_rupiah(price)} ({price:,.0f} IDR)")

    print(
        "Formatting:",
        format_rupiah(750_000_000),
        "|",
        format_rupiah(1_500_000_000),
        "|",
        format_rupiah(12_000_000_000),
    )

    try:
        predict_price("Bandung", 150, 3, 2, 1)
    except ValueError as exc:
        print("Validation check:", exc)
