from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.predict import (
    BUILDING_AREA_BOUNDS,
    CARPORT_BOUNDS,
    FEATURES,
    ROOMS_BOUNDS,
    format_rupiah,
    get_valid_cities,
    load_metrics,
    predict_price,
)

st.set_page_config(
    page_title="Jakarta House Price Predictor",
    page_icon=None,
    layout="centered",
)

st.title("Jakarta House Price Predictor")
st.write(
    "Estimasi harga rumah berdasarkan pola listing historis di DKI Jakarta. "
    "Model menggunakan lima fitur yang sama dengan data training."
)

try:
    cities = get_valid_cities()
except Exception as exc:
    st.error(f"Model belum siap: {exc}")
    st.stop()

with st.form("prediction_form"):
    city = st.selectbox("City", cities)

    building_area = st.number_input(
        "Building area (m²)",
        min_value=BUILDING_AREA_BOUNDS[0],
        max_value=BUILDING_AREA_BOUNDS[1],
        value=150.0,
        step=10.0,
    )

    left, right = st.columns(2)
    bed_rooms = left.number_input(
        "Bedrooms",
        min_value=ROOMS_BOUNDS[0],
        max_value=ROOMS_BOUNDS[1],
        value=3,
        step=1,
    )
    bath_rooms = right.number_input(
        "Bathrooms",
        min_value=ROOMS_BOUNDS[0],
        max_value=ROOMS_BOUNDS[1],
        value=2,
        step=1,
    )

    carport = st.number_input(
        "Carport capacity",
        min_value=CARPORT_BOUNDS[0],
        max_value=CARPORT_BOUNDS[1],
        value=1,
        step=1,
    )

    submitted = st.form_submit_button("Estimate price", use_container_width=True)

if submitted:
    try:
        price = predict_price(city, building_area, bed_rooms, bath_rooms, carport)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.metric("Estimated price", format_rupiah(price))
        st.caption(
            "Estimasi model berbasis listing historis. Bukan harga transaksi, "
            "penilaian resmi, atau jaminan harga pasar terkini."
        )

st.divider()
st.caption(
    "Cakupan wilayah: DKI Jakarta (Jakarta Barat, Jakarta Pusat, Jakarta Selatan, "
    "Jakarta Timur, Jakarta Utara). Model tidak mencakup Jabodetabek di luar DKI Jakarta."
)

metrics = load_metrics()
if metrics:
    with st.expander("Model information"):
        model_info = metrics.get("model", {})
        test_metrics = metrics.get("test_metrics", {})
        st.write(f"Model: {model_info.get('name', 'unknown')}")
        if test_metrics:
            st.write(f"Test MAE: {format_rupiah(test_metrics['mae'])}")
            st.write(f"Test RMSE: {format_rupiah(test_metrics['rmse'])}")
            st.write(f"Test R²: {test_metrics['r2']:.3f}")
        n_samples = metrics.get("n_samples")
        if isinstance(n_samples, int):
            st.write(f"Training samples: {n_samples:,}")
        else:
            st.write("Training samples: unknown")
        st.write(f"Features: {', '.join(FEATURES)}")
        st.write(
            "Model dilatih pada listing historis; error lebih besar pada segmen "
            "rumah harga tinggi."
        )
