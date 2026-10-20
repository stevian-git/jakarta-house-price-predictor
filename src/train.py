from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
BASELINE_PATH = PROJECT_ROOT / "results" / "baseline_metrics.json"
TUNING_PATH = PROJECT_ROOT / "results" / "tuning_results.json"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

DATASET_ID = "abiyyurasyiq/jakarta-house-price-dataset"
FEATURES = [
    "city",
    "building_area",
    "bed_rooms",
    "bath_rooms",
    "carport",
]
NUMERIC_FEATURES = ["building_area", "bed_rooms", "bath_rooms", "carport"]
CATEGORICAL_FEATURES = ["city"]
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET_TRANSFORM = "log1p / expm1 via TransformedTargetRegressor"
PREPROCESSING = (
    "numeric: median imputation + Yeo-Johnson power transform; "
    "city: most-frequent imputation + one-hot (handle_unknown=ignore)"
)

RF_PARAM_GRID = {
    "regressor__model__n_estimators": [300, 500],
    "regressor__model__max_depth": [None, 20, 40],
    "regressor__model__min_samples_leaf": [1, 2, 4],
    "regressor__model__max_features": ["sqrt"],
}
GB_PARAM_GRID = {
    "regressor__model__n_estimators": [200, 400],
    "regressor__model__learning_rate": [0.05, 0.1],
    "regressor__model__max_depth": [2, 3],
    "regressor__model__subsample": [0.8, 1.0],
}

PRICE_BAND_BINS = [0, 1e9, 3e9, 5e9, 1e10, 3e10, np.inf]
PRICE_BAND_LABELS = ["<1B", "1-3B", "3-5B", "5-10B", "10-30B", ">30B"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


SCORERS = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(rmse, greater_is_better=False),
    "r2": make_scorer(r2_score),
}


def load_data() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset tidak ditemukan: {CLEAN_PATH}. Jalankan src/data_prep.py dulu."
        )
    df = pd.read_csv(CLEAN_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def wrap_model(regressor) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_models() -> dict:
    return {
        "dummy": wrap_model(DummyRegressor(strategy="median")),
        "ridge": wrap_model(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "random_forest": wrap_model(
            RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": wrap_model(
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
    }


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, X_train, y_train, cv) -> dict:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    train_metrics = metrics_dict(y_train, model.predict(X_train))
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORERS,
        n_jobs=1,
    )
    cv_metrics = {}
    for metric in ("mae", "rmse", "r2"):
        values = -scores[f"test_{metric}"] if metric != "r2" else scores[f"test_{metric}"]
        cv_metrics[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return {
        "train": train_metrics,
        "cv": cv_metrics,
        "fit_seconds": float(fit_seconds),
    }


def save_comparison_figure(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    means = [results[name]["cv"]["mae"]["mean"] for name in names]
    stds = [results[name]["cv"]["mae"]["std"] for name in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=means, y=names, hue=names, palette="viridis", legend=False, ax=ax)
    ax.errorbar(means, range(len(names)), xerr=stds, fmt="none", color="black", capsize=4)
    ax.set_xlabel("CV MAE (IDR)")
    ax.set_ylabel("")
    ax.set_title("Baseline comparison — 5-fold CV MAE on training set")
    for index, value in enumerate(means):
        ax.text(value, index, f"  Rp{value/1e9:.2f}B", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_baseline_cv_mae.png", bbox_inches="tight")
    plt.close(fig)


def library_versions() -> dict:
    return {
        "python": "3.12",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def readable_params(params: dict) -> dict:
    return {key.replace("regressor__model__", ""): value for key, value in params.items()}


def tune_model(
    name: str,
    estimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
) -> tuple[GridSearchCV, dict]:
    if name == "random_forest":
        estimator.set_params(regressor__model__n_jobs=1)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORERS,
        refit="mae",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    seconds = time.perf_counter() - started

    result = search.cv_results_
    best = search.best_index_
    cv_metrics = {
        "mae": {
            "mean": float(-result["mean_test_mae"][best]),
            "std": float(result["std_test_mae"][best]),
        },
        "rmse": {
            "mean": float(-result["mean_test_rmse"][best]),
            "std": float(result["std_test_rmse"][best]),
        },
        "r2": {
            "mean": float(result["mean_test_r2"][best]),
            "std": float(result["std_test_r2"][best]),
        },
    }

    top_results = []
    for index in np.argsort(result["rank_test_mae"])[:5]:
        top_results.append(
            {
                "params": readable_params(result["params"][index]),
                "cv_mae": float(-result["mean_test_mae"][index]),
                "cv_rmse": float(-result["mean_test_rmse"][index]),
                "cv_r2": float(result["mean_test_r2"][index]),
            }
        )

    meta = {
        "param_grid": param_grid,
        "n_candidates": int(len(result["params"])),
        "n_fits": int(len(result["params"]) * CV_FOLDS + 1),
        "best_params": readable_params(search.best_params_),
        "cv_metrics": cv_metrics,
        "top_5": top_results,
        "search_seconds": float(seconds),
    }
    return search, meta


def run_baseline() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Dataset: {len(df):,} rows")
    print(f"Train:   {len(X_train):,} rows")
    print(f"Test:    {len(X_test):,} rows (held out; not evaluated in this stage)")
    print(f"Features: {FEATURES}")
    print(f"Target:   {TARGET} (training transform: log1p)")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        results[name] = evaluate_model(model, X_train, y_train, cv)
        metrics = results[name]
        print(
            f"  CV MAE: Rp{metrics['cv']['mae']['mean']:,.0f} "
            f"+/- Rp{metrics['cv']['mae']['std']:,.0f} | "
            f"CV RMSE: Rp{metrics['cv']['rmse']['mean']:,.0f} | "
            f"CV R2: {metrics['cv']['r2']['mean']:.4f} | "
            f"{metrics['fit_seconds']:.2f}s"
        )

    table = pd.DataFrame(
        {
            "cv_mae": {name: results[name]["cv"]["mae"]["mean"] for name in results},
            "cv_rmse": {name: results[name]["cv"]["rmse"]["mean"] for name in results},
            "cv_r2": {name: results[name]["cv"]["r2"]["mean"] for name in results},
            "train_mae": {name: results[name]["train"]["mae"] for name in results},
            "train_rmse": {name: results[name]["train"]["rmse"] for name in results},
            "train_r2": {name: results[name]["train"]["r2"] for name in results},
        }
    ).sort_values("cv_mae")

    print("\n== BASELINE METRICS (CV on training set) ==")
    print(table.to_string(float_format=lambda value: f"{value:,.4f}"))

    payload = {
        "stage": "baseline",
        "dataset_id": DATASET_ID,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "target_transform": TARGET_TRANSFORM,
        "preprocessing": PREPROCESSING,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv": {"folds": CV_FOLDS, "shuffle": True, "random_state": RANDOM_STATE},
        "metric_contract": "MAE/RMSE/R2 computed on IDR predictions; TTR inverse transform applied once",
        "models": results,
        "best_model_by_cv_mae": table.index[0],
        "library_versions": library_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    save_comparison_figure(results)

    print(f"\nSaved: {BASELINE_PATH}")
    print(f"Figure: {FIGURES_DIR / '10_baseline_cv_mae.png'}")
    print(f"Best baseline by CV MAE: {table.index[0]}")


def run_tune() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Dataset: {len(df):,} rows | train: {len(X_train):,} | test (untouched): {len(X_test):,}")
    print(f"Tuning on training set with {CV_FOLDS}-fold CV; test set is not evaluated.\n")

    models = build_models()
    rf_search, rf_meta = tune_model(
        "random_forest", models["random_forest"], RF_PARAM_GRID, X_train, y_train, cv
    )
    print(
        f"\nRandom Forest best CV MAE: Rp{rf_meta['cv_metrics']['mae']['mean']:,.0f} "
        f"| CV RMSE: Rp{rf_meta['cv_metrics']['rmse']['mean']:,.0f} "
        f"| CV R2: {rf_meta['cv_metrics']['r2']['mean']:.4f} "
        f"| {rf_meta['search_seconds']:.1f}s"
    )
    print(f"Random Forest best params: {rf_meta['best_params']}")

    gb_search, gb_meta = tune_model(
        "gradient_boosting",
        models["gradient_boosting"],
        GB_PARAM_GRID,
        X_train,
        y_train,
        cv,
    )
    print(
        f"\nGradient Boosting best CV MAE: Rp{gb_meta['cv_metrics']['mae']['mean']:,.0f} "
        f"| CV RMSE: Rp{gb_meta['cv_metrics']['rmse']['mean']:,.0f} "
        f"| CV R2: {gb_meta['cv_metrics']['r2']['mean']:.4f} "
        f"| {gb_meta['search_seconds']:.1f}s"
    )
    print(f"Gradient Boosting best params: {gb_meta['best_params']}")

    fitted = {
        "random_forest_tuned": rf_search.best_estimator_,
        "gradient_boosting_tuned": gb_search.best_estimator_,
    }
    candidates = {
        "random_forest_tuned": rf_meta["cv_metrics"],
        "gradient_boosting_tuned": gb_meta["cv_metrics"],
    }
    baseline_payload = read_json(BASELINE_PATH) if BASELINE_PATH.exists() else None
    if baseline_payload:
        baseline_models = build_models()
        for name in ("ridge", "dummy"):
            if name in baseline_payload["models"]:
                candidates[name] = baseline_payload["models"][name]["cv"]
                fitted[name] = baseline_models[name]
                fitted[name].fit(X_train, y_train)

    selected_name = min(
        candidates,
        key=lambda name: (
            candidates[name]["mae"]["mean"],
            candidates[name]["rmse"]["mean"],
            -candidates[name]["r2"]["mean"],
        ),
    )
    selected_model = fitted[selected_name]

    if selected_name == "random_forest_tuned":
        best_params = rf_meta["best_params"]
    elif selected_name == "gradient_boosting_tuned":
        best_params = gb_meta["best_params"]
    else:
        best_params = {}

    train_metrics = metrics_dict(y_train, selected_model.predict(X_train))
    selected = {
        "name": selected_name,
        "best_params": best_params,
        "cv_metrics": candidates[selected_name],
        "train_metrics": train_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)

    reloaded = joblib.load(MODEL_PATH)
    sample_pred = reloaded.predict(X_train.head(10))
    original_pred = selected_model.predict(X_train.head(10))
    if not np.all(np.isfinite(sample_pred)) or (sample_pred < 0).any():
        raise ValueError("Model reload menghasilkan prediksi non-finite atau negatif.")
    if not np.allclose(sample_pred, original_pred):
        raise ValueError("Prediksi model setelah reload tidak sama dengan model asli.")

    table = pd.DataFrame(
        {
            "cv_mae": {name: candidates[name]["mae"]["mean"] for name in candidates},
            "cv_rmse": {name: candidates[name]["rmse"]["mean"] for name in candidates},
            "cv_r2": {name: candidates[name]["r2"]["mean"] for name in candidates},
        }
    ).sort_values("cv_mae")

    print("\n== TUNING CANDIDATES (CV on training set) ==")
    print(table.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSelected model: {selected_name}")
    print(f"Selection rule: lowest CV MAE, tie-break by RMSE then R2")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Reload check: OK ({len(sample_pred)} predictions finite, nonnegative, identical)")

    payload = {
        "stage": "tuning",
        "dataset_id": DATASET_ID,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "target_transform": TARGET_TRANSFORM,
        "preprocessing": PREPROCESSING,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv": {"folds": CV_FOLDS, "shuffle": True, "random_state": RANDOM_STATE},
        "searches": {"random_forest": rf_meta, "gradient_boosting": gb_meta},
        "candidates": candidates,
        "selected": selected,
        "selection_rule": "lowest CV MAE; tie-break RMSE then R2",
        "test_evaluated": False,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "library_versions": library_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNING_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Tuning results: {TUNING_PATH}")


def error_tables(
    X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    frame = pd.DataFrame(
        {
            "city": X_test["city"].to_numpy(),
            "price": y_test.to_numpy(),
            "prediction": predictions,
        }
    )
    frame["abs_error"] = (frame["price"] - frame["prediction"]).abs()
    frame["squared_error"] = (frame["price"] - frame["prediction"]) ** 2

    city_stats = (
        frame.groupby("city")
        .agg(
            n=("price", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            median_abs_error=("abs_error", "median"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )
    city_payload = {
        row["city"]: {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "median_abs_error": float(row["median_abs_error"]),
        }
        for _, row in city_stats.iterrows()
    }

    frame["price_band"] = pd.cut(
        frame["price"],
        bins=PRICE_BAND_BINS,
        labels=PRICE_BAND_LABELS,
        right=False,
    )
    band_stats = (
        frame.groupby("price_band", observed=True)
        .agg(
            n=("price", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            median_abs_error=("abs_error", "median"),
        )
        .reindex(PRICE_BAND_LABELS)
        .dropna(how="all")
        .reset_index()
    )
    band_payload = {
        str(row["price_band"]): {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "median_abs_error": float(row["median_abs_error"]),
        }
        for _, row in band_stats.iterrows()
    }

    return frame, city_payload, band_stats, band_payload


def create_evaluation_figures(
    frame: pd.DataFrame,
    city_stats: pd.DataFrame,
    band_stats: pd.DataFrame,
    importance: dict,
    candidates: dict,
    selected_name: str,
    test_metrics: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="viridis")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_true = frame["price"].to_numpy()
    y_pred = frame["prediction"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35, s=20, color="#2a6f97", ax=ax)
    limits = [
        min(y_true.min(), y_pred.min()) * 0.8,
        max(y_true.max(), y_pred.max()) * 1.2,
    ]
    ax.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual price (IDR, log scale)")
    ax.set_ylabel("Predicted price (IDR, log scale)")
    ax.set_title("Actual vs predicted — test set")
    ax.text(
        0.04,
        0.96,
        f"MAE: Rp{test_metrics['mae']/1e9:.2f}B\nR²: {test_metrics['r2']:.3f}\nn = {len(frame):,}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=18, color="#e76f51")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e9)
    ax.set_xlabel("Predicted price (IDR, log scale)")
    ax.set_ylabel("Residual = actual − predicted (IDR, symlog)")
    ax.set_title("Residuals — test set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_residuals.png", bbox_inches="tight")
    plt.close(fig)

    abs_error_b = frame["abs_error"] / 1e9
    clip_value = float(abs_error_b.quantile(0.995))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(abs_error_b.clip(upper=clip_value), bins=60, color="#2a9d8f", ax=ax)
    ax.axvline(abs_error_b.median(), color="black", linestyle="--", label=f"median = Rp{abs_error_b.median():.2f}B")
    ax.set_xlabel("Absolute error (IDR billions, clipped at 99.5th percentile)")
    ax.set_ylabel("Listings")
    ax.set_title("Distribution of absolute error — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_abs_error_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=city_stats,
        x="city",
        y="mae",
        hue="city",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Test MAE by city")
    ax.set_xlabel("")
    ax.set_ylabel("MAE (IDR)")
    ax.tick_params(axis="x", rotation=20)
    for patch, (_, row) in zip(ax.patches, city_stats.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            row["mae"],
            f"Rp{row['mae']/1e9:.2f}B\nn={int(row['n']):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_error_by_city.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=band_stats,
        x="price_band",
        y="mae",
        hue="price_band",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Test MAE by actual price band")
    ax.set_xlabel("Actual price band")
    ax.set_ylabel("MAE (IDR)")
    for patch, (_, row) in zip(ax.patches, band_stats.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            row["mae"],
            f"Rp{row['mae']/1e9:.2f}B\nn={int(row['n']):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_error_by_price_band.png", bbox_inches="tight")
    plt.close(fig)

    importance_frame = (
        pd.DataFrame(
            {
                "feature": list(importance.keys()),
                "mean": [importance[feature]["mean"] / 1e9 for feature in importance],
                "std": [importance[feature]["std"] / 1e9 for feature in importance],
            }
        )
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=importance_frame,
        x="mean",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.errorbar(
        importance_frame["mean"],
        range(len(importance_frame)),
        xerr=importance_frame["std"],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_xlabel("Increase in MAE when feature is permuted (IDR billions)")
    ax.set_ylabel("")
    ax.set_title("Permutation importance — test set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "16_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "model": list(candidates.keys()),
            "cv_mae": [candidates[name]["mae"]["mean"] for name in candidates],
        }
    ).sort_values("cv_mae")
    colors = ["#e76f51" if name == selected_name else "#2a6f97" for name in comparison["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(comparison["model"], comparison["cv_mae"], color=colors)
    ax.set_xlabel("CV MAE (IDR)")
    ax.set_ylabel("")
    ax.set_title("Model comparison — CV MAE (selected model highlighted)")
    for index, value in enumerate(comparison["cv_mae"]):
        ax.text(value, index, f"  Rp{value/1e9:.2f}B", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "17_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def run_evaluate(force: bool = False) -> None:
    if METRICS_PATH.exists() and not force:
        raise SystemExit(
            f"{METRICS_PATH} sudah ada. Test set sudah dievaluasi. "
            "Gunakan --force hanya jika kamu sengaja mengulang evaluasi final."
        )
    if not MODEL_PATH.exists() or not TUNING_PATH.exists():
        raise SystemExit("Model atau hasil tuning belum ada. Jalankan src/train.py --tune dulu.")

    tuning = read_json(TUNING_PATH)
    model = joblib.load(MODEL_PATH)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    predictions = np.asarray(model.predict(X_test), dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Prediksi test mengandung nilai non-finite.")
    if (predictions < 0).any():
        raise ValueError("Prediksi test mengandung nilai negatif.")

    test_metrics = metrics_dict(y_test, predictions)
    train_metrics = metrics_dict(y_train, model.predict(X_train))
    selected = tuning["selected"]

    frame, city_payload, band_stats, band_payload = error_tables(
        X_test, y_test, predictions
    )
    city_stats = (
        pd.DataFrame(
            [
                {"city": city, **values}
                for city, values in city_payload.items()
            ]
        )
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = {
        feature: {
            "mean": float(perm.importances_mean[index]),
            "std": float(perm.importances_std[index]),
        }
        for index, feature in enumerate(FEATURES)
    }

    create_evaluation_figures(
        frame,
        city_stats,
        band_stats,
        importance,
        tuning["candidates"],
        selected["name"],
        test_metrics,
    )

    payload = {
        "dataset_id": DATASET_ID,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "target_transform": TARGET_TRANSFORM,
        "preprocessing": PREPROCESSING,
        "random_state": RANDOM_STATE,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "cv": tuning["cv"],
        "model": {
            "name": selected["name"],
            "best_params": selected["best_params"],
            "cv_metrics": selected["cv_metrics"],
            "train_metrics": train_metrics,
        },
        "test_metrics": test_metrics,
        "train_vs_test": {
            "mae_gap": float(test_metrics["mae"] - train_metrics["mae"]),
            "rmse_gap": float(test_metrics["rmse"] - train_metrics["rmse"]),
            "r2_gap": float(train_metrics["r2"] - test_metrics["r2"]),
        },
        "error_by_city": city_payload,
        "error_by_price_band": band_payload,
        "feature_importance": importance,
        "feature_importance_note": (
            "Permutation importance on the test set, computed after model selection "
            "for interpretation only; not used to choose the model."
        ),
        "cities": sorted(df["city"].unique().tolist()),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "library_versions": library_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Test set evaluated once after model selection via 5-fold CV on the training set. "
            "Metrics are in IDR and TTR inverse transform is applied once."
        ),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print("\n== FINAL TEST METRICS ==")
    print(f"Model: {selected['name']}")
    print(f"MAE:   Rp{test_metrics['mae']:,.0f}")
    print(f"RMSE:  Rp{test_metrics['rmse']:,.0f}")
    print(f"R2:    {test_metrics['r2']:.4f}")
    print(f"Train MAE: Rp{train_metrics['mae']:,.0f} | gap: Rp{payload['train_vs_test']['mae_gap']:,.0f}")
    print(f"\nSaved: {METRICS_PATH}")
    print(f"Model: {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, tune, and evaluate the Jakarta house price model.")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning and save the selected model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved model once on the test set.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing final evaluation.")
    args = parser.parse_args()

    if args.tune:
        run_tune()
    elif args.evaluate:
        run_evaluate(force=args.force)
    else:
        run_baseline()


if __name__ == "__main__":
    main()
