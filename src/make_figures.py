from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
BASELINE_PATH = PROJECT_ROOT / "results" / "baseline_metrics.json"
TUNING_PATH = PROJECT_ROOT / "results" / "tuning_results.json"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

PRICE_BAND_ORDER = ["<1B", "1-3B", "3-5B", "5-10B", "10-30B", ">30B"]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the corresponding training stage first.")
    return json.loads(path.read_text())


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_baseline_comparison() -> Path:
    baseline = read_json(BASELINE_PATH)
    models = baseline["models"]
    names = sorted(models, key=lambda name: models[name]["cv"]["mae"]["mean"])
    means = np.array([models[name]["cv"]["mae"]["mean"] for name in names])
    stds = np.array([models[name]["cv"]["mae"]["std"] for name in names])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=means, y=names, hue=names, palette="viridis", legend=False, ax=ax)
    ax.errorbar(means, range(len(names)), xerr=stds, fmt="none", color="black", capsize=4)
    ax.set_xlabel("CV MAE (IDR)")
    ax.set_ylabel("")
    ax.set_title("Baseline comparison - 5-fold CV MAE on training set")
    for index, value in enumerate(means):
        ax.text(value, index, f"  Rp{value / 1e9:.2f}B", va="center", fontsize=9)
    return save_figure(fig, "10_baseline_cv_mae.png")


def plot_error_by_city() -> Path:
    metrics = read_json(METRICS_PATH)
    frame = pd.DataFrame(metrics["error_by_city"]).T.reset_index(names="city")
    frame = frame.astype({"n": int, "mae": float, "rmse": float, "median_abs_error": float})
    frame = frame.sort_values("mae", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=frame,
        x="mae",
        y="city",
        hue="city",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Test MAE by city")
    ax.set_xlabel("MAE (IDR)")
    ax.set_ylabel("")
    for patch, (_, row) in zip(ax.patches, frame.iterrows()):
        ax.text(
            row["mae"],
            patch.get_y() + patch.get_height() / 2,
            f"  Rp{row['mae'] / 1e9:.2f}B (n={int(row['n']):,})",
            va="center",
            fontsize=8,
        )
    return save_figure(fig, "14_error_by_city.png")


def plot_error_by_price_band() -> Path:
    metrics = read_json(METRICS_PATH)
    frame = pd.DataFrame(metrics["error_by_price_band"]).T.reset_index(names="price_band")
    frame = frame.astype({"n": int, "mae": float, "rmse": float, "median_abs_error": float})
    frame["price_band"] = pd.Categorical(frame["price_band"], categories=PRICE_BAND_ORDER, ordered=True)
    frame = frame.sort_values("price_band")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=frame,
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
    for patch, (_, row) in zip(ax.patches, frame.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            row["mae"],
            f"Rp{row['mae'] / 1e9:.2f}B\nn={int(row['n']):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return save_figure(fig, "15_error_by_price_band.png")


def plot_feature_importance() -> Path:
    metrics = read_json(METRICS_PATH)
    importance = metrics["feature_importance"]
    frame = pd.DataFrame(
        {
            "feature": list(importance),
            "mean": [importance[name]["mean"] / 1e9 for name in importance],
            "std": [importance[name]["std"] / 1e9 for name in importance],
        }
    ).sort_values("mean")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=frame,
        x="mean",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.errorbar(
        frame["mean"],
        range(len(frame)),
        xerr=frame["std"],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_xlabel("Increase in MAE when feature is permuted (IDR billions)")
    ax.set_ylabel("")
    ax.set_title("Permutation importance - test set")
    return save_figure(fig, "16_feature_importance.png")


def plot_model_comparison() -> Path:
    tuning = read_json(TUNING_PATH)
    candidates = tuning["candidates"]
    selected_name = tuning["selected"]["name"]
    frame = pd.DataFrame(
        {
            "model": list(candidates),
            "cv_mae": [candidates[name]["mae"]["mean"] for name in candidates],
        }
    ).sort_values("cv_mae")
    colors = ["#e76f51" if name == selected_name else "#2a6f97" for name in frame["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(frame["model"], frame["cv_mae"], color=colors)
    ax.set_xlabel("CV MAE (IDR)")
    ax.set_ylabel("")
    ax.set_title("Model comparison - CV MAE (selected model highlighted)")
    for index, value in enumerate(frame["cv_mae"]):
        ax.text(value, index, f"  Rp{value / 1e9:.2f}B", va="center", fontsize=9)
    return save_figure(fig, "17_model_comparison.png")


def plot_train_cv_test() -> Path:
    metrics = read_json(METRICS_PATH)
    train = metrics["model"]["train_metrics"]
    cv = metrics["model"]["cv_metrics"]
    test = metrics["test_metrics"]

    labels = ["Train", "CV", "Test"]
    mae_b = np.array([train["mae"], cv["mae"]["mean"], test["mae"]]) / 1e9
    r2 = np.array([train["r2"], cv["r2"]["mean"], test["r2"]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(x=labels, y=mae_b, hue=labels, palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title("MAE by evaluation split")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MAE (IDR billions)")
    for index, value in enumerate(mae_b):
        axes[0].text(index, value, f"Rp{value:.2f}B", ha="center", va="bottom", fontsize=9)

    sns.barplot(x=labels, y=r2, hue=labels, palette="viridis", legend=False, ax=axes[1])
    axes[1].set_title("R2 by evaluation split")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R2")
    axes[1].set_ylim(0, 1)
    for index, value in enumerate(r2):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Selected model - train vs cross-validation vs test", y=1.02)
    return save_figure(fig, "18_train_cv_test_metrics.png")


def plot_data_overview() -> Path:
    if not CLEAN_PATH.exists():
        raise SystemExit(f"Missing {CLEAN_PATH}. Run src/data_prep.py first.")
    df = pd.read_csv(CLEAN_PATH)
    price_b = df["price"] / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(price_b, bins=60, color="#2a6f97", alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Price (IDR billions, log scale)")
    axes[0].set_ylabel("Listings (log scale)")
    axes[0].set_title(f"Cleaned price distribution (n={len(df):,})")
    axes[0].axvline(price_b.median(), color="black", linestyle="--", linewidth=1)
    axes[0].text(
        price_b.median(),
        axes[0].get_ylim()[1] * 0.5,
        f" median Rp{price_b.median():.2f}B",
        fontsize=9,
    )

    axes[1].scatter(
        df["building_area"],
        price_b,
        alpha=0.25,
        s=12,
        color="#e76f51",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Building area (m2, log scale)")
    axes[1].set_ylabel("Price (IDR billions, log scale)")
    axes[1].set_title("Building area vs price")
    return save_figure(fig, "19_data_overview.png")


def main() -> None:
    sns.set_theme(style="whitegrid", palette="viridis")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    saved = [
        plot_baseline_comparison(),
        plot_error_by_city(),
        plot_error_by_price_band(),
        plot_feature_importance(),
        plot_model_comparison(),
        plot_train_cv_test(),
        plot_data_overview(),
    ]

    print("Saved figures:")
    for path in saved:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print(
        "\nPrediction-level figures 11-13 are produced by the single final "
        "evaluation in src/train.py --evaluate; this script does not touch the test set."
    )


if __name__ == "__main__":
    main()
