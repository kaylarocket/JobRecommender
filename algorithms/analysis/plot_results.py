from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def resolve_data_paths() -> Tuple[Path, Path]:
    """
    Look for evaluation and alpha tuning CSVs.
    Prefers a top-level data/ directory, falls back to algorithms/data/.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [repo_root / "data", repo_root / "algorithms" / "data"]
    for candidate in candidates:
        eval_path = candidate / "evaluation_results.csv"
        alpha_path = candidate / "alpha_tuning_results.csv"
        if eval_path.exists() and alpha_path.exists():
            return eval_path, alpha_path
    raise FileNotFoundError("Could not locate evaluation_results.csv and alpha_tuning_results.csv in data directories.")


def select_best_alpha(alpha_df: pd.DataFrame) -> pd.Series:
    sort_cols = ["ndcg_at_k", "recall_at_k", "precision_at_k"]
    return alpha_df.sort_values(sort_cols, ascending=False).iloc[0]


def select_best_baseline(eval_df: pd.DataFrame) -> pd.Series:
    baselines = eval_df[~eval_df["model"].str.startswith("hybrid")]
    if baselines.empty:
        raise ValueError("No baseline models found in evaluation_results.csv.")
    sort_cols = ["ndcg_at_k", "recall_at_k", "precision_at_k"]
    return baselines.sort_values(sort_cols, ascending=False).iloc[0]


def extract_best_hybrid(eval_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.Series:
    if "hybrid_best_alpha" in eval_df["model"].values:
        row = eval_df.loc[eval_df["model"] == "hybrid_best_alpha"].iloc[0].copy()
        row["alpha"] = select_best_alpha(alpha_df)["alpha"]
        return row
    best_alpha = select_best_alpha(alpha_df)
    return pd.Series(
        {
            "model": "hybrid_best_alpha",
            "precision_at_k": best_alpha["precision_at_k"],
            "recall_at_k": best_alpha["recall_at_k"],
            "ndcg_at_k": best_alpha["ndcg_at_k"],
            "alpha": best_alpha["alpha"],
        }
    )


def _available_ks(eval_df: pd.DataFrame) -> List[int]:
    pattern = re.compile(r"^(precision|recall|ndcg|hr|map)_at_(\d+)$")
    ks = set()
    for col in eval_df.columns:
        match = pattern.match(col)
        if match:
            ks.add(int(match.group(2)))
    return sorted(ks)


def _metric_series(row: pd.Series, metric: str, ks: List[int]) -> List[float] | None:
    values = []
    for k in ks:
        col = f"{metric}_at_{k}"
        if col not in row:
            return None
        values.append(float(row[col]))
    return values


def _metric_ylim(metric: str, values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, 1e-6)
    if metric in {"precision"}:
        pad = max(0.02, 0.15 * span)
    elif metric in {"hr", "recall"}:
        pad = max(0.02, 0.10 * span)
    else:
        pad = max(0.02, 0.12 * span)
    y_min = max(0.0, vmin - pad)
    y_max = min(1.0, vmax + pad)
    if y_max - y_min < 0.06:
        mid = (y_min + y_max) / 2
        y_min = max(0.0, mid - 0.03)
        y_max = min(1.0, mid + 0.03)
    return y_min, y_max


def plot_multi_k_metrics(eval_df: pd.DataFrame, best_hybrid: pd.Series, output_path: Path) -> bool:
    ks = _available_ks(eval_df)
    if not ks:
        print("[Warn] No multi-K columns found in evaluation_results.csv (e.g., precision_at_1). Skipping graph5.")
        return False

    try:
        best_baseline = select_best_baseline(eval_df)
    except ValueError as exc:
        print(f"[Warn] {exc}. Skipping graph5.")
        return False
    random_row = eval_df.loc[eval_df["model"] == "random"].iloc[0] if "random" in eval_df["model"].values else None

    model_specs = [
        {
            "label": f"Best baseline ({best_baseline['model']})",
            "row": best_baseline,
            "color": "#1F77B4",
            "lw": 2.0,
            "ls": "-",
            "marker": "o",
            "ms": 5,
            "alpha": 0.95,
        },
        {
            "label": "Hybrid (best α)",
            "row": best_hybrid,
            "color": "#C44E52",
            "lw": 2.6,
            "ls": "-",
            "marker": "s",
            "ms": 6,
            "alpha": 0.95,
        },
    ]
    if random_row is not None:
        model_specs.append(
            {
                "label": "Random (baseline)",
                "row": random_row,
                "color": "#B0B0B0",
                "lw": 1.2,
                "ls": "--",
                "marker": "x",
                "ms": 5,
                "alpha": 0.7,
            }
        )

    metrics = [
        ("hr", "Hit Rate@K"),
        ("ndcg", "NDCG@K"),
        ("map", "MAP@K"),
        ("precision", "Precision@K"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    plotted = 0

    for ax, (metric, title) in zip(axes, metrics):
        y_values: List[float] = []
        for spec in model_specs:
            series = _metric_series(spec["row"], metric, ks)
            if series is None:
                continue
            y_values.extend(series)
            ax.plot(
                ks,
                series,
                marker=spec["marker"],
                markersize=spec["ms"],
                color=spec["color"],
                linewidth=spec["lw"],
                linestyle=spec["ls"],
                alpha=spec["alpha"],
                label=spec["label"],
            )
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_xticks(ks)
        if y_values:
            ax.set_ylim(*_metric_ylim(metric, y_values))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    if plotted == 0:
        plt.close(fig)
        print("[Warn] No valid multi-K metric series found in evaluation_results.csv. Skipping graph5.")
        return False

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Multi-K Ranking Metrics (Implicit Feedback; 1 held-out positive/user; negative sampling)",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path.exists()


def plot_hybrid_focus(eval_df: pd.DataFrame, alpha_df: pd.DataFrame, output_path: Path) -> bool:
    ks = _available_ks(eval_df)
    if not ks:
        print("[Warn] No multi-K columns found in evaluation_results.csv (e.g., precision_at_1). Skipping graph6.")
        return False

    if not eval_df["model"].str.startswith("hybrid").any():
        print("[Warn] No hybrid rows found in evaluation_results.csv. Skipping graph6.")
        return False

    best_alpha_row = extract_best_hybrid(eval_df, alpha_df)
    default_row = (
        eval_df.loc[eval_df["model"] == "hybrid_alpha_0.6"].iloc[0]
        if "hybrid_alpha_0.6" in eval_df["model"].values
        else None
    )

    model_specs = [
        {
            "label": f"Hybrid best α ({best_alpha_row.get('alpha', 'n/a')})",
            "row": best_alpha_row,
            "color": "#C44E52",
            "lw": 2.6,
            "ls": "-",
            "marker": "s",
            "ms": 6,
            "alpha": 0.95,
        },
    ]
    if default_row is not None:
        model_specs.append(
            {
                "label": "Hybrid α=0.6",
                "row": default_row,
                "color": "#4C72B0",
                "lw": 1.8,
                "ls": "--",
                "marker": "o",
                "ms": 5,
                "alpha": 0.85,
            }
        )

    metrics = [
        ("hr", "Hit Rate@K"),
        ("ndcg", "NDCG@K"),
        ("map", "MAP@K"),
        ("precision", "Precision@K"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    plotted = 0

    for ax, (metric, title) in zip(axes, metrics):
        y_values: List[float] = []
        for spec in model_specs:
            series = _metric_series(spec["row"], metric, ks)
            if series is None:
                continue
            y_values.extend(series)
            ax.plot(
                ks,
                series,
                marker=spec["marker"],
                markersize=spec["ms"],
                color=spec["color"],
                linewidth=spec["lw"],
                linestyle=spec["ls"],
                alpha=spec["alpha"],
                label=spec["label"],
            )
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_xticks(ks)
        if y_values:
            ax.set_ylim(*_metric_ylim(metric, y_values))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    if plotted == 0:
        plt.close(fig)
        print("[Warn] No valid hybrid metric series found in evaluation_results.csv. Skipping graph6.")
        return False

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Hybrid Model Performance Across K (Implicit Feedback; 1 held-out positive/user; negative sampling)",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path.exists()


def plot_model_comparison(eval_df: pd.DataFrame, output_path: Path) -> None:
    eval_df = eval_df.copy()
    eval_df["is_hybrid"] = eval_df["model"].str.startswith("hybrid")
    eval_df = eval_df.sort_values(["is_hybrid", "model"])

    models = eval_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, eval_df["precision_at_k"], width, label="Precision@10")
    ax.bar(x, eval_df["recall_at_k"], width, label="Recall@10")
    ax.bar(x + width, eval_df["ndcg_at_k"], width, label="NDCG@10")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison @10 (all evaluated models)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_sweep(alpha_df: pd.DataFrame, best_row: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alpha_df["alpha"], alpha_df["precision_at_k"], marker="o", label="Precision@10")
    ax.plot(alpha_df["alpha"], alpha_df["recall_at_k"], marker="s", label="Recall@10")
    ax.plot(alpha_df["alpha"], alpha_df["ndcg_at_k"], marker="^", label="NDCG@10")

    ax.scatter(best_row["alpha"], best_row["ndcg_at_k"], color="red", zorder=5, label="Best α (NDCG@10)")
    ax.axvline(best_row["alpha"], color="red", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Best α = {best_row['alpha']:.2f}",
        (best_row["alpha"], best_row["ndcg_at_k"]),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color="red",
    )

    ax.set_xlabel("α (Hybrid weight)")
    ax.set_ylabel("Score")
    ax.set_title("Hybrid Alpha Sweep @10")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_vs_baselines(eval_df: pd.DataFrame, best_hybrid: pd.Series, output_path: Path) -> None:
    baselines = eval_df[~eval_df["model"].str.startswith("hybrid")].copy()
    baselines["is_best_baseline"] = False
    best_baseline = select_best_baseline(eval_df)
    baselines.loc[baselines["model"] == best_baseline["model"], "is_best_baseline"] = True

    comparison_df = pd.concat([baselines, best_hybrid.to_frame().T], ignore_index=True)
    comparison_df["is_hybrid"] = comparison_df["model"].str.startswith("hybrid")

    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    colors = ["#4C72B0" if not row.is_hybrid else "#C44E52" for row in comparison_df.itertuples()]
    hatches = ["//" if getattr(row, "is_best_baseline", False) else "" for row in comparison_df.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_prec = ax.bar(x - width, comparison_df["precision_at_k"], width, color=colors, hatch=hatches, label="Precision@10")
    bars_rec = ax.bar(x, comparison_df["recall_at_k"], width, color=colors, hatch=hatches, label="Recall@10")
    bars_ndcg = ax.bar(x + width, comparison_df["ndcg_at_k"], width, color=colors, hatch=hatches, label="NDCG@10")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Best Hybrid vs Baselines (highlighting strongest baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars for quick reading
    for bars in [bars_prec, bars_rec, bars_ndcg]:
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(best_hybrid: pd.Series, best_baseline: pd.Series, output_path: Path) -> None:
    metrics = ["precision_at_k", "recall_at_k", "ndcg_at_k"]
    categories = [m.replace("_at_k", "").upper() for m in metrics]
    deltas = [best_hybrid[m] - best_baseline[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, deltas, color="#55A868")
    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel("Hybrid - Best Baseline")
    ax.set_title("Delta of Best Hybrid vs Strongest Baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, delta in zip(bars, deltas):
        ax.annotate(
            f"{delta:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary_metrics(eval_df: pd.DataFrame, best_hybrid: pd.Series, best_baseline: pd.Series) -> pd.DataFrame:
    def _alpha_for_model(model_name: str) -> float | pd.NA:
        if model_name.startswith("hybrid_alpha_"):
            try:
                return float(model_name.split("_")[-1])
            except ValueError:
                return pd.NA
        if model_name == "hybrid_best_alpha":
            return best_hybrid.get("alpha", pd.NA)
        return pd.NA

    df = eval_df.copy()
    df["alpha"] = df["model"].apply(_alpha_for_model)

    if "alpha" not in best_hybrid:
        best_hybrid = best_hybrid.copy()
        best_hybrid["alpha"] = pd.NA

    if "alpha" not in best_baseline:
        best_baseline = best_baseline.copy()
        best_baseline["alpha"] = pd.NA

    return pd.concat([df, best_hybrid.to_frame().T, best_baseline.to_frame().T], ignore_index=True).drop_duplicates(subset=["model"])


def main() -> None:
    _apply_plot_style()
    eval_path, alpha_path = resolve_data_paths()
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(eval_path)
    alpha_df = pd.read_csv(alpha_path)

    best_alpha_row = select_best_alpha(alpha_df)
    best_hybrid_row = extract_best_hybrid(eval_df, alpha_df)
    best_baseline_row = select_best_baseline(eval_df)

    graph1_path = figures_dir / "graph1_baseline_comparison.png"
    graph2_path = figures_dir / "graph2_alpha_sweep.png"
    graph3_path = figures_dir / "graph3_best_hybrid_vs_baselines.png"
    graph4_path = figures_dir / "graph4_delta_vs_baselines.png"
    graph5_path = figures_dir / "graph5_multi_k_metrics.png"
    graph6_path = figures_dir / "graph6_hybrid_focus.png"
    summary_path = figures_dir / "summary_metrics.csv"

    plot_model_comparison(eval_df, graph1_path)
    plot_alpha_sweep(alpha_df, best_alpha_row, graph2_path)
    plot_best_vs_baselines(eval_df, best_hybrid_row, graph3_path)
    plot_deltas(best_hybrid_row, best_baseline_row, graph4_path)
    graph5_saved = plot_multi_k_metrics(eval_df, best_hybrid_row, graph5_path)
    graph6_saved = plot_hybrid_focus(eval_df, alpha_df, graph6_path)

    summary_df = build_summary_metrics(eval_df, best_hybrid_row, best_baseline_row)
    summary_df.to_csv(summary_path, index=False)

    saved_paths = [graph1_path, graph2_path, graph3_path, graph4_path, summary_path]
    if graph5_saved and graph5_path.exists():
        saved_paths.append(graph5_path)
    if graph6_saved and graph6_path.exists():
        saved_paths.append(graph6_path)

    print("Saved figures and summary:")
    for path in saved_paths:
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
