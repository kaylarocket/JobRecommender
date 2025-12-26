from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    summary_path = figures_dir / "summary_metrics.csv"

    plot_model_comparison(eval_df, graph1_path)
    plot_alpha_sweep(alpha_df, best_alpha_row, graph2_path)
    plot_best_vs_baselines(eval_df, best_hybrid_row, graph3_path)
    plot_deltas(best_hybrid_row, best_baseline_row, graph4_path)

    summary_df = build_summary_metrics(eval_df, best_hybrid_row, best_baseline_row)
    summary_df.to_csv(summary_path, index=False)

    print("Saved figures and summary:")
    print(f"- {graph1_path}")
    print(f"- {graph2_path}")
    print(f"- {graph3_path}")
    print(f"- {graph4_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
