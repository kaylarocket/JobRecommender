from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
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
    """
    Select the best alpha.

    Primary: max NDCG@10 (ndcg_at_k)
    Tie-breaker: max Recall@10 (recall_at_k), then Precision@10 (precision_at_k)

    If ndcg_at_k is missing (older CSV), falls back to max recall, then precision.
    """
    sort_cols = ["recall_at_k", "precision_at_k"]
    if "ndcg_at_k" in alpha_df.columns:
        sort_cols = ["ndcg_at_k"] + sort_cols
    sorted_df = alpha_df.sort_values(sort_cols, ascending=False)
    return sorted_df.iloc[0]


def plot_baseline_comparison(eval_df: pd.DataFrame, output_path: Path) -> None:
    models = ["tfidf", "lightfm", "hybrid_alpha_0.6"]
    subset = eval_df.set_index("model").loc[models]
    x = range(len(models))
    has_ndcg = "ndcg_at_k" in subset.columns
    width = 0.25 if has_ndcg else 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    if has_ndcg:
        ax.bar([i - width for i in x], subset["precision_at_k"], width, label="Precision@10")
        ax.bar([i for i in x], subset["recall_at_k"], width, label="Recall@10")
        ax.bar([i + width for i in x], subset["ndcg_at_k"], width, label="NDCG@10")
        title = "Baseline Model Performance @10 (Precision, Recall, NDCG)"
    else:
        ax.bar([i - width / 2 for i in x], subset["precision_at_k"], width, label="Precision@10")
        ax.bar([i + width / 2 for i in x], subset["recall_at_k"], width, label="Recall@10")
        title = "Baseline Model Performance @10 (Precision, Recall)"

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_sweep(alpha_df: pd.DataFrame, best_row: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alpha_df["alpha"], alpha_df["precision_at_k"], marker="o", label="Precision@10")
    ax.plot(alpha_df["alpha"], alpha_df["recall_at_k"], marker="s", label="Recall@10")
    has_ndcg = "ndcg_at_k" in alpha_df.columns
    if has_ndcg:
        ax.plot(alpha_df["alpha"], alpha_df["ndcg_at_k"], marker="^", label="NDCG@10")

    best_y = best_row["ndcg_at_k"] if has_ndcg else best_row["recall_at_k"]
    best_label = "Best α (NDCG@10)" if has_ndcg else "Best α (Recall@10)"
    ax.scatter(best_row["alpha"], best_y, color="red", zorder=5, label=best_label)
    ax.annotate(
        f"Best α = {best_row['alpha']:.2f}",
        (best_row["alpha"], best_y),
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
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_vs_baselines(comparison_df: pd.DataFrame, output_path: Path) -> None:
    models = comparison_df["model"].tolist()
    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], comparison_df["precision_at_k"], width, label="Precision@10")
    ax.bar([i + width / 2 for i in x], comparison_df["recall_at_k"], width, label="Recall@10")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Best Hybrid vs Baselines (Precision/Recall @10)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(best_row: pd.Series, tfidf_row: pd.Series, lightfm_row: pd.Series, output_path: Path) -> None:
    categories = [
        "ΔPrecision@10 vs TF-IDF",
        "ΔRecall@10 vs TF-IDF",
        "ΔPrecision@10 vs LightFM",
        "ΔRecall@10 vs LightFM",
    ]
    deltas = [
        best_row["precision_at_k"] - tfidf_row["precision_at_k"],
        best_row["recall_at_k"] - tfidf_row["recall_at_k"],
        best_row["precision_at_k"] - lightfm_row["precision_at_k"],
        best_row["recall_at_k"] - lightfm_row["recall_at_k"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, deltas, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel("Delta")
    ax.set_title("Hybrid (Best α) Improvement Over Baselines")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Label bars with delta values for quick reading.
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


def build_summary_metrics(eval_df: pd.DataFrame, best_row: pd.Series) -> pd.DataFrame:
    eval_indexed = eval_df.set_index("model")
    eval_has_ndcg = "ndcg_at_k" in eval_indexed.columns
    best_has_ndcg = "ndcg_at_k" in best_row.index
    rows = [
        {
            "model": "tfidf",
            "alpha": pd.NA,
            "precision_at_10": eval_indexed.loc["tfidf", "precision_at_k"],
            "recall_at_10": eval_indexed.loc["tfidf", "recall_at_k"],
            "ndcg_at_10": eval_indexed.loc["tfidf", "ndcg_at_k"] if eval_has_ndcg else pd.NA,
        },
        {
            "model": "lightfm",
            "alpha": pd.NA,
            "precision_at_10": eval_indexed.loc["lightfm", "precision_at_k"],
            "recall_at_10": eval_indexed.loc["lightfm", "recall_at_k"],
            "ndcg_at_10": eval_indexed.loc["lightfm", "ndcg_at_k"] if eval_has_ndcg else pd.NA,
        },
        {
            "model": "hybrid_alpha_0.6",
            "alpha": 0.6,
            "precision_at_10": eval_indexed.loc["hybrid_alpha_0.6", "precision_at_k"],
            "recall_at_10": eval_indexed.loc["hybrid_alpha_0.6", "recall_at_k"],
            "ndcg_at_10": eval_indexed.loc["hybrid_alpha_0.6", "ndcg_at_k"] if eval_has_ndcg else pd.NA,
        },
        {
            "model": "hybrid_best_alpha",
            "alpha": best_row["alpha"],
            "precision_at_10": best_row["precision_at_k"],
            "recall_at_10": best_row["recall_at_k"],
            "ndcg_at_10": best_row["ndcg_at_k"] if best_has_ndcg else pd.NA,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    eval_path, alpha_path = resolve_data_paths()
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluation results from: {eval_path}")
    print(f"Loading alpha tuning results from: {alpha_path}")

    eval_df = pd.read_csv(eval_path)
    alpha_df = pd.read_csv(alpha_path)

    best_row = select_best_alpha(alpha_df)
    has_ndcg = "ndcg_at_k" in alpha_df.columns
    if has_ndcg:
        print("Best alpha selection: max NDCG@10, tie-breaker Recall@10, then Precision@10.")
        print(
            f"Best alpha: {best_row['alpha']:.2f} "
            f"(NDCG@10={best_row['ndcg_at_k']:.4f}, Precision@10={best_row['precision_at_k']:.4f}, Recall@10={best_row['recall_at_k']:.4f})"
        )
    else:
        print("Best alpha selection (fallback): max Recall@10, tie-breaker Precision@10. (ndcg_at_k missing in CSV)")
        print(
            f"Best alpha: {best_row['alpha']:.2f} "
            f"(Precision@10={best_row['precision_at_k']:.4f}, Recall@10={best_row['recall_at_k']:.4f})"
        )

    graph1_path = figures_dir / "graph1_baseline_comparison.png"
    graph2_path = figures_dir / "graph2_alpha_sweep.png"
    graph3_path = figures_dir / "graph3_best_hybrid_vs_baselines.png"
    graph4_path = figures_dir / "graph4_delta_vs_baselines.png"
    summary_path = figures_dir / "summary_metrics.csv"

    plot_baseline_comparison(eval_df, graph1_path)
    plot_alpha_sweep(alpha_df, best_row, graph2_path)

    eval_indexed = eval_df.set_index("model")
    comparison_df = pd.DataFrame(
        {
            "model": ["tfidf", "lightfm", "hybrid_best_alpha"],
            "precision_at_k": [
                eval_indexed.loc["tfidf", "precision_at_k"],
                eval_indexed.loc["lightfm", "precision_at_k"],
                best_row["precision_at_k"],
            ],
            "recall_at_k": [
                eval_indexed.loc["tfidf", "recall_at_k"],
                eval_indexed.loc["lightfm", "recall_at_k"],
                best_row["recall_at_k"],
            ],
        }
    )
    plot_best_vs_baselines(comparison_df, graph3_path)
    plot_deltas(best_row, eval_indexed.loc["tfidf"], eval_indexed.loc["lightfm"], graph4_path)

    summary_df = build_summary_metrics(eval_df, best_row)
    summary_df.to_csv(summary_path, index=False)

    print("Saved figures and summary:")
    print(f"- {graph1_path}")
    print(f"- {graph2_path}")
    print(f"- {graph3_path}")
    print(f"- {graph4_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
