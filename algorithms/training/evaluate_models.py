"""
Evaluation pipeline for comparing TF-IDF, SBERT, LightFM, NCF, and Hybrid recommenders.

README:
- Run evaluations: python3 evaluate_models.py
- Plot thesis figures: python3 algorithms/analysis/plot_results.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from algorithms.core.data_loading import (
    MAX_JOBS,
    MAX_USERS,
    build_job_table,
    build_synthetic_interactions,
    build_user_table,
    load_applicants_dataset,
    load_jobstreet_job_dataset,
)
from algorithms.core.metrics import mean, ndcg_at_k, precision_at_k, recall_at_k
from algorithms.core.registry import register_model
from algorithms.models.hybrid_model import compute_hybrid_scores
from algorithms.models.lightfm_model import (
    build_lightfm_dataset,
    predict_lightfm_scores_for_user,
    train_lightfm,
)
from algorithms.models.ncf_model import (
    build_ncf_training_data,
    predict_ncf_scores_for_user,
    train_ncf_model,
)
from algorithms.models.sbert_model import (
    build_sbert_representations,
    compute_sbert_scores_for_user,
)
from algorithms.models.tfidf_model import (
    build_tfidf_representations,
    compute_content_scores_for_user,
)

TEST_SIZE = 0.2
TOP_K = 10
DEFAULT_ALPHA = 0.6
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_test_split_interactions(
    interactions: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split per user for implicit interactions.
    Users with <2 interactions are kept entirely in train.
    """
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []

    for _, group in interactions.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        n_test = max(1, int(len(group) * test_size))
        test_sample = group.sample(n=n_test, random_state=rng.integers(0, 1_000_000))
        train_sample = group.drop(test_sample.index)
        train_rows.append(train_sample)
        test_rows.append(test_sample)

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=interactions.columns)
    return train_df, test_df


def _rank_jobs(
    scores: np.ndarray,
    jobs: pd.DataFrame,
    job_index: Dict[str, int],
    exclude_ids: Set[str],
    top_k: int,
) -> List[str]:
    ranked_scores = scores.copy()
    for jid in exclude_ids:
        idx = job_index.get(jid)
        if idx is not None:
            ranked_scores[idx] = -np.inf
    order = np.argsort(ranked_scores)[::-1]
    ranked_ids = [jobs.loc[i, "job_id"] for i in order if ranked_scores[i] != -np.inf]
    return ranked_ids[:top_k]


def _empty_metric_store(names: Iterable[str], alphas: Iterable[float]):
    metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"precision": [], "recall": [], "ndcg": []} for name in names
    }
    alpha_metrics: Dict[float, Dict[str, List[float]]] = {
        alpha: {"precision": [], "recall": [], "ndcg": []} for alpha in alphas
    }
    return metrics, alpha_metrics


def _update_metric(
    store: Dict[str, List[float]],
    predicted: List[str],
    actual: Set[str],
    k: int,
) -> None:
    store["precision"].append(precision_at_k(predicted, actual, k))
    store["recall"].append(recall_at_k(predicted, actual, k))
    store["ndcg"].append(ndcg_at_k(predicted, actual, k))


def _build_score_registry(
    job_tfidf,
    user_tfidf,
    job_sbert: np.ndarray,
    user_sbert: np.ndarray,
    lfm_model,
    lfm_dataset,
    lfm_jobs: pd.DataFrame,
    user_features_matrix,
    item_features_matrix,
    ncf_model,
    ncf_user_index: Dict[str, int],
    ncf_job_index: Dict[str, int],
) -> Dict[str, callable]:
    """
    Register scoring functions for each model to keep evaluation declarative.
    """
    scorers = {
        "tfidf": lambda user_id, user_idx: compute_content_scores_for_user(user_idx, job_tfidf, user_tfidf),
        "sbert": lambda user_id, user_idx: compute_sbert_scores_for_user(user_idx, job_sbert, user_sbert),
        "lightfm": lambda user_id, user_idx: predict_lightfm_scores_for_user(
            user_id=user_id,
            model=lfm_model,
            dataset=lfm_dataset,
            jobs=lfm_jobs,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
        ),
        "ncf": lambda user_id, user_idx: predict_ncf_scores_for_user(
            user_id=user_id,
            model=ncf_model,
            job_index=ncf_job_index,
            user_index=ncf_user_index,
            n_items=len(lfm_jobs),
        ),
    }
    for name, fn in scorers.items():
        register_model(name, fn)
    return scorers


def evaluate_all_models(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    job_sbert: np.ndarray,
    user_sbert: np.ndarray,
    lfm_model,
    lfm_dataset,
    user_features_matrix,
    item_features_matrix,
    ncf_model,
    ncf_user_index: Dict[str, int],
    ncf_job_index: Dict[str, int],
    alphas: List[float],
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    job_index = {jid: idx for idx, jid in enumerate(jobs["job_id"])}
    user_index = {uid: idx for idx, uid in enumerate(users["user_id"])}

    metrics, alpha_metrics = _empty_metric_store(
        names=["tfidf", "sbert", "lightfm", "ncf", f"hybrid_alpha_{DEFAULT_ALPHA}"],
        alphas=alphas,
    )

    scorers = _build_score_registry(
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        job_sbert=job_sbert,
        user_sbert=user_sbert,
        lfm_model=lfm_model,
        lfm_dataset=lfm_dataset,
        lfm_jobs=jobs,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        ncf_model=ncf_model,
        ncf_user_index=ncf_user_index,
        ncf_job_index=ncf_job_index,
    )

    for user_id, group in test_interactions.groupby("user_id"):
        ground_truth = set(group["job_id"])
        if len(ground_truth) == 0 or user_id not in user_index:
            continue

        train_seen = set(train_interactions.loc[train_interactions["user_id"] == user_id, "job_id"])
        uidx = user_index[user_id]

        tfidf_scores = scorers["tfidf"](user_id, uidx)
        sbert_scores = scorers["sbert"](user_id, uidx)
        lfm_scores = scorers["lightfm"](user_id, uidx)
        ncf_scores = scorers["ncf"](user_id, uidx)

        hybrid_scores_default, content_norm, lfm_norm = compute_hybrid_scores(
            content_scores=tfidf_scores,
            lfm_scores=lfm_scores,
            alpha=DEFAULT_ALPHA,
        )

        tfidf_ranked = _rank_jobs(content_norm, jobs, job_index, train_seen, top_k)
        sbert_ranked = _rank_jobs(sbert_scores, jobs, job_index, train_seen, top_k)
        lfm_ranked = _rank_jobs(lfm_norm, jobs, job_index, train_seen, top_k)
        ncf_ranked = _rank_jobs(ncf_scores, jobs, job_index, train_seen, top_k)
        hybrid_ranked = _rank_jobs(hybrid_scores_default, jobs, job_index, train_seen, top_k)

        _update_metric(metrics["tfidf"], tfidf_ranked, ground_truth, top_k)
        _update_metric(metrics["sbert"], sbert_ranked, ground_truth, top_k)
        _update_metric(metrics["lightfm"], lfm_ranked, ground_truth, top_k)
        _update_metric(metrics["ncf"], ncf_ranked, ground_truth, top_k)
        _update_metric(metrics[f"hybrid_alpha_{DEFAULT_ALPHA}"], hybrid_ranked, ground_truth, top_k)

        for alpha in alphas:
            hybrid_scores = alpha * content_norm + (1 - alpha) * lfm_norm
            hybrid_ranked_alpha = _rank_jobs(hybrid_scores, jobs, job_index, train_seen, top_k)
            _update_metric(alpha_metrics[alpha], hybrid_ranked_alpha, ground_truth, top_k)

    eval_rows = []
    for name, store in metrics.items():
        eval_rows.append(
            {
                "model": name,
                "precision_at_k": mean(store["precision"]),
                "recall_at_k": mean(store["recall"]),
                "ndcg_at_k": mean(store["ndcg"]),
            }
        )

    alpha_rows = []
    for alpha, store in alpha_metrics.items():
        alpha_rows.append(
            {
                "alpha": alpha,
                "precision_at_k": mean(store["precision"]),
                "recall_at_k": mean(store["recall"]),
                "ndcg_at_k": mean(store["ndcg"]),
            }
        )

    alpha_df = pd.DataFrame(alpha_rows).sort_values("alpha").reset_index(drop=True)
    best_alpha_row = select_best_alpha(alpha_df)
    eval_rows.append(
        {
            "model": "hybrid_best_alpha",
            "precision_at_k": best_alpha_row["precision_at_k"],
            "recall_at_k": best_alpha_row["recall_at_k"],
            "ndcg_at_k": best_alpha_row["ndcg_at_k"],
        }
    )

    eval_df = pd.DataFrame(eval_rows)
    return eval_df, alpha_df


def select_best_alpha(alpha_df: pd.DataFrame) -> pd.Series:
    sort_cols = ["ndcg_at_k", "recall_at_k", "precision_at_k"]
    return alpha_df.sort_values(sort_cols, ascending=False).iloc[0]


def main() -> None:
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_jobs = load_jobstreet_job_dataset()
    raw_applicants = load_applicants_dataset()

    if len(raw_jobs) > MAX_JOBS:
        raw_jobs = raw_jobs.sample(MAX_JOBS, random_state=42)
    if len(raw_applicants) > MAX_USERS:
        raw_applicants = raw_applicants.sample(MAX_USERS, random_state=42)

    jobs = build_job_table(raw_jobs).reset_index(drop=True)
    users = build_user_table(raw_applicants).reset_index(drop=True)
    print(f"Prepared {len(jobs)} jobs and {len(users)} users.")

    interactions_df = build_synthetic_interactions(users, jobs)
    train_interactions, test_interactions = train_test_split_interactions(interactions_df, test_size=TEST_SIZE)
    print(f"Train interactions: {len(train_interactions)}, Test interactions: {len(test_interactions)}")

    # Content-based models
    _, job_tfidf, user_tfidf = build_tfidf_representations(users, jobs)
    _, job_sbert, user_sbert = build_sbert_representations(
        users=users["user_text"].values,
        jobs=jobs["job_text"].values,
        seed=42,
    )

    # LightFM
    dataset, interactions, weights, user_features_matrix, item_features_matrix = build_lightfm_dataset(
        interactions_df=train_interactions,
        users=users,
        jobs=jobs,
    )
    lfm_model = train_lightfm(
        interactions=interactions,
        weights=weights,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )

    # NCF
    (
        ncf_user_indices,
        ncf_item_indices,
        ncf_labels,
        ncf_user_index,
        ncf_job_index,
    ) = build_ncf_training_data(train_interactions, users, jobs)
    if len(ncf_labels) == 0:
        raise ValueError("NCF training data is empty. Check interactions preprocessing.")
    ncf_model = train_ncf_model(
        user_indices=ncf_user_indices,
        item_indices=ncf_item_indices,
        labels=ncf_labels,
        n_users=len(users),
        n_items=len(jobs),
        epochs=4,
        embedding_dim=32,
    )

    alpha_candidates = sorted(set(ALPHAS + [DEFAULT_ALPHA]))
    eval_df, alpha_df = evaluate_all_models(
        users=users,
        jobs=jobs,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        job_sbert=job_sbert,
        user_sbert=user_sbert,
        lfm_model=lfm_model,
        lfm_dataset=dataset,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        ncf_model=ncf_model,
        ncf_user_index=ncf_user_index,
        ncf_job_index=ncf_job_index,
        alphas=alpha_candidates,
        top_k=TOP_K,
    )

    eval_path = OUT_DIR / "evaluation_results.csv"
    alpha_path = OUT_DIR / "alpha_tuning_results.csv"
    eval_df.to_csv(eval_path, index=False)
    alpha_df.to_csv(alpha_path, index=False)

    best_alpha = select_best_alpha(alpha_df)["alpha"]
    print("Saved evaluation outputs:")
    print(f"- {eval_path}")
    print(f"- {alpha_path}")
    print(f"Best alpha (by NDCG@{TOP_K}, recall tie-breaker): {best_alpha}")


if __name__ == "__main__":
    main()
