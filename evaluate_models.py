"""
Evaluation pipeline for the hybrid recommender.
Performs train/test split on synthetic interactions, computes precision/recall
for TF-IDF, LightFM, and hybrid models, and sweeps alpha values.

README:
- Run: python evaluate_models.py
- Outputs evaluation_results.csv and alpha_tuning_results.csv to guide alpha selection.
- Mirrors the same TF-IDF + LightFM scoring that feeds the FastAPI service.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from data_loading import (
    MAX_JOBS,
    MAX_USERS,
    build_job_table,
    build_synthetic_interactions,
    build_user_table,
    load_applicants_dataset,
    load_jobstreet_job_dataset,
)
from models import (
    build_lightfm_dataset,
    build_tfidf_representations,
    compute_content_scores_for_user,
    compute_hybrid_scores,
    predict_lightfm_scores_for_user,
    train_lightfm,
)

TEST_SIZE = 0.2
TOP_K = 10
DEFAULT_ALPHA = 0.6
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


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


def precision_at_k(predicted: List[str], actual: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(predicted[:k]) & actual)
    return hits / float(k)


def recall_at_k(predicted: List[str], actual: Set[str], k: int) -> float:
    if len(actual) == 0:
        return 0.0
    hits = len(set(predicted[:k]) & actual)
    return hits / float(len(actual))


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


def evaluate_models(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    model,
    dataset,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    user_features_matrix=None,
    item_features_matrix=None,
    alpha: float = DEFAULT_ALPHA,
    top_k: int = TOP_K,
) -> Dict[str, float]:
    """
    Compute mean precision/recall for TF-IDF, LightFM, and hybrid models.
    """
    job_index = {jid: idx for idx, jid in enumerate(jobs["job_id"])}
    user_index = {uid: idx for idx, uid in enumerate(users["user_id"])}

    content_precisions: List[float] = []
    lfm_precisions: List[float] = []
    hybrid_precisions: List[float] = []

    content_recalls: List[float] = []
    lfm_recalls: List[float] = []
    hybrid_recalls: List[float] = []

    for user_id, group in test_interactions.groupby("user_id"):
        ground_truth = set(group["job_id"])
        if len(ground_truth) == 0 or user_id not in user_index:
            continue

        train_seen = set(train_interactions.loc[train_interactions["user_id"] == user_id, "job_id"])
        uidx = user_index[user_id]

        content_scores = compute_content_scores_for_user(uidx, job_tfidf, user_tfidf)
        lfm_scores = predict_lightfm_scores_for_user(
            user_id=user_id,
            model=model,
            dataset=dataset,
            jobs=jobs,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
        )

        hybrid_scores, content_norm, lfm_norm = compute_hybrid_scores(
            content_scores=content_scores,
            lfm_scores=lfm_scores,
            alpha=alpha,
        )

        content_ranked = _rank_jobs(content_norm, jobs, job_index, train_seen, top_k)
        lfm_ranked = _rank_jobs(lfm_norm, jobs, job_index, train_seen, top_k)
        hybrid_ranked = _rank_jobs(hybrid_scores, jobs, job_index, train_seen, top_k)

        content_precisions.append(precision_at_k(content_ranked, ground_truth, top_k))
        lfm_precisions.append(precision_at_k(lfm_ranked, ground_truth, top_k))
        hybrid_precisions.append(precision_at_k(hybrid_ranked, ground_truth, top_k))

        content_recalls.append(recall_at_k(content_ranked, ground_truth, top_k))
        lfm_recalls.append(recall_at_k(lfm_ranked, ground_truth, top_k))
        hybrid_recalls.append(recall_at_k(hybrid_ranked, ground_truth, top_k))

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "tfidf_precision": _mean(content_precisions),
        "tfidf_recall": _mean(content_recalls),
        "lightfm_precision": _mean(lfm_precisions),
        "lightfm_recall": _mean(lfm_recalls),
        "hybrid_precision": _mean(hybrid_precisions),
        "hybrid_recall": _mean(hybrid_recalls),
    }


def evaluate_alpha_sweep(
    alphas: List[float],
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    model,
    dataset,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    user_features_matrix=None,
    item_features_matrix=None,
    top_k: int = TOP_K,
) -> List[Dict[str, float]]:
    """
    Evaluate hybrid model performance across multiple alpha values.
    """
    results = []
    for alpha in alphas:
        metrics = evaluate_models(
            users=users,
            jobs=jobs,
            job_tfidf=job_tfidf,
            user_tfidf=user_tfidf,
            model=model,
            dataset=dataset,
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            user_features_matrix=user_features_matrix,
            item_features_matrix=item_features_matrix,
            alpha=alpha,
            top_k=top_k,
        )
        results.append(
            {
                "alpha": alpha,
                "precision_at_k": metrics["hybrid_precision"],
                "recall_at_k": metrics["hybrid_recall"],
            }
        )
    return results


def main():
    # 1) Load data
    raw_jobs = load_jobstreet_job_dataset()
    raw_applicants = load_applicants_dataset()

    if len(raw_jobs) > MAX_JOBS:
        raw_jobs = raw_jobs.sample(MAX_JOBS, random_state=42)
    if len(raw_applicants) > MAX_USERS:
        raw_applicants = raw_applicants.sample(MAX_USERS, random_state=42)

    jobs = build_job_table(raw_jobs).reset_index(drop=True)
    users = build_user_table(raw_applicants).reset_index(drop=True)
    print(f"Prepared {len(jobs)} jobs and {len(users)} users.")

    # 2) Synthetic interactions and split
    interactions_df = build_synthetic_interactions(users, jobs)
    train_interactions, test_interactions = train_test_split_interactions(interactions_df, test_size=TEST_SIZE)
    print(f"Train interactions: {len(train_interactions)}, Test interactions: {len(test_interactions)}")

    # 3) TF-IDF and LightFM training (train set only)
    _, job_tfidf, user_tfidf = build_tfidf_representations(users, jobs)
    dataset, interactions, weights, user_features_matrix, item_features_matrix = build_lightfm_dataset(
        interactions_df=train_interactions,
        users=users,
        jobs=jobs,
    )
    model = train_lightfm(
        interactions=interactions,
        weights=weights,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )

    # 4) Evaluate baseline models
    metrics = evaluate_models(
        users=users,
        jobs=jobs,
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        model=model,
        dataset=dataset,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        alpha=DEFAULT_ALPHA,
        top_k=TOP_K,
    )
    eval_df = pd.DataFrame(
        [
            {"model": "tfidf", "precision_at_k": metrics["tfidf_precision"], "recall_at_k": metrics["tfidf_recall"]},
            {"model": "lightfm", "precision_at_k": metrics["lightfm_precision"], "recall_at_k": metrics["lightfm_recall"]},
            {"model": f"hybrid_alpha_{DEFAULT_ALPHA}", "precision_at_k": metrics["hybrid_precision"], "recall_at_k": metrics["hybrid_recall"]},
        ]
    )
    eval_df.to_csv("evaluation_results.csv", index=False)
    print("Saved evaluation_results.csv")

    # 5) Alpha sweep
    alpha_results = evaluate_alpha_sweep(
        alphas=ALPHAS,
        users=users,
        jobs=jobs,
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        model=model,
        dataset=dataset,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        top_k=TOP_K,
    )
    alpha_df = pd.DataFrame(alpha_results)
    alpha_df.to_csv("alpha_tuning_results.csv", index=False)
    print("Saved alpha_tuning_results.csv")


if __name__ == "__main__":
    main()
