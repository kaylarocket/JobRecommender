"""
Train the Neural Collaborative Filtering (NCF) recommender and export sample recommendations.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Set

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
from algorithms.models.ncf_model import (
    build_ncf_training_data,
    predict_ncf_scores_for_user,
    train_ncf_model,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
TOP_K = 10
N_USERS_PREVIEW = 5


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


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

    interactions_df = build_synthetic_interactions(users, jobs)

    (
        user_indices,
        item_indices,
        labels,
        user_index,
        job_index,
    ) = build_ncf_training_data(interactions_df, users, jobs)
    if len(labels) == 0:
        raise ValueError("NCF training data is empty. Check interactions preprocessing.")

    model = train_ncf_model(
        user_indices=user_indices,
        item_indices=item_indices,
        labels=labels,
        n_users=len(users),
        n_items=len(jobs),
        epochs=4,
        embedding_dim=32,
    )

    job_lookup = {jid: idx for idx, jid in enumerate(jobs["job_id"])}
    previews = []
    for i in range(min(N_USERS_PREVIEW, len(users))):
        user_id = users.loc[i, "user_id"]
        seen = set(interactions_df.loc[interactions_df["user_id"] == user_id, "job_id"])
        scores = predict_ncf_scores_for_user(
            user_id=user_id,
            model=model,
            job_index=job_index,
            user_index=user_index,
            n_items=len(jobs),
        )
        ranked_jobs = _rank_jobs(scores, jobs, job_lookup, seen, TOP_K)
        for rank, job_id in enumerate(ranked_jobs, start=1):
            previews.append({"user_id": user_id, "job_id": job_id, "rank": rank, "score": scores[job_lookup[job_id]]})

    if previews:
        preview_df = pd.DataFrame(previews)
        out_path = OUT_DIR / "sample_ncf_recommendations.csv"
        preview_df.to_csv(out_path, index=False)
        print(f"Saved sample recommendations to {out_path}")


if __name__ == "__main__":
    main()
