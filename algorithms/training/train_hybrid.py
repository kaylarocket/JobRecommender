"""
End-to-end training script for the hybrid job recommender.
Runs TF-IDF + LightFM, generates sample recommendations, and saves outputs.

README:
- Run training locally with: python train_hybrid.py
- Outputs sample_user_recommendations.csv for quick inspection.
- The API (api_main.py) reuses the same TF-IDF/LightFM + compute_hybrid_scores pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
OUT_DIR = Path(__file__).resolve().parents[1] / "data"


from algorithms.core.data_loading import (
    MAX_JOBS,
    MAX_USERS,
    build_job_table,
    build_synthetic_interactions,
    build_user_table,
    load_applicants_dataset,
    load_jobstreet_job_dataset,
)
from algorithms.core.models import (
    build_lightfm_dataset,
    build_tfidf_representations,
    compute_content_scores_for_user,
    compute_hybrid_scores,
    predict_lightfm_scores_for_user,
    train_lightfm,
)

ALPHA_CONTENT = 0.6
TOP_K = 10
N_USERS_PREVIEW = 5


def generate_recommendations(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    model,
    dataset,
    user_features_matrix=None,
    item_features_matrix=None,
    alpha: float = ALPHA_CONTENT,
    top_k: int = TOP_K,
):
    """
    Compute hybrid recommendations for the first N_USERS_PREVIEW users.
    """
    all_recs = []
    for i in range(min(N_USERS_PREVIEW, len(users))):
        user_id = users.loc[i, "user_id"]

        content_scores = compute_content_scores_for_user(i, job_tfidf, user_tfidf)
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

        rec_df = jobs.copy()
        rec_df["content_score"] = content_norm
        rec_df["lfm_score"] = lfm_norm
        rec_df["final_score"] = hybrid_scores
        rec_df = rec_df.sort_values("final_score", ascending=False).head(top_k)

        print("=" * 60)
        print(f"Top recommendations for user {user_id}:")
        print(rec_df[["job_id", "final_score", "content_score", "lfm_score"]])

        rec_out = pd.DataFrame(
            {
                "user_id": user_id,
                "job_id": rec_df["job_id"].values,
                "rank": np.arange(1, len(rec_df) + 1),
                "final_score": rec_df["final_score"].values,
                "content_score": rec_df["content_score"].values,
                "lfm_score": rec_df["lfm_score"].values,
            }
        )
        all_recs.append(rec_out)
    return all_recs


def main():
    # 1) Load raw data
    raw_jobs = load_jobstreet_job_dataset()
    raw_applicants = load_applicants_dataset()

    # 2) Subsample for speed if necessary
    if len(raw_jobs) > MAX_JOBS:
        raw_jobs = raw_jobs.sample(MAX_JOBS, random_state=42)
    if len(raw_applicants) > MAX_USERS:
        raw_applicants = raw_applicants.sample(MAX_USERS, random_state=42)

    # 3) Feature engineering
    jobs = build_job_table(raw_jobs).reset_index(drop=True)
    users = build_user_table(raw_applicants).reset_index(drop=True)
    print(f"Prepared {len(jobs)} jobs and {len(users)} users.")

    # 4) Synthetic interactions
    interactions_df = build_synthetic_interactions(users, jobs)

    # 5) TF-IDF representations
    vectorizer, job_tfidf, user_tfidf = build_tfidf_representations(users, jobs)

    # 6) LightFM dataset + training
    dataset, interactions, weights, user_features_matrix, item_features_matrix = build_lightfm_dataset(
        interactions_df=interactions_df,
        users=users,
        jobs=jobs,
    )
    model = train_lightfm(
        interactions=interactions,
        weights=weights,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )

    # 7) Recommendations for sample users
    all_recs = generate_recommendations(
        users=users,
        jobs=jobs,
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        model=model,
        dataset=dataset,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        alpha=ALPHA_CONTENT,
        top_k=TOP_K,
    )

    # 8) Save sample recommendations
    if all_recs:
        df_out = pd.concat(all_recs, ignore_index=True)
        df_out.to_csv(OUT_DIR / "sample_user_recommendations.csv", index=False)
        print("Saved sample_user_recommendations.csv")


if __name__ == "__main__":
    main()
