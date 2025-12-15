"""
Inference helpers for the hybrid recommender.
These functions are meant to be called by external clients (e.g., Flutter app).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from models import (
    compute_content_scores_for_user,
    compute_hybrid_scores,
    predict_lightfm_scores_for_user,
)


def load_trained_models(
    model_path: Optional[str] = None,
    vectorizer_path: Optional[str] = None,
    job_matrix_path: Optional[str] = None,
    user_matrix_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Placeholder for loading persisted artifacts (LightFM model, TF-IDF vectorizer, matrices).
    Replace with joblib/pickle loading logic when serialization is added.
    """
    _ = model_path, vectorizer_path, job_matrix_path, user_matrix_path
    raise NotImplementedError("Model loading not yet implemented. Add joblib/pickle loading here.")


def recommend_for_user_id(
    user_id: str,
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    model,
    dataset,
    alpha: float = 0.6,
    top_k: int = 10,
    user_features_matrix=None,
    item_features_matrix=None,
) -> pd.DataFrame:
    """
    Return top-N recommendations for a given user_id with component scores.
    """
    if "user_id" not in users.columns:
        raise ValueError("users DataFrame must contain user_id column.")

    user_lookup = {uid: idx for idx, uid in enumerate(users["user_id"])}
    if user_id not in user_lookup:
        raise ValueError(f"user_id {user_id} not found in provided users DataFrame.")

    user_idx = user_lookup[user_id]
    content_scores = compute_content_scores_for_user(user_idx, job_tfidf, user_tfidf)
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
    return rec_df.reset_index(drop=True)

