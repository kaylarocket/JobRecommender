"""
TF-IDF content-based recommender utilities.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_representations(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: Optional[str] = "english",
):
    """
    Vectorize job_text and user_text jointly so cosine similarity is meaningful.
    """
    corpus = jobs["job_text"].tolist() + users["user_text"].tolist()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_jobs = len(jobs)
    job_tfidf = tfidf_matrix[:n_jobs]
    user_tfidf = tfidf_matrix[n_jobs:]
    return vectorizer, job_tfidf, user_tfidf


def compute_tfidf_scores_for_user(
    user_idx: int,
    job_tfidf,
    user_tfidf,
) -> np.ndarray:
    """
    Compute cosine similarity between a user's TF-IDF vector and all job vectors.
    """
    return cosine_similarity(user_tfidf[user_idx], job_tfidf).ravel()


# Backward-compatible alias
compute_content_scores_for_user = compute_tfidf_scores_for_user


__all__ = [
    "build_tfidf_representations",
    "compute_tfidf_scores_for_user",
    "compute_content_scores_for_user",
]
