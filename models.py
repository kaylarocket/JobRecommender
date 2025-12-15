"""
Model building blocks for the hybrid recommender:
- TF-IDF representations and cosine similarity
- LightFM collaborative filtering with optional user/item features
- Hybrid score computation
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from lightfm import LightFM
from lightfm.data import Dataset


def build_tfidf_representations(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: Optional[str] = "english",
):
    """
    Vectorize job_text + user_text jointly to place them in the same TF-IDF space.
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


def compute_content_scores_for_user(
    user_idx: int,
    job_tfidf,
    user_tfidf,
) -> np.ndarray:
    """
    Compute cosine similarity between a user's TF-IDF vector and all job vectors.
    """
    return cosine_similarity(user_tfidf[user_idx], job_tfidf).ravel()


def _extract_feature_vocab(feature_tuples: Sequence[Tuple[str, Sequence[str]]]) -> list[str]:
    vocab = set()
    for _, feats in feature_tuples:
        vocab.update(str(f) for f in feats)
    return sorted(vocab)


def _ensure_feature_tuples(df: pd.DataFrame, id_col: str) -> list[Tuple[str, list[str]]]:
    tuples: list[Tuple[str, list[str]]] = []
    for _, row in df.iterrows():
        features = row.get("features", [])
        if isinstance(features, str):
            features_list = [features]
        elif isinstance(features, Iterable):
            features_list = [str(f) for f in features]
        else:
            features_list = []
        tuples.append((str(row[id_col]), features_list))
    return tuples


def build_lightfm_dataset(
    interactions_df: pd.DataFrame,
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    user_features: Optional[pd.DataFrame] = None,
    item_features: Optional[pd.DataFrame] = None,
) -> Tuple[Dataset, object, object, Optional[object], Optional[object]]:
    """
    Build a LightFM Dataset along with interaction and optional feature matrices.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Must contain columns user_id, job_id, weight.
    user_features : pd.DataFrame, optional
        DataFrame with columns user_id and features (iterable or string).
    item_features : pd.DataFrame, optional
        DataFrame with columns job_id and features (iterable or string).
    """
    user_feature_tuples: Optional[list[Tuple[str, list[str]]]] = None
    item_feature_tuples: Optional[list[Tuple[str, list[str]]]] = None

    if user_features is not None:
        user_feature_tuples = _ensure_feature_tuples(user_features, "user_id")
    if item_features is not None:
        item_feature_tuples = _ensure_feature_tuples(item_features, "job_id")

    dataset = Dataset()
    dataset.fit(
        users=users["user_id"].tolist(),
        items=jobs["job_id"].tolist(),
        user_features=_extract_feature_vocab(user_feature_tuples) if user_feature_tuples else None,
        item_features=_extract_feature_vocab(item_feature_tuples) if item_feature_tuples else None,
    )

    tuples = [
        (row["user_id"], row["job_id"], float(row["weight"]))
        for _, row in interactions_df.iterrows()
    ]
    interactions, weights = dataset.build_interactions(tuples)

    user_features_matrix = (
        dataset.build_user_features(user_feature_tuples) if user_feature_tuples else None
    )
    item_features_matrix = (
        dataset.build_item_features(item_feature_tuples) if item_feature_tuples else None
    )

    return dataset, interactions, weights, user_features_matrix, item_features_matrix


def train_lightfm(
    interactions,
    weights,
    user_features=None,
    item_features=None,
    loss: str = "warp",
    no_components: int = 50,
    epochs: int = 15,
    num_threads: int = 4,
    random_state: int = 42,
) -> LightFM:
    """
    Train a LightFM model with provided interactions and optional features.
    """
    model = LightFM(
        loss=loss,
        no_components=no_components,
        random_state=random_state,
    )
    model.fit(
        interactions,
        sample_weight=weights,
        user_features=user_features,
        item_features=item_features,
        epochs=epochs,
        num_threads=num_threads,
    )
    return model


def predict_lightfm_scores_for_user(
    user_id: str,
    model: LightFM,
    dataset: Dataset,
    jobs: pd.DataFrame,
    user_features=None,
    item_features=None,
) -> np.ndarray:
    """
    Predict LightFM scores for a given user across all jobs.
    """
    n_jobs = len(jobs)
    user_id_map, _, item_id_map, _ = dataset.mapping()

    if user_id not in user_id_map:
        return np.zeros(n_jobs)

    uid = user_id_map[user_id]
    job_ids = jobs["job_id"].tolist()
    item_internal_ids = np.array([item_id_map[jid] for jid in job_ids])

    scores = model.predict(
        uid,
        item_internal_ids,
        user_features=user_features,
        item_features=item_features,
    )
    return scores


def compute_hybrid_scores(
    content_scores: np.ndarray,
    lfm_scores: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend normalized content and LightFM scores using weight alpha.
    """
    if len(content_scores) != len(lfm_scores):
        raise ValueError("content_scores and lfm_scores must have equal length.")

    content_norm = minmax_scale(content_scores) if len(content_scores) > 1 else content_scores
    lfm_norm = minmax_scale(lfm_scores) if len(lfm_scores) > 1 else lfm_scores

    hybrid = alpha * content_norm + (1 - alpha) * lfm_norm
    return hybrid, content_norm, lfm_norm

