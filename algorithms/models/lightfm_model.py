"""
LightFM collaborative filtering helpers.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


def _extract_feature_vocab(feature_tuples: Sequence[Tuple[str, Iterable[str]]]) -> list[str]:
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


__all__ = [
    "build_lightfm_dataset",
    "train_lightfm",
    "predict_lightfm_scores_for_user",
]
