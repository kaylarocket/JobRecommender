"""
Common ranking metrics used across recommenders.
"""

from __future__ import annotations

import math
from typing import Iterable, Set

import numpy as np


def precision_at_k(predicted: Iterable[str], actual: Set[str], k: int) -> float:
    """
    Precision@k for binary relevance.
    """
    if k == 0:
        return 0.0
    predicted_list = list(predicted)[:k]
    hits = len(set(predicted_list) & actual)
    return hits / float(k)


def recall_at_k(predicted: Iterable[str], actual: Set[str], k: int) -> float:
    """
    Recall@k for binary relevance.
    """
    if not actual:
        return 0.0
    predicted_list = list(predicted)[:k]
    hits = len(set(predicted_list) & actual)
    return hits / float(len(actual))


def dcg_at_k(predicted: Iterable[str], actual: Set[str], k: int) -> float:
    """
    Discounted Cumulative Gain @k with binary relevance.
    """
    dcg = 0.0
    for rank, item_id in enumerate(list(predicted)[:k], start=1):
        rel = 1.0 if item_id in actual else 0.0
        dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(predicted: Iterable[str], actual: Set[str], k: int) -> float:
    """
    Normalized DCG @k with binary relevance.
    """
    if not actual or k == 0:
        return 0.0
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(predicted, actual, k) / idcg


def mean(values: Iterable[float]) -> float:
    """
    Safe mean that returns 0.0 for empty iterables.
    """
    values_list = list(values)
    return float(np.mean(values_list)) if values_list else 0.0


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "mean",
]
