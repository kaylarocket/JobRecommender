"""
Hybrid scorer that blends content and collaborative signals.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import minmax_scale


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


__all__ = ["compute_hybrid_scores"]
