"""
Backwards-compatible imports for model utilities.
New code should import from algorithms.models.* directly.
"""

from __future__ import annotations

from algorithms.models.hybrid_model import compute_hybrid_scores
from algorithms.models.lightfm_model import (
    build_lightfm_dataset,
    predict_lightfm_scores_for_user,
    train_lightfm,
)
from algorithms.models.tfidf_model import (
    build_tfidf_representations,
    compute_content_scores_for_user,
    compute_tfidf_scores_for_user,
)

__all__ = [
    "build_tfidf_representations",
    "compute_content_scores_for_user",
    "compute_tfidf_scores_for_user",
    "build_lightfm_dataset",
    "predict_lightfm_scores_for_user",
    "train_lightfm",
    "compute_hybrid_scores",
]
