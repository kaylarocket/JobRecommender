"""
Sentence-BERT content-based recommender utilities.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "sentence-transformers is required for the SBERT recommender. "
            "Install with `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer


def _require_torch(seed: int):
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyTorch is required for the SBERT recommender. "
            "Install with `pip install torch`."
        ) from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch


def build_sbert_representations(
    users: np.ndarray,
    jobs: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Encode jobs and users with SBERT. Returns (model, job_embeddings, user_embeddings).
    """
    SentenceTransformer = _require_sentence_transformers()
    torch = _require_torch(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)

    model = SentenceTransformer(model_name, device=device)

    job_embeddings = model.encode(
        jobs.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    user_embeddings = model.encode(
        users.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return model, job_embeddings, user_embeddings


def compute_sbert_scores_for_user(
    user_idx: int,
    job_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Cosine similarity between a user's SBERT embedding and all job embeddings.
    """
    user_vec = user_embeddings[user_idx]
    # Embeddings are already normalized; dot product is cosine similarity.
    return job_embeddings @ user_vec


__all__ = [
    "build_sbert_representations",
    "compute_sbert_scores_for_user",
]
