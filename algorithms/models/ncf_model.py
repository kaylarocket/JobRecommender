"""
Neural Collaborative Filtering (NCF) utilities built with PyTorch.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyTorch is required for the NCF recommender. "
            "Install with `pip install torch`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


class NCFModel:
    """
    Thin wrapper around a torch.nn.Module for type-friendly export.
    """

    def __init__(self, module, device: str):
        self.module = module
        self.device = device

    def eval(self):
        self.module.eval()

    def parameters(self):
        return self.module.parameters()


def _build_module(
    n_users: int,
    n_items: int,
    embedding_dim: int,
    hidden_layers: Tuple[int, ...],
    dropout: float,
    nn_module,
    torch_module,
):
    layers = []
    input_dim = embedding_dim * 2
    for hidden in hidden_layers:
        layers.extend(
            [
                nn_module.Linear(input_dim, hidden),
                nn_module.ReLU(),
                nn_module.Dropout(dropout),
            ]
        )
        input_dim = hidden
    layers.append(nn_module.Linear(input_dim, 1))
    mlp = nn_module.Sequential(*layers)

    class _NCF(nn_module.Module):
        def __init__(self):
            super().__init__()
            self.user_embedding = nn_module.Embedding(n_users, embedding_dim)
            self.item_embedding = nn_module.Embedding(n_items, embedding_dim)
            self.mlp = mlp

        def forward(self, user_indices, item_indices):
            user_vecs = self.user_embedding(user_indices)
            item_vecs = self.item_embedding(item_indices)
            x = nn_module.functional.dropout(
                torch_module.cat([user_vecs, item_vecs], dim=1),
                p=dropout,
                training=self.training,
            )
            logits = self.mlp(x).squeeze(-1)
            return logits

    return _NCF()


def build_ncf_training_data(
    train_interactions: pd.DataFrame,
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Build (user_idx, item_idx, label) triples with per-user negative sampling.
    """
    rng = np.random.default_rng(seed)
    user_index = {uid: idx for idx, uid in enumerate(users["user_id"])}
    job_index = {jid: idx for idx, jid in enumerate(jobs["job_id"])}
    n_items = len(job_index)

    user_indices = []
    item_indices = []
    labels = []

    for user_id, group in train_interactions.groupby("user_id"):
        if user_id not in user_index:
            continue
        uidx = user_index[user_id]
        pos_items = [job_index[jid] for jid in group["job_id"] if jid in job_index]
        if not pos_items:
            continue
        pos_set = set(pos_items)
        candidates = np.array([i for i in range(n_items) if i not in pos_set])

        for pos_item in pos_items:
            user_indices.append(uidx)
            item_indices.append(pos_item)
            labels.append(1.0)

            if len(candidates) == 0:
                continue
            n_neg = min(negatives_per_positive, len(candidates))
            sampled = rng.choice(candidates, size=n_neg, replace=False)
            user_indices.extend([uidx] * len(sampled))
            item_indices.extend(sampled.tolist())
            labels.extend([0.0] * len(sampled))

    return (
        np.array(user_indices, dtype=np.int64),
        np.array(item_indices, dtype=np.int64),
        np.array(labels, dtype=np.float32),
        user_index,
        job_index,
    )


def train_ncf_model(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    labels: np.ndarray,
    n_users: int,
    n_items: int,
    embedding_dim: int = 32,
    hidden_layers: Tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
    batch_size: int = 256,
    epochs: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: Optional[str] = None,
    seed: int = 42,
) -> NCFModel:
    """
    Train a lightweight NCF model on implicit feedback.
    """
    torch, nn_module, DataLoader, TensorDataset = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = _build_module(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        nn_module=nn_module,
        torch_module=torch,
    ).to(target_device)

    dataset = TensorDataset(
        torch.from_numpy(user_indices),
        torch.from_numpy(item_indices),
        torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn_module.BCEWithLogitsLoss()

    module.train()
    for _ in range(epochs):
        for batch_users, batch_items, batch_labels in loader:
            batch_users = batch_users.to(target_device)
            batch_items = batch_items.to(target_device)
            batch_labels = batch_labels.to(target_device)

            logits = module(batch_users, batch_items)
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    module.eval()
    return NCFModel(module=module, device=target_device)


def predict_ncf_scores_for_user(
    user_id: str,
    model: NCFModel,
    job_index: Dict[str, int],
    user_index: Dict[str, int],
    n_items: int,
) -> np.ndarray:
    """
    Predict sigmoid scores for all items for a given user.
    """
    torch, _, _, _ = _require_torch()
    if user_id not in user_index:
        return np.zeros(n_items)

    device = torch.device(model.device)
    uidx = user_index[user_id]
    user_tensor = torch.full((n_items,), uidx, dtype=torch.long, device=device)
    item_tensor = torch.arange(n_items, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.module(user_tensor, item_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores


__all__ = [
    "build_ncf_training_data",
    "train_ncf_model",
    "predict_ncf_scores_for_user",
    "NCFModel",
]
