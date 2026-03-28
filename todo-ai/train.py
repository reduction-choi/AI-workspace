"""
Dataset, caching, and training utilities.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

from model import RelevanceNet, build_embedding_backend, EmbeddingBackend


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class RelevanceDataset(Dataset):
    """
    Loads training data from a JSON file.

    Expected JSON format  (list of dicts):
    [
      {
        "purpose": "일본어 실력 향상",
        "action":  "자막 없이 일본 애니메이션 시청",
        "label":   0.75
      },
      ...
    ]
    Labels should be floats in [0, 1].
    Binary (0/1) labels are also accepted and will be cast to float.
    """

    def __init__(
        self,
        data_path: str,
        backend: EmbeddingBackend,
        cache_dir: Optional[str] = None,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.purpose_embs: list[np.ndarray] = []
        self.action_embs: list[np.ndarray] = []
        self.labels: list[float] = []

        for i, item in enumerate(raw):
            purpose = item["purpose"]
            action = item["action"]
            label = float(item["label"])

            p_emb = self._get_embedding(backend, "purpose", i, purpose)
            a_emb = self._get_embedding(backend, "action", i, action)

            self.purpose_embs.append(p_emb)
            self.action_embs.append(a_emb)
            self.labels.append(label)

        self.dim = self.purpose_embs[0].shape[0]

    def _get_embedding(
        self,
        backend: EmbeddingBackend,
        role: str,
        idx: int,
        text: str,
    ) -> np.ndarray:
        if self.cache_dir:
            cache_path = self.cache_dir / f"{role}_{idx}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        emb = backend.encode(text)

        if self.cache_dir:
            with open(cache_path, "wb") as f:
                pickle.dump(emb, f)

        return emb

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.purpose_embs[idx], dtype=torch.float32),
            torch.tensor(self.action_embs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(
    data_path: str,
    save_path: str = "relevance_model.pt",
    embedding_backend: str = "sentence-transformers",
    embedding_kwargs: Optional[dict] = None,
    val_split: float = 0.15,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    cache_dir: str = "emb_cache",
    device: Optional[str] = None,
) -> dict:
    """
    Full training pipeline.

    Returns a dict with training history and best validation loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    # --- Embedding backend ---
    backend = build_embedding_backend(embedding_backend, **(embedding_kwargs or {}))
    print(f"[train] embedding dim={backend.dim}")

    # --- Dataset & splits ---
    dataset = RelevanceDataset(data_path, backend, cache_dir=cache_dir)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print(f"[train] train={n_train}, val={n_val}")

    # --- Model ---
    model = RelevanceNet(embedding_dim=backend.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # MSE loss works well for continuous [0,1] labels.
    # If you use strictly binary labels, BCELoss is equivalent here
    # because Sigmoid is already in the head.
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for p_emb, a_emb, labels in train_loader:
            p_emb, a_emb, labels = p_emb.to(device), a_emb.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(p_emb, a_emb)
            loss = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for p_emb, a_emb, labels in val_loader:
                p_emb, a_emb, labels = p_emb.to(device), a_emb.to(device), labels.to(device)
                preds = model(p_emb, a_emb)
                val_losses.append(criterion(preds, labels).item())

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "embedding_dim": backend.dim,
                    "embedding_backend": embedding_backend,
                    "embedding_kwargs": embedding_kwargs or {},
                },
                save_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

    print(f"[train] Best val loss: {best_val_loss:.4f}  → saved to {save_path}")
    return {"history": history, "best_val_loss": best_val_loss}
