"""
Purpose-Action Relevance AI
============================
Architecture:
  1. Dual Embedding Stage  : Independently embed "purpose" and "action" texts
  2. Cross-Attention Fusion: Learn cross-modal interactions between the two vectors
  3. Relevance Head        : MLP that outputs a 0-1 relevance score

Embedding back-ends supported:
  - sentence-transformers  (local, free)
  - OpenAI text-embedding-3-small (API key required)
"""

import os
import json
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1.  Embedding back-ends
# ──────────────────────────────────────────────

class EmbeddingBackend:
    """Abstract base – returns numpy arrays of shape (dim,)."""
    dim: int

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model_name = model_name
        self.dim = 1536  # text-embedding-3-small default

    def encode(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(input=text, model=self._model_name)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)


def build_embedding_backend(
    backend: Literal["sentence-transformers", "openai"] = "sentence-transformers",
    **kwargs,
) -> EmbeddingBackend:
    if backend == "sentence-transformers":
        return SentenceTransformerBackend(**kwargs)
    elif backend == "openai":
        return OpenAIEmbeddingBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ──────────────────────────────────────────────
# 2.  Neural Relevance Network
# ──────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention between two token-less vectors.

    We treat each embedding as a single 'token' and apply
    multi-head cross-attention in both directions so the network can
    learn which dimensions of purpose are relevant to which dimensions
    of action – much richer than simple concatenation.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, purpose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        purpose, action: (B, dim)
        Returns fused vector: (B, 2*dim)
        """
        # Unsqueeze to (B, 1, dim) for attention API
        p = purpose.unsqueeze(1)
        a = action.unsqueeze(1)

        # purpose attends to action
        p_out, _ = self.attn(
            self.q_proj(p), self.k_proj(a), self.v_proj(a)
        )
        p_fused = self.norm1(p + p_out).squeeze(1)

        # action attends to purpose
        a_out, _ = self.attn(
            self.q_proj(a), self.k_proj(p), self.v_proj(p)
        )
        a_fused = self.norm2(a + a_out).squeeze(1)

        return torch.cat([p_fused, a_fused], dim=-1)   # (B, 2*dim)


class RelevanceHead(nn.Module):
    """MLP that maps fused vector → scalar relevance ∈ (0, 1)."""

    def __init__(self, in_dim: int, hidden_dims: list[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 64]

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class RelevanceNet(nn.Module):
    """Full two-tower + cross-attention + MLP head."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Projection layers: independently transform each embedding
        # These are the 'trainable' parts for each tower.
        self.purpose_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.fusion = CrossAttentionFusion(embedding_dim, num_heads, dropout)
        self.head = RelevanceHead(embedding_dim * 2, hidden_dims, dropout)

    def forward(
        self,
        purpose_emb: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        p = self.purpose_proj(purpose_emb)
        a = self.action_proj(action_emb)
        fused = self.fusion(p, a)
        return self.head(fused)
