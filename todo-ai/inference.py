"""
Inference engine – load a trained model and score (purpose, action) pairs.
"""

import torch
from model import RelevanceNet, build_embedding_backend


class RelevanceAI:
    """
    High-level interface for the trained Purpose-Action Relevance AI.

    Usage
    -----
    ai = RelevanceAI.load("relevance_model.pt")
    score = ai.score("일본어 실력 향상", "자막 없이 일본 애니메이션 시청")
    print(score)   # e.g. 0.74
    """

    def __init__(self, model: RelevanceNet, backend, device: str):
        self.model = model.to(device)
        self.model.eval()
        self.backend = backend
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str, device: str | None = None) -> "RelevanceAI":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint_path, map_location=device)
        backend = build_embedding_backend(
            ckpt["embedding_backend"],
            **ckpt.get("embedding_kwargs", {}),
        )
        model = RelevanceNet(embedding_dim=ckpt["embedding_dim"])
        model.load_state_dict(ckpt["model_state"])
        return cls(model, backend, device)

    @torch.no_grad()
    def score(self, purpose: str, action: str) -> float:
        """
        Returns a relevance score in [0, 1].
        Closer to 1 = more relevant to the purpose.
        """
        p_emb = torch.tensor(
            self.backend.encode(purpose), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        a_emb = torch.tensor(
            self.backend.encode(action), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        return float(self.model(p_emb, a_emb).item())

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (purpose, action) pairs at once."""
        return [self.score(p, a) for p, a in pairs]

    def describe(self, purpose: str, action: str) -> str:
        """Human-readable relevance description."""
        s = self.score(purpose, action)
        if s >= 0.8:
            level = "매우 높음 🟢"
        elif s >= 0.6:
            level = "높음 🟡"
        elif s >= 0.4:
            level = "보통 🟠"
        elif s >= 0.2:
            level = "낮음 🔴"
        else:
            level = "매우 낮음 ⚫"

        return (
            f"목적  : {purpose}\n"
            f"행위  : {action}\n"
            f"연관성: {s:.3f}  ({level})\n"
        )
