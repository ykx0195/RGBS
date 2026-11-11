"""Frozen BGE loading and normalized sentence encoding."""

from __future__ import annotations

import numpy as np


def load_encoder(model_path: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, device=device).eval()


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
