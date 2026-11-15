"""Shared geometry, coverage scoring, and the original v3 sphere optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Settings:
    threshold: float = 0.85
    temperature: float = 0.02
    steps: int = 20
    learning_rate: float = 0.05
    starts: int = 6
    near_weight: float = 0.25
    density_k: int = 16
    density_floor: float = 0.1


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def validate_train(train: np.ndarray, k: int) -> np.ndarray:
    train = np.asarray(train, dtype=np.float32)
    if train.ndim != 2 or k < 1 or k > len(train):
        raise ValueError("Expected a [n, d] Train array and 1 <= k <= n")
    if not np.isfinite(train).all():
        raise ValueError("Train contains NaN or infinity")
    if not np.allclose(np.linalg.norm(train, axis=1), 1.0, atol=0.002):
        raise ValueError("Train embeddings must already be L2-normalized")
    return unit(train)


def coverage_mask(train: np.ndarray, vectors: np.ndarray, threshold: float) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros(len(train), dtype=bool)
    return np.max(train @ vectors.T, axis=1) >= threshold


def greedy_order(vectors: np.ndarray, train: np.ndarray, threshold: float) -> tuple[np.ndarray, list[int]]:
    """Order a fixed vector set by Train marginal union coverage; do not change it."""
    hits = train @ vectors.T >= threshold
    covered = np.zeros(len(train), dtype=bool)
    remaining = set(range(len(vectors)))
    order, gains = [], []
    while remaining:
        index = max(remaining, key=lambda i: (int(np.count_nonzero(hits[:, i] & ~covered)), -i))
        gain = int(np.count_nonzero(hits[:, index] & ~covered))
        covered |= hits[:, index]
        order.append(index)
        gains.append(gain)
        remaining.remove(index)
    return np.asarray(order, dtype=np.int32), gains


def density_weights(train: np.ndarray, k: int, floor: float, chunk: int = 256) -> np.ndarray:
    """Global Train k-neighbor density weights from the original v3 optimizer."""
    if len(train) < 2:
        return np.ones(len(train), dtype=np.float32)
    k = min(k, len(train) - 1)
    values = np.empty(len(train), dtype=np.float32)
    for start in range(0, len(train), chunk):
        similarities = train[start:start + chunk] @ train.T
        row = np.arange(len(similarities))
        similarities[row, start + row] = -np.inf
        neighbors = np.partition(similarities, -k, axis=1)[:, -k:]
        values[start:start + len(similarities)] = neighbors.mean(axis=1)
    median = float(np.median(values))
    return np.clip(values / max(median, 1e-8), floor, 1.0).astype(np.float32)


def soft_score(vector: np.ndarray, targets: np.ndarray, settings: Settings) -> float:
    logits = np.clip((targets @ vector - settings.threshold) / settings.temperature, -50, 50)
    return float((1.0 / (1.0 + np.exp(-logits))).mean())


def local_score(vector: np.ndarray, targets: np.ndarray, weights: np.ndarray,
                settings: Settings) -> tuple[float, int, float]:
    hits = targets @ vector >= settings.threshold
    return float(weights[hits].sum()), int(hits.sum()), soft_score(vector, targets, settings)


def optimize_on_sphere(start: np.ndarray, targets: np.ndarray, weights: np.ndarray,
                       settings: Settings, device: torch.device) -> np.ndarray:
    """Sigmoid threshold surrogate; tangent gradient followed by normalization."""
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    z = F.normalize(torch.as_tensor(start, dtype=torch.float32, device=device), dim=0)
    for _ in range(settings.steps):
        z = z.detach().requires_grad_(True)
        similarities = target_tensor @ z
        near = torch.sigmoid((similarities - settings.threshold) / settings.temperature)
        sharp = torch.sigmoid((similarities - settings.threshold) /
                              max(settings.temperature * 0.5, 1e-4))
        objective = ((settings.near_weight * near + sharp) * weight_tensor).mean()
        gradient, = torch.autograd.grad(objective, z)
        tangent = gradient - z * torch.dot(z, gradient)
        z = F.normalize(z + settings.learning_rate * tangent, dim=0)
    return z.detach().cpu().numpy().astype(np.float32)


def best_natural_starts(targets: np.ndarray, anchors: np.ndarray, settings: Settings) -> list[np.ndarray]:
    hard = np.empty(len(anchors), dtype=np.int32)
    soft = np.empty(len(anchors), dtype=np.float32)
    for start in range(0, len(anchors), 256):
        scores = anchors[start:start + 256] @ targets.T
        hard[start:start + len(scores)] = (scores >= settings.threshold).sum(axis=1)
        logits = np.clip((scores - settings.threshold) / settings.temperature, -50, 50)
        soft[start:start + len(scores)] = (1 / (1 + np.exp(-logits))).sum(axis=1)
    indices = np.lexsort((-soft, -hard))[:settings.starts]
    return [anchors[int(index)] for index in indices]


def optimize_one_guide(targets: np.ndarray, weights: np.ndarray, anchors: np.ndarray,
                       settings: Settings, device: torch.device,
                       extra_start: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    center = unit(np.average(targets, axis=0, weights=weights))
    starts = [("weighted_centroid", center)]
    starts += [(f"natural_anchor_{i}", anchor)
               for i, anchor in enumerate(best_natural_starts(targets, anchors, settings))]
    if extra_start is not None:
        starts.append(("kmeans_centroid", unit(extra_start)))
    best_vector, best_name, best_score = None, None, None
    for name, start in starts:
        for suffix, vector in (("start", start),
                               ("optimized", optimize_on_sphere(start, targets, weights, settings, device))):
            score = local_score(vector, targets, weights, settings)
            if best_score is None or score > best_score:
                best_vector, best_name, best_score = vector, f"{name}:{suffix}", score
    return unit(best_vector), {"selected_start": best_name,
                               "weighted_hard": best_score[0], "hard": best_score[1],
                               "soft": best_score[2], "start_count": len(starts)}
