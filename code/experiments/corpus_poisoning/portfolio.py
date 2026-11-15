"""Pure Train-only coverage and beam-selection helpers for the hybrid search."""

from __future__ import annotations

import numpy as np


def hit_mask(train: np.ndarray, vector: np.ndarray, threshold: float) -> np.ndarray:
    return (train @ vector) >= threshold


def union_mask(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("At least one coverage mask is required")
    return np.logical_or.reduce(masks)


def residual_without(masks: list[np.ndarray], slot: int) -> np.ndarray:
    if not 0 <= slot < len(masks):
        raise IndexError(slot)
    other = [mask for index, mask in enumerate(masks) if index != slot]
    return ~union_mask(other) if other else np.ones_like(masks[slot], dtype=bool)


def smooth_near_miss(scores: np.ndarray, residual: np.ndarray, threshold: float,
                     temperature: float) -> float:
    if not np.any(residual):
        return 0.0
    values = np.clip((scores[residual] - threshold) / temperature, -30, 30)
    return float((1.0 / (1.0 + np.exp(-values))).sum())


def rank_candidate(row: dict, residual: np.ndarray, guide: np.ndarray,
                   threshold: float, temperature: float) -> tuple:
    hits = row["mask"]
    scores = row["scores"]
    gain = int((hits & residual).sum())
    guide_gain = int((hits & residual & guide).sum())
    soft = smooth_near_miss(scores, residual & guide, threshold, temperature)
    return gain, guide_gain, soft, float(row["target_cosine"]), -float(row.get("ppl") or 0)


def select_beam(rows: list[dict], residual: np.ndarray, guide: np.ndarray,
                threshold: float, temperature: float, width: int) -> list[dict]:
    """Keep hard-coverage leaders plus one near-threshold exploration branch."""
    if width < 1:
        raise ValueError("width must be positive")
    unique = {}
    for row in rows:
        unique.setdefault(row["text"].casefold().strip(), row)
    ordered = sorted(unique.values(), key=lambda row: rank_candidate(
        row, residual, guide, threshold, temperature), reverse=True)
    selected = ordered[:width]
    if width > 1 and len(ordered) > width:
        explorer = max(ordered, key=lambda row: (
            smooth_near_miss(row["scores"], residual & guide, threshold, temperature),
            rank_candidate(row, residual, guide, threshold, temperature)))
        if all(explorer is not row for row in selected):
            selected[-1] = explorer
    return selected


def greedy_portfolio(rows: list[dict], budget: int) -> list[dict]:
    if budget < 1 or len(rows) < budget:
        raise ValueError("Not enough rows for the requested budget")
    selected = []
    covered = np.zeros_like(rows[0]["mask"], dtype=bool)
    remaining = list(rows)
    for _ in range(budget):
        winner = max(remaining, key=lambda row: (
            int((row["mask"] & ~covered).sum()),
            float(row["target_cosine"]), -float(row.get("ppl") or 0)))
        selected.append(winner)
        covered |= winner["mask"]
        remaining = [row for row in remaining if row is not winner]
    return selected


def swap_portfolio(selected: list[dict], pool: list[dict], passes: int) -> list[dict]:
    selected = list(selected)
    for _ in range(passes):
        improved = False
        for slot in range(len(selected)):
            residual = residual_without([row["mask"] for row in selected], slot)
            current_gain = int((selected[slot]["mask"] & residual).sum())
            available = [row for row in pool if all(
                row is not selected[other] for other in range(len(selected)) if other != slot)]
            winner = max(available, key=lambda row: int((row["mask"] & residual).sum()))
            new_gain = int((winner["mask"] & residual).sum())
            if new_gain > current_gain:
                selected[slot] = winner
                improved = True
        if not improved:
            break
    return selected
