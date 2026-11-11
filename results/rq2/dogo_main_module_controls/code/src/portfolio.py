"""Train-only portfolio construction, coverage diagnostics, and one-swap refinement."""

from __future__ import annotations

import numpy as np

from .common import normalized_text
from .objectives import coverage_jaccard


def coverage_masks(query_vectors, text_vectors, threshold):
    return np.asarray(query_vectors @ np.asarray(text_vectors).T >= threshold, dtype=bool).T


def portfolio_metrics(query_vectors, text_vectors, threshold):
    masks = coverage_masks(query_vectors, text_vectors, threshold)
    union = masks.any(axis=0) if len(masks) else np.zeros(len(query_vectors), dtype=bool)
    overlaps = [coverage_jaccard(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
    scores = query_vectors @ np.asarray(text_vectors).T if len(text_vectors) else np.empty((len(query_vectors), 0))
    return {
        "covered": int(union.sum()), "total": int(len(query_vectors)),
        "coverage": float(union.mean()) if len(union) else 0.0,
        "mean_best_cosine": float(scores.max(axis=1).mean()) if scores.shape[1] else 0.0,
        "mean_pairwise_coverage_jaccard": float(np.mean(overlaps)) if overlaps else 0.0,
        "max_pairwise_coverage_jaccard": float(np.max(overlaps)) if overlaps else 0.0,
    }


def _score(indices, rows, vectors, train, threshold):
    metrics = portfolio_metrics(train, vectors[indices], threshold)
    ppls = [rows[i].get("ppl") for i in indices if rows[i].get("ppl") is not None]
    mean_ppl = float(np.mean(ppls)) if ppls else 0.0
    return (
        metrics["covered"], metrics["mean_best_cosine"],
        -metrics["max_pairwise_coverage_jaccard"], -mean_ppl,
    )


def greedy_initialize(rows, vectors, train, threshold, budget):
    """Choose a text portfolio by exact Train marginal coverage.

    Search rounds only determine where candidates came from. A good candidate
    from one round may cover queries associated with several target guides.
    """
    if budget < 1 or budget > len(rows) or len(rows) != len(vectors):
        raise ValueError("Invalid greedy portfolio budget or candidate pool")
    masks = coverage_masks(train, vectors, threshold)
    identities = [normalized_text(row["text"]) for row in rows]
    eligible = np.ones(len(rows), dtype=bool)
    covered = np.zeros(len(train), dtype=bool)
    selected = []
    for _ in range(budget):
        gains = np.count_nonzero(masks & ~covered[None, :], axis=1)
        gains[~eligible] = -1
        index = int(np.argmax(gains))
        if gains[index] < 0:
            raise ValueError("Too few distinct candidate texts for the budget")
        selected.append(index)
        covered |= masks[index]
        identity = identities[index]
        eligible &= np.asarray([value != identity for value in identities])
    return selected


def optimize_one_swap(rows, vectors, train, threshold, initial_indices, passes):
    selected = list(initial_indices)
    if len(selected) != len(set(selected)):
        raise ValueError("Initial portfolio contains duplicate indices")
    if len({normalized_text(rows[i]["text"]) for i in selected}) != len(selected):
        raise ValueError("Initial portfolio contains duplicate texts")
    if passes == 0:
        return selected, []
    # Reuse the same exact Train cosines and threshold masks across swap trials.
    # The original implementation re-encoded every trial against all Train rows,
    # which is prohibitively slow when K=15 and the archive has thousands of texts.
    scores = train @ np.asarray(vectors).T
    masks = scores >= threshold
    identities = [normalized_text(row["text"]) for row in rows]

    def cached_score(indices, covered=None):
        chosen_masks = masks[:, indices].T
        if covered is None:
            covered = int(chosen_masks.any(axis=0).sum())
        mean_best = float(scores[:, indices].max(axis=1).mean())
        maximum_overlap = max(
            (coverage_jaccard(chosen_masks[i], chosen_masks[j])
             for i in range(len(indices)) for j in range(i + 1, len(indices))),
            default=0.0,
        )
        ppls = [rows[i].get("ppl") for i in indices if rows[i].get("ppl") is not None]
        mean_ppl = float(np.mean(ppls)) if ppls else 0.0
        return covered, mean_best, -maximum_overlap, -mean_ppl

    history = []
    for pass_index in range(1, passes + 1):
        before = cached_score(selected)
        winner, winner_score = None, before
        for position in range(len(selected)):
            others = selected[:position] + selected[position + 1:]
            other_union = masks[:, others].any(axis=1) if others else np.zeros(len(train), dtype=bool)
            covered = int(other_union.sum()) + np.count_nonzero(
                masks & ~other_union[:, None], axis=0
            )
            other_identities = {identities[index] for index in others}
            for candidate in np.flatnonzero(covered >= winner_score[0]):
                candidate = int(candidate)
                if candidate in selected:
                    continue
                if identities[candidate] in other_identities:
                    continue
                trial = selected.copy()
                trial[position] = candidate
                score = cached_score(trial, int(covered[candidate]))
                if score > winner_score:
                    winner, winner_score = trial, score
        if winner is None:
            break
        history.append({
            "pass": pass_index, "before_covered": int(before[0]),
            "after_covered": int(winner_score[0]), "selected_indices": winner,
        })
        selected = winner
    return selected, history


def threshold_sweep(query_vectors, text_vectors, thresholds):
    return [
        {"threshold": float(value), **portfolio_metrics(query_vectors, text_vectors, value)}
        for value in thresholds
    ]
