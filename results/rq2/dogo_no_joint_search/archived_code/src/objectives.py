"""Portfolio-conditioned fixed-threshold objectives.

The differentiable loss proposes edits. Exact hard marginal coverage from the real
encoder decides which candidates survive.
"""

import numpy as np
import torch


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60, 60)))


def coverage_jaccard(left, right):
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    union = left | right
    return float((left & right).sum() / union.sum()) if union.any() else 1.0


class ResidualCoverageObjective:
    def __init__(self, train, target, threshold, settings, covered_mask=None):
        self.train = np.asarray(train, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.threshold, self.settings = float(threshold), settings
        if covered_mask is None:
            covered_mask = np.zeros(len(self.train), dtype=bool)
        self.covered_mask = np.asarray(covered_mask, dtype=bool).copy()
        if self.covered_mask.shape != (len(self.train),):
            raise ValueError("covered_mask shape mismatch")
        self.residual_mask = ~self.covered_mask
        self.guide_mask = self.train @ self.target >= self.threshold
        self.guide_total = int(self.guide_mask.sum())
        if not self.guide_total:
            raise ValueError("Guide covers no Train queries")

    def coverage_mask(self, vector):
        return self.train @ vector >= self.threshold

    def metrics(self, vector):
        scores = self.train @ vector
        hits = scores >= self.threshold
        union = hits | self.covered_mask
        new = hits & self.residual_mask
        local = scores[self.guide_mask]
        retained = int(hits[self.guide_mask].sum())
        temperature = self.settings["coverage_temperature"]
        if self.residual_mask.any():
            residual_soft = float(sigmoid((scores[self.residual_mask] - self.threshold) / temperature).mean())
        else:
            residual_soft = 0.0
        return {
            "target_cosine": float(vector @ self.target),
            "train_covered": int(hits.sum()),
            "marginal_gain": int(new.sum()),
            "redundant_covered": int((hits & self.covered_mask).sum()),
            "union_covered": int(union.sum()),
            "union_coverage": float(union.mean()),
            "residual_total": int(self.residual_mask.sum()),
            "residual_soft_coverage": residual_soft,
            "guide_retained": retained,
            "guide_retention": retained / self.guide_total,
            "outside_guide_covered": int(hits.sum()) - retained,
            "guide_soft_coverage": float(sigmoid((local - self.threshold) / temperature).mean()),
            "guide_margin_deficit": float(
                np.maximum(self.threshold + self.settings["margin"] - local, 0).mean()
            ),
        }

    def loss(self, vector, target, train, method):
        settings, threshold = self.settings, self.threshold
        alignment = 1 - vector @ target
        if method == "gradient_alignment":
            return alignment
        scores = train @ vector
        soft = torch.sigmoid((scores - threshold) / settings["coverage_temperature"])
        if method == "gradient_coverage":
            return -soft.mean() + settings["coverage_alignment_weight"] * alignment

        residual = torch.as_tensor(self.residual_mask, device=vector.device)
        residual_term = -soft[residual].mean() if bool(residual.any().item()) else alignment * 0
        residual_term = residual_term + settings["residual_alignment_weight"] * alignment
        if method == "gradient_residual":
            return residual_term
        if method != "gradient_residual_retention":
            raise ValueError(f"Unknown objective: {method}")

        guide = torch.as_tensor(self.guide_mask, device=vector.device)
        local = scores[guide]
        detached = local.detach()
        weights = torch.ones_like(local)
        near = (detached >= threshold - settings["near_window"]) & (detached < threshold)
        at_risk = (detached >= threshold) & (detached < threshold + settings["margin"])
        weights[near] = settings["near_weight"]
        weights[at_risk] = settings["protect_weight"]
        deficit = (weights * torch.relu(threshold + settings["margin"] - local)).mean()
        retention = torch.sigmoid((local - threshold) / settings["coverage_temperature"]).mean()
        return (
            residual_term
            - settings["retention_weight"] * retention
            + settings["retention_gap_weight"] * deficit
        )

    def near_miss_examples(self, vector, rows, count):
        if count <= 0 or not self.residual_mask.any():
            return []
        scores = self.train @ vector
        candidates = np.flatnonzero(self.residual_mask & (scores < self.threshold))
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
        return [
            {"train_index": int(i), "text": rows[int(i)]["text"], "cosine": float(scores[int(i)])}
            for i in ordered
        ]

    def residual_focus_groups(self, rows, center_count, example_count, pool_size):
        """Choose diverse Train-only uncovered regions near the continuous guide."""
        eligible = np.flatnonzero(self.residual_mask & self.guide_mask)
        if not len(eligible):
            eligible = np.flatnonzero(self.residual_mask)
        if not len(eligible):
            return []
        guide_scores = self.train[eligible] @ self.target
        order = np.argsort(-guide_scores, kind="stable")
        pool = eligible[order[:pool_size]]
        centers = [int(pool[0])]
        pool_vectors = self.train[pool]
        while len(centers) < min(center_count, len(pool)):
            nearest = (pool_vectors @ self.train[centers].T).max(axis=1)
            novelty = 1.0 - nearest
            novelty[np.isin(pool, centers)] = -np.inf
            centers.append(int(pool[int(np.argmax(novelty))]))

        groups = []
        for center in centers:
            category = rows[center].get("category")
            same_category = ([int(i) for i in eligible
                              if category is not None and rows[int(i)].get("category") == category])
            neighbors = np.asarray(same_category if same_category else eligible, dtype=int)
            similarities = self.train[neighbors] @ self.train[center]
            ranked = neighbors[np.argsort(-similarities, kind="stable")]
            examples = [int(i) for i in ranked[:example_count]]
            if len(examples) < example_count:
                all_scores = self.train[eligible] @ self.train[center]
                for index in eligible[np.argsort(-all_scores, kind="stable")]:
                    index = int(index)
                    if index not in examples:
                        examples.append(index)
                    if len(examples) == example_count:
                        break
            groups.append([
                {"train_index": index, "text": rows[index]["text"],
                 "category": rows[index].get("category")}
                for index in examples
            ])
        return groups


def ranking(row, method):
    raw_ppl = row.get("ppl")
    ppl = float(raw_ppl) if raw_ppl is not None else 0.0
    if method == "gradient_alignment":
        return row["target_cosine"], row["marginal_gain"], row["union_covered"], -ppl
    if method == "gradient_coverage":
        return row["train_covered"], row["marginal_gain"], row["target_cosine"], -ppl
    if method == "gradient_residual":
        return row["marginal_gain"], row["union_covered"], row["residual_soft_coverage"], row["target_cosine"], -ppl
    if method == "gradient_residual_retention":
        return (
            row["marginal_gain"], row["guide_retained"], row["residual_soft_coverage"],
            -row["guide_margin_deficit"], row["target_cosine"], -ppl,
        )
    raise ValueError(f"Unknown objective: {method}")
