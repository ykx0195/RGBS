"""Train-only KMeans→Riemann guide and natural seed preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from hybrid_search import save_json, sha256
from riemann_common import Settings, density_weights, greedy_order, optimize_one_guide, unit, validate_train


def anchor_indices(guide: np.ndarray, train: np.ndarray, threshold: float,
                   count: int = 20, retrieval_pool: int = 256,
                   alignment_weight: float = 0.05) -> list[int]:
    """Same diverse natural-anchor selection as v10_current/dataset_pipeline.py."""
    similarity = train @ guide
    pool = np.argsort(-similarity, kind="stable")[:retrieval_pool]
    count = min(count, len(pool))
    hits = train[pool] @ train.T >= threshold
    chosen = [0]
    covered = hits[0].copy()
    while len(chosen) < count:
        index = max((i for i in range(len(pool)) if i not in chosen),
                    key=lambda i: (float((hits[i] & ~covered).mean())
                                   + alignment_weight * float(similarity[pool[i]]),
                                   float(similarity[pool[i]]), -int(pool[i])))
        chosen.append(index)
        covered |= hits[index]
    return [int(pool[i]) for i in chosen]


def prepare(train: np.ndarray, train_texts: list[str], *, threshold: float,
            k: int, seed: int, device: str, out_dir: Path,
            identity: dict) -> tuple[np.ndarray, list[dict]]:
    """Rebuild targets for this exact data, model, threshold, K and seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "target_vectors.npy"
    seeds_path = out_dir / "seed_selection.json"
    manifest_path = out_dir / "target_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("Target directory belongs to different data/model/threshold/K/seed")
        if not target_path.is_file() or not seeds_path.is_file():
            raise ValueError("Incomplete target preparation; inspect output directory")
        if sha256(target_path) != manifest["target_sha256"] or sha256(seeds_path) != manifest["seeds_sha256"]:
            raise ValueError("Prepared target or seed checksum changed")
        return np.load(target_path), json.loads(seeds_path.read_text(encoding="utf-8"))["selected"]
    if any(path.exists() for path in (target_path, seeds_path)):
        raise ValueError("Partial target preparation exists; inspect output directory")
    train = validate_train(train, k)
    if len(train) != len(train_texts):
        raise ValueError("Train text/vector count mismatch")
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = Settings(threshold=threshold)
    weights = density_weights(train, settings.density_k, settings.density_floor)
    fit = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(train)
    centers = unit(fit.cluster_centers_)
    guides, details = [], []
    for cluster in range(k):
        own = fit.labels_ == cluster
        guide, detail = optimize_one_guide(train[own], weights[own], train[own],
                                           settings, torch.device(device),
                                           extra_start=centers[cluster])
        guides.append(guide)
        details.append({"cluster_id": cluster, "size": int(own.sum()), **detail})
    guides = unit(np.stack(guides))
    order, gains = greedy_order(guides, train, threshold)
    ordered = guides[order]
    selected = []
    covered = np.zeros(len(train), dtype=bool)
    used = set()
    used_texts = set()
    for index, guide in enumerate(ordered):
        anchors = anchor_indices(guide, train, threshold)
        available = [i for i in anchors if i not in used and train_texts[i] not in used_texts]
        if not available:
            available = [int(i) for i in np.argsort(-(train @ guide), kind="stable")
                         if int(i) not in used and train_texts[int(i)] not in used_texts][:256]
        if not available:
            raise ValueError("Fewer unique Train texts than requested targets")
        # Choose one natural Train sentence per ordered guide by marginal union gain.
        winner = max(available, key=lambda i: (int(((train @ train[i] >= threshold) & ~covered).sum()),
                                               float(train[i] @ guide), -i))
        used.add(winner)
        used_texts.add(train_texts[winner])
        covered |= train @ train[winner] >= threshold
        selected.append({"guide_index": index, "train_index": winner,
                         "text": train_texts[winner], "anchor_indices": anchors})
    np.save(target_path, ordered)
    save_json(seeds_path, {"selected": selected})
    save_json(manifest_path, {"identity": identity,
                              "method": "kmeans_then_fixed_cluster_riemann",
                              "cluster_order": [int(i) for i in order],
                              "marginal_train_gains": gains, "clusters": details,
                              "target_sha256": sha256(target_path),
                              "seeds_sha256": sha256(seeds_path),
                              "heldout_loaded": False})
    return ordered, selected
