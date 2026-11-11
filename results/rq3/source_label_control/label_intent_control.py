"""Source-label control for paired raw/deidentified semantic-cache matches.

The keys are frozen, existing natural Train questions. Their source labels are
read from the Train JSONL, never assigned to RKGB-generated text. Key source
IDs are selected on deidentified Train only, then both key and Test-query
representations are changed together. This is a label-proxy analysis, not a
human task audit or an answer-reuse measurement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HOME = Path.home()
DATA = HOME / "Desktop/ykx/data"
RESULTS = HOME / "Desktop/results"
BASELINES = RESULTS / "优先补充实验_20261031/baselines"
FIXED = RESULTS / "固定文本_原文脱敏配对_20261031"
MODEL = HOME / "Desktop/ykx/bge-small-en-v1.5"
CODE = RESULTS / "newcode/src"
OUT = RESULTS / "来源标签跨意图配对_20261031"
TAUS = (0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9)
METHODS = ("greedy_train", "greedy_train_swap4", "riemann_nearest", "random_train_seed42")
DATASETS = {
    "banking77": {"folder": "banking77", "fixed": FIXED / "banking77", "labels": ("category",)},
    "abcd": {"folder": "abcd", "fixed": FIXED / "ABCD补充/abcd", "labels": ("category", "flow")},
    "dogo": {"folder": "multidogo", "fixed": FIXED / "dogo", "labels": ("category", "domain")},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_csv(path: Path, items: list[dict]) -> None:
    if not items:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def paired_ci(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    diff = y.astype(np.int8) - x.astype(np.int8)
    rng = np.random.default_rng(seed)
    boot = np.empty(5000, dtype=np.float64)
    for i in range(len(boot)):
        boot[i] = diff[rng.integers(0, len(diff), len(diff))].mean() * 100
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def metrics(sim: np.ndarray, key_labels: np.ndarray, query_labels: np.ndarray, tau: float) -> dict:
    match = sim >= tau
    different = key_labels[None, :] != query_labels[:, None]
    same = ~different
    cross = match & different
    covered = match.any(axis=1)
    cross_covered = cross.any(axis=1)
    all_cross = covered & ~(match & same).any(axis=1)
    return {
        "match": match,
        "cross": cross,
        "covered": covered,
        "cross_covered": cross_covered,
        "all_cross": all_cross,
        "matched_pairs": int(match.sum()),
        "cross_label_pairs": int(cross.sum()),
        "covered_queries": int(covered.sum()),
        "cross_label_queries": int(cross_covered.sum()),
        "all_matches_cross_label_queries": int(all_cross.sum()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(CODE))
    from hybrid_search import BGE

    OUT.mkdir(parents=True, exist_ok=True)
    model_hash = sha(MODEL / "model.safetensors")
    encoder = BGE(MODEL, "cpu", "bge")
    assert encoder.max_seq_length == 512
    summary: list[dict] = []
    per_query: list[dict] = []
    frozen: list[dict] = []
    provenance: dict = {
        "protocol": "Frozen source IDs selected on deidentified Train; natural Train keys and source-aligned Test queries both switch raw/deidentified representation; original source labels are proxy labels, not human judgments of generated RKGB keys.",
        "thresholds": list(TAUS),
        "primary_threshold": 0.85,
        "methods": list(METHODS),
        "model_sha256": model_hash,
        "encoder_code_sha256": sha(CODE / "hybrid_search.py"),
        "pooling_sha256": sha(MODEL / "1_Pooling/config.json"),
        "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
        "max_seq_length": encoder.max_seq_length,
        "dtype": "float32",
        "device": "cpu",
        "bootstrap": "5000 paired Test-query resamples, conditional on frozen keys, source labels, and threshold; excludes label error and search variation",
        "datasets": {},
    }
    for dataset, cfg in DATASETS.items():
        folder = cfg["folder"]
        raw_train_path = DATA / folder / "non_desensitized/train.jsonl"
        delex_train_path = DATA / folder / "desensitized/train.jsonl"
        raw_test_path = DATA / folder / "non_desensitized/test.jsonl"
        delex_test_path = DATA / folder / "desensitized/test.jsonl"
        raw_train, delex_train = rows(raw_train_path), rows(delex_train_path)
        raw_test, delex_test = rows(raw_test_path), rows(delex_test_path)
        for name, a, b in (("Train", raw_train, delex_train), ("Test", raw_test, delex_test)):
            assert len(a) == len(b) and len(a) > 0, (dataset, name)
            ids_a = [(r["source_id"], r["variant_id"]) for r in a]
            ids_b = [(r["source_id"], r["variant_id"]) for r in b]
            assert ids_a == ids_b and len(set(ids_a)) == len(a), (dataset, name, "pair alignment")
            assert all(r["split"].lower() == name.lower() for r in a + b)
            for r, d in zip(a, b):
                assert all(r.get(k) == d.get(k) for k in cfg["labels"]), (dataset, name, "label mismatch")
        assert len({r["source_id"] for r in delex_train} & {r["source_id"] for r in delex_test}) == 0

        identity = json.loads((cfg["fixed"] / "embedding_identity.json").read_text(encoding="utf-8"))
        assert identity["model_sha256"] == model_hash
        assert identity["raw_test_sha256"] == sha(raw_test_path)
        assert identity["delex_test_sha256"] == sha(delex_test_path)
        assert identity["code_sha256"] == provenance["encoder_code_sha256"]
        with np.load(cfg["fixed"] / "embeddings.npz") as z:
            raw_test_vec = z["raw"].copy()
            delex_test_vec = z["delex"].copy()
        assert raw_test_vec.shape == delex_test_vec.shape == (len(raw_test), 384)
        assert np.allclose(np.linalg.norm(raw_test_vec, axis=1), 1, atol=1e-5)
        assert np.allclose(np.linalg.norm(delex_test_vec, axis=1), 1, atol=1e-5)

        selection_path = BASELINES / dataset / "frozen_baseline_selection.json"
        selected = json.loads(selection_path.read_text(encoding="utf-8"))["methods"]
        train_identity = json.loads((BASELINES / dataset / "train_vectors_identity.json").read_text(encoding="utf-8"))
        assert train_identity["train_sha256"] == sha(delex_train_path)
        assert train_identity["model_sha256"] == model_hash
        delex_train_vec = np.load(BASELINES / dataset / "train_vectors.npy", mmap_mode="r")
        assert delex_train_vec.shape == (len(delex_train), 384)
        unique_indices = sorted({int(item["train_index"]) for method in METHODS for item in selected[method]})
        raw_keys = [raw_train[i]["text"] for i in unique_indices]
        delex_keys = [delex_train[i]["text"] for i in unique_indices]
        print(dataset, "encoding", len(unique_indices), "raw and deidentified natural Train keys")
        raw_key_vec = encoder.encode(raw_keys, 32)
        delex_key_check = encoder.encode(delex_keys, 32)
        archived_delex = np.asarray(delex_train_vec[unique_indices], dtype=np.float32)
        max_vector_difference = float(np.max(np.abs(delex_key_check - archived_delex)))
        assert max_vector_difference < 2e-5, (dataset, "cached Train vector mismatch", max_vector_difference)
        vector_index = {train_index: pos for pos, train_index in enumerate(unique_indices)}

        provenance["datasets"][dataset] = {
            "train": len(delex_train), "test": len(delex_test),
            "label_fields": list(cfg["labels"]),
            "input_sha256": {"raw_train": sha(raw_train_path), "delex_train": sha(delex_train_path),
                             "raw_test": sha(raw_test_path), "delex_test": sha(delex_test_path)},
            "selection_sha256": sha(selection_path),
            "test_embedding_sha256": sha(cfg["fixed"] / "embeddings.npz"),
            "encoded_unique_key_sources": len(unique_indices),
            "max_abs_cached_delex_train_vector_difference": max_vector_difference,
        }
        for method in METHODS:
            items = selected[method]
            assert len(items) == {"banking77": 15, "abcd": 5, "dogo": 10}[dataset]
            indices = [int(item["train_index"]) for item in items]
            assert len(set(indices)) == len(indices)
            for item in items:
                idx = int(item["train_index"])
                assert item["source_id"] == delex_train[idx]["source_id"]
                assert item["text"] == delex_train[idx]["text"]
                frozen.append({"dataset": dataset, "method": method, "key_index": len([x for x in frozen if x["dataset"] == dataset and x["method"] == method]),
                               "train_index": idx, "source_id": item["source_id"],
                               "category": delex_train[idx].get("category", ""),
                               "flow": delex_train[idx].get("flow", ""),
                               "domain": delex_train[idx].get("domain", "")})
            positions = [vector_index[i] for i in indices]
            sr = raw_test_vec @ raw_key_vec[positions].T
            sd = delex_test_vec @ archived_delex[positions].T
            for label_field in cfg["labels"]:
                key_labels = np.array([delex_train[i][label_field] for i in indices], dtype=object)
                query_labels = np.array([r[label_field] for r in delex_test], dtype=object)
                assert all(label is not None and str(label) for label in key_labels)
                assert all(label is not None and str(label) for label in query_labels)
                for tau in TAUS:
                    raw = metrics(sr, key_labels, query_labels, tau)
                    delex = metrics(sd, key_labels, query_labels, tau)
                    n = len(query_labels)
                    row = {
                        "dataset": dataset, "method": method, "label_field": label_field,
                        "k": len(indices), "threshold": tau, "test_total": n,
                        "raw_covered": raw["covered_queries"], "delex_covered": delex["covered_queries"],
                        "raw_cross_label_queries": raw["cross_label_queries"],
                        "delex_cross_label_queries": delex["cross_label_queries"],
                        "raw_all_matches_cross_label_queries": raw["all_matches_cross_label_queries"],
                        "delex_all_matches_cross_label_queries": delex["all_matches_cross_label_queries"],
                        "raw_matched_pairs": raw["matched_pairs"], "delex_matched_pairs": delex["matched_pairs"],
                        "raw_cross_label_pairs": raw["cross_label_pairs"],
                        "delex_cross_label_pairs": delex["cross_label_pairs"],
                        "newly_cross_label_queries": int((~raw["cross_covered"] & delex["cross_covered"]).sum()),
                        "lost_cross_label_queries": int((raw["cross_covered"] & ~delex["cross_covered"]).sum()),
                        "cross_label_delta_pp": round(100 * (delex["cross_label_queries"] - raw["cross_label_queries"]) / n, 4),
                        "total_coverage_delta_pp": round(100 * (delex["covered_queries"] - raw["covered_queries"]) / n, 4),
                    }
                    if tau == 0.85:
                        row["cross_label_delta_ci95_pp"] = list(paired_ci(raw["cross_covered"], delex["cross_covered"], 1000 + len(summary)))
                        row["total_coverage_delta_ci95_pp"] = list(paired_ci(raw["covered"], delex["covered"], 2000 + len(summary)))
                        for i in range(n):
                            if method == "greedy_train" and label_field == "category":
                                per_query.append({"dataset": dataset, "test_index": i,
                                                  "source_id": delex_test[i]["source_id"],
                                                  "category": query_labels[i],
                                                  "raw_covered": int(raw["covered"][i]),
                                                  "delex_covered": int(delex["covered"][i]),
                                                  "raw_cross_label": int(raw["cross_covered"][i]),
                                                  "delex_cross_label": int(delex["cross_covered"][i]),
                                                  "raw_cross_label_key_indices": json.dumps(np.flatnonzero(raw["cross"][i]).tolist()),
                                                  "delex_cross_label_key_indices": json.dumps(np.flatnonzero(delex["cross"][i]).tolist())})
                    else:
                        row["cross_label_delta_ci95_pp"] = ""
                        row["total_coverage_delta_ci95_pp"] = ""
                    assert row["newly_cross_label_queries"] - row["lost_cross_label_queries"] == delex["cross_label_queries"] - raw["cross_label_queries"]
                    summary.append(row)
        print(dataset, "complete")
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "greedy_query_level_085.csv", per_query)
    write_csv(OUT / "frozen_key_sources.csv", frozen)
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", OUT)


if __name__ == "__main__":
    main()
