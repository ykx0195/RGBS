import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
root = Path("C:/Users/ykx01/Desktop/results")
historical = root / "ykx现行实验/v10_kmeans_riemann_beam/paper_supplement_20261018"
run = historical / "exports/extracted_15246/v10_method/runs/paper_no_gradient_085"
model_path = Path("C:/Users/ykx01/Desktop/ykx/bge-small-en-v1.5")
output = Path("outputs/Banking77_no_gradient_20x4_test")
output.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = [
    run / "03_portfolio_evaluation/portfolios.json",
    run / "03_portfolio_evaluation/report.json",
    run / "config_snapshot.json",
    historical / "data/test.jsonl",
    historical / "data/validation.jsonl",
    historical / "data/validation.npy",
    root / "固定文本_原文脱敏配对_20261031/banking77/embeddings.npz",
]
hashes_before = {str(path): sha(path) for path in sources}
portfolios = read(sources[0])
assert len(portfolios) == 1
portfolio = portfolios[0]
assert portfolio["budget"] == 15 and portfolio["selected_on"] == "train"
assert portfolio["validation_used_for_selection"] is False
texts = [row["text"] for row in portfolio["selected"]]
report = read(sources[1])
assert report["test_loaded"] is False
expected = report["reports"][0]
assert texts == [row["text"] for row in expected["selected"]]
assert len(texts) == len(set(texts)) == 15
identity = read(root / "固定文本_原文脱敏配对_20261031/banking77/embedding_identity.json")
assert sha(model_path / "model.safetensors") == identity["model_sha256"]
assert sha(model_path / "tokenizer.json") == identity["tokenizer_sha256"]
assert sha(model_path / "1_Pooling/config.json") == identity["pooling_sha256"]
assert hashes_before[str(historical / "data/test.jsonl")] == identity["delex_test_sha256"]
pooling = read(model_path / "1_Pooling/config.json")
assert pooling["pooling_mode_cls_token"] and not pooling["pooling_mode_mean_tokens"]
max_length = read(model_path / "sentence_bert_config.json")["max_seq_length"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()

def encode(batch_texts):
    tokens = tokenizer(batch_texts, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    with torch.inference_mode():
        vectors = model(**tokens).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(vectors, dim=-1).numpy()

keys = encode(texts)
validation_vectors = np.load(historical / "data/validation.npy")
validation_scores = validation_vectors @ keys.T
validation_count = int((validation_scores.max(axis=1) >= 0.85).sum())
print("Validation reproduction", validation_count, "/", len(validation_vectors), flush=True)
assert validation_count == expected["validation"]["covered"] == 605

queries = [json.loads(line) for line in sources[3].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
cached = np.load(sources[-1])
test_vectors = cached["delex"]
assert len(test_vectors) == len(queries) == 1485
sample_indices = np.linspace(0, len(queries)-1, 32, dtype=int)
fresh_sample = encode([queries[i]["text"] for i in sample_indices])
sample_error = float(np.abs(fresh_sample - test_vectors[sample_indices]).max())
print("Cached test embedding sample max error", sample_error, flush=True)
assert sample_error < 1e-5

scores = test_vectors @ keys.T
scores64 = test_vectors.astype(np.float64) @ keys.astype(np.float64).T
covered = scores.max(axis=1) >= 0.85
assert np.array_equal(covered, scores64.max(axis=1) >= 0.85)
count = int(covered.sum())
summary = {
    "historical_run": "paper_no_gradient_085",
    "search_iterations": 20, "beam_width": 4, "k": 15, "threshold": 0.85,
    "selected_on": "train", "frozen_before_current_test_evaluation": True,
    "validation_covered": validation_count, "validation_total": len(validation_vectors),
    "test_covered": count, "test_total": len(queries), "test_coverage_pct": 100 * count/len(queries),
    "metric": "fraction of Test queries with at least one key at cosine >= 0.85",
    "model_sha256": identity["model_sha256"], "tokenizer_sha256": identity["tokenizer_sha256"],
    "pooling": "CLS, L2 normalization", "max_length": max_length,
    "device": "cpu", "dtype": "float32", "prefix": "",
    "cached_test_sample_checked": len(sample_indices), "cached_test_sample_max_abs_error": sample_error,
    "float64_threshold_decisions_agree": True,
    "selected_origin_counts": {"seed": sum(row["origin"] == "seed" for row in portfolio["selected"])},
    "source_hashes": hashes_before,
}
assert all(sha(Path(path)) == digest for path, digest in hashes_before.items())
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(output / "frozen_texts.json").write_text(json.dumps(portfolio, ensure_ascii=False, indent=2), encoding="utf-8")
np.savez_compressed(output / "evaluation_vectors.npz", keys=keys, test=test_vectors, similarities=scores)
with (output / "query_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_index", "source_id", "query", "covered", "top1_key_index", "max_cosine"])
    for i, query in enumerate(queries):
        writer.writerow([i, query.get("source_id", ""), query["text"], int(covered[i]),
                         int(scores[i].argmax()), float(scores[i].max())])
print("Test", count, "/", len(queries), f"= {100*count/len(queries):.6f}%", flush=True)
