"""Sentence-level beam search with HotFlip proposals around frozen Riemann targets.

Search and portfolio selection use Train only. Validation and Test are read only
after all 15 targets have finished, for reporting. No clustering is recomputed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from portfolio import (greedy_portfolio, hit_mask, rank_candidate,
                       residual_without, select_beam, swap_portfolio, union_mask)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN_SHA256 = "24ed385f7d6c37814ca197b02bf81cc0a42ae474da26b7d2f16f498787d194df"
TARGET_SHA256 = "5909ca95756abff48e4bd2e0b993c92e50340b9ca9499cb38640eab544af42e8"
PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def texts_from_jsonl(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def target_for_slots(selected: list[dict]) -> list[int]:
    """Use each frozen target once; v10's duplicate round 14 fills missing 7."""
    assigned: list[int | None] = [None] * len(selected)
    unused = set(range(15))
    for slot, row in enumerate(selected):
        match = re.search(r"kmeans_riemann_(\d+)$", row.get("source_round", ""))
        index = int(match.group(1)) - 1 if match else -1
        if index in unused:
            assigned[slot] = index
            unused.remove(index)
    if len(unused) != assigned.count(None):
        raise ValueError("Cannot map 15 v10 sentences to 15 frozen targets")
    fill = iter(sorted(unused))
    return [int(next(fill) if value is None else value) for value in assigned]


class BGE:
    def __init__(self, model_dir: Path, device: str, encoder_kind: str = "bge"):
        if encoder_kind not in {"bge", "e5", "gte", "minilm"}:
            raise ValueError(f"Unknown encoder: {encoder_kind}")
        self.encoder_kind = encoder_kind
        self.prefix = "query: " if encoder_kind == "e5" else ""
        pooling_config = json.loads((model_dir / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
        self.pool_cls = bool(pooling_config.get("pooling_mode_cls_token"))
        self.pool_mean = bool(pooling_config.get("pooling_mode_mean_tokens"))
        if self.pool_cls == self.pool_mean:
            raise ValueError("Expected exactly one CLS or mean pooling mode")
        sequence_config = json.loads((model_dir / "sentence_bert_config.json").read_text(encoding="utf-8"))
        self.max_seq_length = int(sequence_config["max_seq_length"])
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for exact text-span edits")
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embeddings = self.model.get_input_embeddings()
        self.weight = self.embeddings.weight.detach()
        tokens = self.tokenizer.convert_ids_to_tokens(list(range(self.weight.shape[0])))
        specials = set(self.tokenizer.all_special_ids)
        self.starts = torch.tensor([
            index not in specials and not token.startswith("[unused") and token.isascii()
            and token.isalpha() for index, token in enumerate(tokens)
        ], dtype=torch.bool, device=self.device)
        self.suffixes = torch.tensor([
            index not in specials and token.startswith("##") and token[2:].isascii()
            and token[2:].isalpha() for index, token in enumerate(tokens)
        ], dtype=torch.bool, device=self.device)
        self.tokens = tokens

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool_cls:
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            features = self.tokenizer([self.prefix + text for text in texts[start:start + batch_size]], padding=True,
                                      truncation=True, max_length=self.max_seq_length,
                                      return_tensors="pt").to(self.device)
            with torch.inference_mode():
                pooled = self._pool(self.model(**features).last_hidden_state,
                                    features["attention_mask"])
                vectors.append(F.normalize(pooled, p=2, dim=-1).float().cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)

    def hotflip_proposals(self, text: str, focus: np.ndarray, target: np.ndarray,
                          threshold: float, temperature: float, align_weight: float,
                          positions_limit: int, tokens_per_position: int,
                          total_limit: int) -> list[dict]:
        features = self.tokenizer(self.prefix + text, return_offsets_mapping=True, truncation=True,
                                  max_length=self.max_seq_length, return_tensors="pt")
        offsets = features.pop("offset_mapping")[0].tolist()
        features = features.to(self.device)
        ids = features["input_ids"]
        embedded = self.embeddings(ids).detach().requires_grad_(True)
        kwargs = {"inputs_embeds": embedded,
                  "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            kwargs["token_type_ids"] = features["token_type_ids"]
        hidden = self.model(**kwargs).last_hidden_state
        vector = F.normalize(self._pool(hidden, features["attention_mask"]), p=2, dim=-1)[0]
        focus_tensor = torch.as_tensor(focus, dtype=vector.dtype, device=self.device)
        target_tensor = torch.as_tensor(target, dtype=vector.dtype, device=self.device)
        objective = torch.sigmoid((focus_tensor @ vector - threshold) / temperature).mean()
        objective += align_weight * (target_tensor @ vector)
        gradient = torch.autograd.grad(objective, embedded)[0][0]
        if not torch.isfinite(gradient).all():
            raise ValueError("Non-finite HotFlip gradient")
        protected = [match.span() for match in PLACEHOLDER.finditer(text)]
        ranked_positions = []
        for position, (start, end) in enumerate(offsets):
            if start < len(self.prefix):
                continue
            start -= len(self.prefix)
            end -= len(self.prefix)
            if end <= start or any(start < right and end > left for left, right in protected):
                continue
            old_id = int(ids[0, position])
            old_token = self.tokens[old_id]
            if not (old_token.isalpha() or old_token.startswith("##") and old_token[2:].isalpha()):
                continue
            allowed = self.suffixes if old_token.startswith("##") else self.starts
            scores = self.weight @ gradient[position]
            scores = scores.masked_fill(~allowed, float("-inf"))
            scores[old_id] = float("-inf")
            values, replacements = torch.topk(scores, k=min(tokens_per_position, int(allowed.sum())))
            for value, replacement in zip(values.tolist(), replacements.tolist()):
                new_token = self.tokens[replacement].removeprefix("##")
                candidate = text[:start] + new_token + text[end:]
                if candidate != text:
                    ranked_positions.append({
                        "text": candidate, "position": position,
                        "old": text[start:end], "new": new_token,
                        "gradient_delta": float(value - (self.weight[old_id] @ gradient[position]).item()),
                    })
        ranked_positions.sort(key=lambda row: row["gradient_delta"], reverse=True)
        positions = set()
        proposals = []
        seen = set()
        for row in ranked_positions:
            if row["position"] not in positions and len(positions) >= positions_limit:
                continue
            positions.add(row["position"])
            if row["text"] not in seen:
                proposals.append(row)
                seen.add(row["text"])
            if len(proposals) >= total_limit:
                break
        return proposals


class Fluency:
    def __init__(self, model_dir: Path, device: str):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                       trust_remote_code=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True,
            torch_dtype=dtype).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cache: dict[str, float] = {}
        self.readability_cache: dict[tuple[str, str], bool] = {}

    def judge_readability(self, texts: list[str], area: str) -> list[bool]:
        """Local Qwen screen for one coherent request in the chosen dataset scope."""
        decisions = []
        for text in texts:
            key = (area, text)
            if key not in self.readability_cache:
                instruction = (
                    "Decide whether this is one coherent, grammatical English customer question "
                    f"plausible in at least one of these service areas: {area}. "
                    "Reject questions outside all listed areas, gibberish, unrelated keyword lists, "
                    "contradictory requests, and implausible placeholder roles. "
                    "Reply YES or NO only.\nQuestion: " + text
                )
                prompt = (self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}], tokenize=False,
                    add_generation_prompt=True) if hasattr(self.tokenizer, "apply_chat_template")
                    else instruction)
                features = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                          max_length=768).to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(**features, do_sample=False, max_new_tokens=3,
                                                 pad_token_id=self.tokenizer.eos_token_id)
                answer = self.tokenizer.decode(output[0, features["input_ids"].shape[1]:],
                                                skip_special_tokens=True).strip().casefold()
                self.readability_cache[key] = answer.startswith("yes")
            decisions.append(self.readability_cache[key])
        return decisions

    def ppl(self, text: str) -> float:
        if text in self.cache:
            return self.cache[text]
        normalized = PLACEHOLDER.sub(lambda m: m.group()[1:-1].lower().replace("_", " "), text)
        ids = self.tokenizer(normalized, return_tensors="pt", truncation=True,
                             max_length=192)["input_ids"].to(self.device)
        if ids.shape[1] < 2:
            return float("inf")
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits
            loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:])
        value = math.exp(min(float(loss), 20.0))
        self.cache[text] = value
        return value

    def repair(self, text: str, placeholders: list[str]) -> str:
        instruction = (
            "Rewrite the following as one fluent, natural English question. "
            "It may change banking intent. Keep every typed placeholder exactly as written. "
            "Do not add a second question or commentary.\n"
            f"Required placeholders: {', '.join(placeholders) or 'none'}\n"
            f"Text: {text}"
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], tokenize=False,
                add_generation_prompt=True)
        else:
            prompt = instruction
        features = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=768).to(self.device)
        with torch.inference_mode():
            result = self.model.generate(**features, do_sample=False, max_new_tokens=96,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(result[0, features["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip().strip('"')


def valid_text(text: str, seed: str) -> bool:
    if not text or Counter(PLACEHOLDER.findall(text)) != Counter(PLACEHOLDER.findall(seed)):
        return False
    if not 5 <= len(WORD.findall(text)) <= 60:
        return False
    visible = [char for char in text if not char.isspace()]
    return bool(visible) and sum(char.isalpha() for char in visible) / len(visible) >= 0.65


def focus_indices(train: np.ndarray, current: np.ndarray, target: np.ndarray,
                  residual: np.ndarray, threshold: float, limit: int) -> np.ndarray:
    current_scores = train @ current
    guide = train @ target >= threshold
    primary = np.flatnonzero(residual & guide & (current_scores < threshold))
    primary = primary[np.argsort(-current_scores[primary], kind="stable")]
    outside = np.flatnonzero(residual & ~guide & (current_scores < threshold))
    outside = outside[np.argsort(-current_scores[outside], kind="stable")]
    result = np.concatenate([primary[:limit * 3 // 4], outside[:limit // 4]])
    if not len(result):
        pool = np.flatnonzero(residual)
        result = pool[np.argsort(-current_scores[pool], kind="stable")[:limit]]
    return result


def evaluate_rows(texts: list[str], encoder: BGE, train: np.ndarray,
                  target: np.ndarray, threshold: float, fluency: Fluency | None,
                  seed: str, max_ppl: float, relative_ppl: float,
                  batch_size: int, residual: np.ndarray | None = None,
                  guide: np.ndarray | None = None, temperature: float = 0.02,
                  ppl_prefilter: int = 20) -> list[dict]:
    unique = list(dict.fromkeys(text for text in texts if valid_text(text, seed)))
    if not unique:
        return []
    vectors = encoder.encode(unique, batch_size)
    rows = []
    for text, vector in zip(unique, vectors):
        scores = train @ vector
        rows.append({"text": text, "vector": vector, "scores": scores,
                     "mask": scores >= threshold, "target_cosine": float(vector @ target),
                     "ppl": None})
    if fluency:
        if residual is not None and guide is not None and len(rows) > ppl_prefilter:
            rows = sorted(rows, key=lambda row: rank_candidate(
                row, residual, guide, threshold, temperature), reverse=True)[:ppl_prefilter]
        seed_ppl = fluency.ppl(seed)
        feasible = []
        for row in rows:
            row["ppl"] = fluency.ppl(row["text"])
            if row["text"] == seed or (np.isfinite(row["ppl"]) and
                row["ppl"] <= max_ppl and row["ppl"] <= seed_ppl * relative_ppl):
                feasible.append(row)
        rows = feasible
    return rows


def serialize_row(row: dict, **extra) -> dict:
    return {"text": row["text"], "target_cosine": row["target_cosine"],
            "train_covered": int(row["mask"].sum()), "ppl": row.get("ppl"), **extra}


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.jsonl"
    targets_path = args.freeze_dir / "frozen_riemann_targets.npy"
    if sha256(train_path) != TRAIN_SHA256 or sha256(targets_path) != TARGET_SHA256:
        raise ValueError("Train split or frozen Riemann targets differ from v10's verified inputs")
    selected = json.loads(args.seeds_file.read_text(encoding="utf-8"))["selected"]
    if len(selected) != 15:
        raise ValueError("Expected v10's 15 selected sentences")
    targets = np.load(targets_path)
    records = [json.loads(line) for line in (args.freeze_dir / "frozen_riemann_targets.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    if targets.shape != (15, 384) or len(records) != 15:
        raise ValueError("Expected 15 frozen BGE vectors and target records")
    slot_targets = target_for_slots(selected)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BGE(args.model_dir, device)
    fluency = Fluency(args.lm_dir, device) if args.lm_dir else None
    if fluency is None and not args.allow_no_lm:
        raise ValueError("Supply --lm-dir for PPL and repair, or explicitly use --allow-no-lm")
    all_train = texts_from_jsonl(train_path)
    train_texts = all_train[:args.smoke_train_limit] if args.smoke_train_limit else all_train
    cache = args.out_dir / "train_vectors.npy"
    if cache.exists():
        train = np.load(cache)
        if train.shape != (len(train_texts), 384):
            raise ValueError("Cached Train vectors do not match this run")
    else:
        train = encoder.encode(train_texts, args.batch_size)
        np.save(cache, train)
    settings = {
        "train_sha256": TRAIN_SHA256, "target_sha256": TARGET_SHA256,
        "seeds_sha256": sha256(args.seeds_file), "model_dir": str(args.model_dir),
        "lm_dir": str(args.lm_dir) if args.lm_dir else None,
        "smoke_train_limit": args.smoke_train_limit, "threshold": args.threshold,
        "steps_per_slot": args.steps_per_slot, "beam_width": args.beam_width,
        "candidates_per_parent": args.candidates_per_parent,
        "positions_per_parent": args.positions_per_parent,
        "tokens_per_position": args.tokens_per_position,
        "max_total_edits": args.max_total_edits, "repair_every": args.repair_every,
        "extra_seeds": args.extra_seeds, "focus_size": args.focus_size,
        "temperature": args.temperature, "align_weight": args.align_weight,
        "swap_passes": args.swap_passes,
        "ppl_prefilter": args.ppl_prefilter,
        "seed": args.seed, "max_ppl": args.max_ppl, "relative_ppl": args.relative_ppl,
    }
    settings_path = args.out_dir / "config.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Output directory belongs to a different search configuration")
    save_json(settings_path, settings)
    original_texts = [row["text"] for row in selected]
    original_vectors = encoder.encode(original_texts, args.batch_size)
    portfolio = []
    archive = {}
    for slot, (text, vector) in enumerate(zip(original_texts, original_vectors)):
        scores = train @ vector
        row = {"text": text, "vector": vector, "scores": scores,
               "mask": scores >= args.threshold,
               "target_cosine": float(vector @ targets[slot_targets[slot]]),
               "ppl": fluency.ppl(text) if fluency else None, "edits": 0,
               "target_index": slot_targets[slot]}
        portfolio.append(row)
        archive[text] = row
    baseline_covered = int(union_mask([row["mask"] for row in portfolio]).sum())
    print(f"v10 baseline={baseline_covered}/{len(train)}; device={device}; PPL={fluency is not None}", flush=True)
    state_path = args.out_dir / "state.json"
    start_slot = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != settings:
            raise ValueError("Checkpoint configuration changed")
        start_slot = state["completed_slots"]
        restorable = list(dict.fromkeys(state["archive_texts"] + state["portfolio_texts"]))
        restored = encoder.encode(restorable, args.batch_size)
        for text, vector in zip(restorable, restored):
            scores = train @ vector
            target_index = int(state["archive_target_indexes"].get(text, 0))
            archive[text] = {"text": text, "vector": vector, "scores": scores,
                             "mask": scores >= args.threshold,
                             "target_cosine": float(vector @ targets[target_index]),
                             "ppl": fluency.ppl(text) if fluency else None, "edits": 0,
                             "target_index": target_index}
        portfolio = [archive[text] for text in state["portfolio_texts"]]
        print(f"resuming after slot {start_slot}", flush=True)
    end_slot = min(args.max_slots, 15)
    for slot in range(start_slot, end_slot):
        target_index = slot_targets[slot]
        target = targets[target_index]
        guide = train @ target >= args.threshold
        residual = residual_without([row["mask"] for row in portfolio], slot)
        seed_text = portfolio[slot]["text"]
        # Original target examples widen the sentence beam; no same-intent rule is used.
        seed_options = [seed_text] + [item["text"] for item in records[target_index]["seeds"][:args.extra_seeds]]
        initial = evaluate_rows(seed_options, encoder, train, target, args.threshold,
                                fluency, seed_text, args.max_ppl, args.relative_ppl,
                                args.batch_size)
        for row in initial:
            row["edits"] = 0
            row["target_index"] = target_index
            archive.setdefault(row["text"], row)
        beam = select_beam(initial, residual, guide, args.threshold,
                           args.temperature, args.beam_width)
        seen = set(row["text"] for row in beam)
        trace = []
        for step in range(1, args.steps_per_slot + 1):
            proposals = []
            for parent in beam:
                if parent["edits"] >= args.max_total_edits:
                    continue
                focus = focus_indices(train, parent["vector"], target, residual,
                                      args.threshold, args.focus_size)
                if not len(focus):
                    continue
                for candidate in encoder.hotflip_proposals(
                    parent["text"], train[focus], target, args.threshold,
                    args.temperature, args.align_weight, args.positions_per_parent,
                    args.tokens_per_position, args.candidates_per_parent
                ):
                    text = candidate["text"]
                    if text not in seen and valid_text(text, seed_text):
                        proposals.append((text, parent["edits"] + 1, "hotflip"))
                        seen.add(text)
            if fluency and args.repair_every and step % args.repair_every == 0 and beam:
                parent = beam[0]
                repaired = fluency.repair(parent["text"], PLACEHOLDER.findall(seed_text))
                if repaired not in seen and valid_text(repaired, seed_text):
                    proposals.append((repaired, parent["edits"], "lm_repair"))
                    seen.add(repaired)
            rows = evaluate_rows([text for text, _, _ in proposals], encoder, train,
                                 target, args.threshold, fluency, seed_text,
                                 args.max_ppl, args.relative_ppl, args.batch_size,
                                 residual, guide, args.temperature, args.ppl_prefilter)
            metadata = {text: (edits, origin) for text, edits, origin in proposals}
            for row in rows:
                row["edits"], row["origin"] = metadata[row["text"]]
                row["target_index"] = target_index
                archive.setdefault(row["text"], row)
            beam = select_beam(beam + rows, residual, guide, args.threshold,
                               args.temperature, args.beam_width)
            best = max(beam, key=lambda row: rank_candidate(
                row, residual, guide, args.threshold, args.temperature))
            trace.append({"step": step, "proposals": len(proposals),
                          "passed_gate": len(rows), "beam": [row["text"] for row in beam],
                          "best_gain": int((best["mask"] & residual).sum())})
            print(f"slot={slot + 1:02d}/15 target={target_index + 1:02d} "
                  f"step={step} passed={len(rows)} gain={trace[-1]['best_gain']}", flush=True)
            if not rows and all(row["edits"] >= args.max_total_edits for row in beam):
                break
        best = max(beam, key=lambda row: rank_candidate(
            row, residual, guide, args.threshold, args.temperature))
        portfolio[slot] = best
        save_json(args.out_dir / f"slot_{slot + 1:02d}.json", {
            "slot": slot + 1, "target_index": target_index,
            "starting_text": seed_text, "selected": serialize_row(best), "trace": trace,
        })
        save_json(state_path, {"config": settings, "completed_slots": slot + 1,
                               "portfolio_texts": [row["text"] for row in portfolio],
                               "archive_texts": list(archive),
                               "archive_target_indexes": {
                                   text: row["target_index"] for text, row in archive.items()}})
    if end_slot < 15:
        print("Partial/smoke run: Validation and Test were not loaded.", flush=True)
        return
    pool = list(archive.values())
    greedy = greedy_portfolio(pool, 15)
    greedy = swap_portfolio(greedy, pool, args.swap_passes)
    alternatives = [
        ("v10_baseline", [archive[text] for text in original_texts]),
        ("coordinate_beam", portfolio),
        ("global_greedy_swap", greedy),
    ]
    name, final = max(alternatives, key=lambda pair: int(union_mask(
        [row["mask"] for row in pair[1]]).sum()))
    final_texts = [row["text"] for row in final]
    final_vectors = encoder.encode(final_texts, args.batch_size)
    metrics = {}
    for split in ("train", "validation", "test"):
        vectors = train if split == "train" else encoder.encode(
            texts_from_jsonl(args.data_dir / f"{split}.jsonl"), args.batch_size)
        similarity = vectors @ final_vectors.T
        covered = int((similarity.max(axis=1) >= args.threshold).sum())
        metrics[split] = {"covered": covered, "total": len(vectors),
                          "coverage": covered / len(vectors)}
    save_json(args.out_dir / "report.json", {
        "config": settings, "portfolio_source": name,
        "baseline_train_covered": baseline_covered,
        "selected": [serialize_row(row) for row in final], "metrics": metrics,
        "note": "BGE cosine-threshold union coverage; not cache Top-1 or wrong-answer reuse. "
                "Train alone selected texts; Validation/Test were evaluated after selection. "
                "No same-intent constraint. PPL gate enabled only if lm_dir is set.",
    })
    for split, result in metrics.items():
        print(f"{split}: {result['covered']}/{result['total']} "
              f"({result['coverage']:.2%})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/banking77/desensitized")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "bge-small-en-v1.5")
    parser.add_argument("--freeze-dir", type=Path, default=ROOT / "kmeans_riemann_hotflip_aggd")
    parser.add_argument("--seeds-file", type=Path, default=ROOT / "v10_kmeans_riemann_beam/best_k15_selection.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results/full")
    parser.add_argument("--lm-dir", type=Path, help="Local causal LM used for PPL and periodic fluency repair")
    parser.add_argument("--allow-no-lm", action="store_true", help="Ablation only: disable PPL and repair")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-slots", type=int, default=15)
    parser.add_argument("--smoke-train-limit", type=int, default=0)
    parser.add_argument("--steps-per-slot", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--candidates-per-parent", type=int, default=12)
    parser.add_argument("--positions-per-parent", type=int, default=8)
    parser.add_argument("--tokens-per-position", type=int, default=4)
    parser.add_argument("--max-total-edits", type=int, default=3)
    parser.add_argument("--extra-seeds", type=int, default=2)
    parser.add_argument("--focus-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--align-weight", type=float, default=0.10)
    parser.add_argument("--max-ppl", type=float, default=150.0)
    parser.add_argument("--relative-ppl", type=float, default=1.5)
    parser.add_argument("--ppl-prefilter", type=int, default=20)
    parser.add_argument("--repair-every", type=int, default=3)
    parser.add_argument("--swap-passes", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_slots <= 15 or args.threshold <= 0 or args.temperature <= 0:
        parser.error("Invalid slot count or threshold/temperature")
    if args.smoke_train_limit and args.max_slots == 15:
        parser.error("A truncated Train set is only for a partial smoke run")
    run(args)


if __name__ == "__main__":
    main()
