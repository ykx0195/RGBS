"""Configurable Train-only KMeans-Riemann + HotFlip dual-API beam search.

APIs propose rewrites and independently screen dataset scope; coverage scoring
uses Train embeddings and fluency scoring uses a local model. The final
portfolio is frozen before Test is opened; Test coverage is the final result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

import numpy as np
import torch

from hybrid_search import (BGE, Fluency, HERE, ROOT, PLACEHOLDER, focus_indices, save_json, serialize_row,
                           sha256, texts_from_jsonl, valid_text)
from portfolio import greedy_portfolio, residual_without, swap_portfolio, union_mask
from target_generation import prepare


THRESHOLDS = (0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9)
DATASETS = {"abcd": 5, "banking77": 15, "dogo": 10}
MODEL_DIRS = {"bge": "bge-small-en-v1.5", "e5": "e5-small-v2",
              "gte": "gte-small", "minilm": "all-MiniLM-L6-v2"}
AREAS = {
    "abcd": "account access, manage account, order issue, product defect, purchase dispute, shipping issue, subscription inquiry, or site troubleshooting",
    "banking77": "banking support",
    "dogo": "airline, fast food, finance, insurance, media, or software support",
}


WORD_OR_PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>|[A-Za-z]+(?:'[A-Za-z]+)?")


@contextmanager
def api_deadline(seconds: int):
    """Bound a synchronous API call on the Linux main thread."""
    if not hasattr(signal, "SIGALRM"):
        # The OpenAI client's HTTP timeout remains active on Windows.
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise TimeoutError(f"API call exceeded {seconds} seconds")
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def typed_placeholder_context_ok(text: str) -> bool:
    """Reject fluent-looking rewrites that relabel PIN or ZIP placeholders."""
    for match in PLACEHOLDER.finditer(text):
        prefix = text[max(0, match.start() - 40):match.start()]
        if match.group().startswith("<PIN_NUMBER_"):
            if not re.search(r"\bpin(?:\s+(?:number|code))?\s*$", prefix, re.I):
                return False
        elif match.group().startswith("<ZIP_CODE_"):
            if not re.search(r"\b(?:zip(?:\s+code)?|postal\s+code)\s*$", prefix, re.I):
                return False
    return True


def surface_fluency_ok(text: str) -> bool:
    """Exclude recurring awkward fee clauses that the language-model PPL misses."""
    if re.search(r"\bit\s+as\s+(?:an?\s+)?(?:option|alternative|favor|courtesy|possibility)\b", text, re.I):
        return False
    if "shipping" in text.casefold() and re.search(
        r"\bwondering\s+if\s+I\s+(?:could|might)\s+(?:reverse|change|alter|fix|dispute)\s+it\b", text, re.I
    ):
        return False
    return True


def word_change_count(before: str, after: str) -> int:
    """Count replaced, inserted and removed whole words, ignoring punctuation."""
    left = [m.group().casefold() for m in WORD_OR_PLACEHOLDER.finditer(before)]
    right = [m.group().casefold() for m in WORD_OR_PLACEHOLDER.finditer(after)]
    edits = SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
    return sum(max(a1 - a0, b1 - b0) for tag, a0, a1, b0, b1 in edits
               if tag != "equal")


def near_miss(scores: np.ndarray, residual: np.ndarray, threshold: float,
              temperature: float) -> float:
    values = scores[residual & (scores < threshold)]
    if not len(values):
        return 0.0
    z = np.clip((values - threshold) / temperature, -30, 0)
    return float((1 / (1 + np.exp(-z))).sum())


def rank(row: dict, residual: np.ndarray, threshold: float,
         temperature: float) -> tuple:
    """First maximize replacement's portfolio Train union, then near misses."""
    return (int((row["mask"] & residual).sum()),
            near_miss(row["scores"], residual, threshold, temperature),
            -float(row["ppl"] if row.get("ppl") is not None else 0))


def select(rows: list[dict], residual: np.ndarray, threshold: float,
           temperature: float, width: int) -> list[dict]:
    unique = {}
    for row in rows:
        unique.setdefault(" ".join(row["text"].casefold().split()), row)
    return sorted(unique.values(), key=lambda row: rank(row, residual, threshold,
                  temperature), reverse=True)[:width]


def hotflip_hints(encoder: BGE, text: str, focus: np.ndarray, target: np.ndarray,
                  args: argparse.Namespace) -> list[dict]:
    """Turn WordPiece gradients into word-level locations and directions."""
    proposals = encoder.hotflip_proposals(
        text, focus, target, args.threshold, args.temperature, args.align_weight,
        args.positions_per_parent, args.tokens_per_position,
        args.positions_per_parent * args.tokens_per_position * 4)
    words = list(WORD_OR_PLACEHOLDER.finditer(text))
    hints = {}
    for item in proposals:
        candidate = item["text"]
        start = 0
        while start < min(len(text), len(candidate)) and text[start] == candidate[start]:
            start += 1
        left, right = len(text), len(candidate)
        while left > start and right > start and text[left - 1] == candidate[right - 1]:
            left -= 1
            right -= 1
        end = left
        match = next((m for m in words if m.start() <= start and end <= m.end()
                      and not m.group().startswith("<")), None)
        if match is None:
            continue
        span = match.span()
        candidate_word = candidate[span[0]:span[1] + len(candidate) - len(text)]
        if not candidate_word.isalpha():
            continue
        record = hints.setdefault(span, {"word": match.group(), "span": list(span),
                                         "directions": [], "score": float("-inf")})
        if candidate_word.casefold() != match.group().casefold() and candidate_word not in record["directions"]:
            record["directions"].append(candidate_word)
        record["score"] = max(record["score"], item["gradient_delta"])
    ordered = sorted(hints.values(), key=lambda item: item["score"], reverse=True)
    return ordered[:args.positions_per_parent]


def load_provider_specs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = data.get("language", {}).get("api_generators") if "language" in data else data.get("api_generators")
    if not isinstance(specs, list) or len(specs) < 2:
        raise ValueError("API configuration must contain at least two api_generators")
    names = set()
    for item in specs:
        if not all(item.get(key) for key in ("name", "model", "base_url")):
            raise ValueError("Each API needs name, model and base_url")
        if item["name"] in names:
            raise ValueError("API names must be unique")
        names.add(item["name"])
    return specs


class APIRewriter:
    def __init__(self, spec: dict, config_dir: Path, cache_dir: Path):
        self.name = spec["name"]
        self.model = spec["model"]
        self.extra_body = spec.get("extra_body", {})
        self.cache_dir = cache_dir / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = next((os.environ[name].strip() for name in spec.get("key_envs", [])
                    if os.environ.get(name, "").strip()), "")
        if not key and spec.get("key_file"):
            key_path = Path(spec["key_file"])
            if not key_path.is_absolute():
                key_path = config_dir / key_path
            if key_path.is_file():
                key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"Missing API credential for {self.name}")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=spec["base_url"],
                             timeout=60, max_retries=0)

    def propose(self, prompt: str, count: int, max_tokens: int) -> list[str]:
        request = {"model": self.model, "prompt": prompt, "count": count,
                   "max_tokens": max_tokens, "extra_body": self.extra_body}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["texts"]
        last_response_shape = "empty"
        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.8, max_tokens=max_tokens,
                response_format={"type": "json_object"}, extra_body=self.extra_body)
            content = response.choices[0].message.content or ""
            try:
                body = json.loads(content)
                values = body.get("texts", []) if isinstance(body, dict) else []
            except (TypeError, ValueError):
                values = []
            texts = []
            for value in values if isinstance(values, list) else []:
                if isinstance(value, str) and value.strip() and value.strip() not in texts:
                    texts.append(value.strip())
            texts = texts[:count]
            if texts:
                save_json(path, {"request": request, "texts": texts})
                return texts
        reason = response.choices[0].finish_reason
        raise ValueError(f"{self.name} returned no usable JSON sentence proposals "
                         f"after two attempts (finish_reason={reason}, "
                         f"content_prefix={content[:300]!r})")

    def judge(self, texts: list[str], dataset: str) -> list[bool]:
        """Fail closed unless this API accepts each sentence in the dataset scope."""
        if not texts:
            return []
        prompt = (
            "For every numbered sentence, judge whether it is one fluent, plausible, "
            f"standalone English customer request in this dataset scope: {AREAS[dataset]}. "
            "The original intent may differ. Reject cross-domain requests, keyword lists, "
            "gibberish, contradictory requests and implausible roles for IDs or placeholders. "
            "Return only JSON with a boolean array named accept, in the original order.\n"
            + json.dumps(texts, ensure_ascii=False)
        )
        request = {"model": self.model, "judge_prompt": prompt}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"judge_{key}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))["accept"]
            if len(result) == len(texts) and all(type(value) is bool for value in result):
                return result
        last_response_shape = "empty"
        for attempt in range(3):
            instruction = prompt if attempt == 0 else (
                prompt + '\nReturn exactly one JSON object of the form '
                '{"accept": [true, false]}; include one boolean for each sentence.'
            )
            for retry in range(5):
                try:
                    with api_deadline(90):
                        response = self.client.chat.completions.create(
                            model=self.model, messages=[{"role": "user", "content": instruction}],
                            temperature=0, max_tokens=max(128, len(texts) * 12),
                            response_format={"type": "json_object"}, extra_body=self.extra_body)
                    break
                except Exception as exc:
                    if getattr(exc, "status_code", None) != 429 or retry == 4:
                        raise
                    delay = 2 ** (retry + 1)
                    print(f"API {self.name} throttled (429); retrying in {delay}s", flush=True)
                    time.sleep(delay)
            content = response.choices[0].message.content or ""
            try:
                body = json.loads(content)
            except (ValueError, TypeError):
                last_response_shape = f"non-JSON, finish={response.choices[0].finish_reason}"
                continue
            # Some compatible endpoints ignore json_object and return the array itself.
            result = body.get("accept") if isinstance(body, dict) else body
            if (isinstance(result, list) and len(result) == len(texts)
                    and all(type(value) is bool for value in result)):
                save_json(path, {"request": request, "accept": result})
                return result
            last_response_shape = (f"{type(body).__name__}/{type(result).__name__}, "
                                   f"length={len(result) if isinstance(result, list) else 'na'}, "
                                   f"finish={response.choices[0].finish_reason}")
        # A missing verdict cannot count as an approval from both APIs.
        print(f"API {self.name} invalid domain verdicts after three tries "
              f"({last_response_shape}); rejected {len(texts)} candidates", flush=True)
        return [False] * len(texts)


def prompt_for(parent: dict, hints: list[dict], examples: list[str],
               feedback: list[dict], args: argparse.Namespace) -> str:
    directions = [{"word": h["word"], "character_span": h["span"],
                   "possible_direction_words": h["directions"]} for h in hints]
    evidence = [{"before": row["before"], "after": row["after"],
                 "measured_new_Train_hits": row["new_hits"]} for row in feedback]
    typed_note = ("Keep <PIN_NUMBER_1> labeled as a PIN and <ZIP_CODE_1> labeled as a ZIP or postal code.\n"
                  if "<PIN_NUMBER_" in parent["text"] or "<ZIP_CODE_" in parent["text"] else "")
    style_note = ("Use natural customer support English. The company can reverse or waive a shipping fee; "
                  "the customer can cancel an order. Do not end with 'it as a courtesy/favor/option'.\n"
                  if "I'm wondering" in parent["text"] and "shipping" in parent["text"] else "")
    return (
        f"Propose {args.rewrites_per_api} distinct fluent English sentences as local rewrites. "
        'Return only a JSON object with a "texts" array of strings.\n'
        "The HotFlip words are suggestions about where and in what direction to edit; "
        "you may choose a different natural word at those locations. "
        "{typed_note}"
        "{style_note}"
        "The sentence must be a single coherent customer request within: {areas}. "
        "Change at most {limit} words in each proposal. Preserve every typed placeholder "
        "exactly as it appears in the current sentence. The original intent need not be "
        "preserved. Do not copy the Train examples wholesale. Do not judge whether a "
        "proposal succeeds; local {encoder} and perplexity scoring will do that.\n"
        "Current sentence: {sentence}\n"
        "HotFlip word locations and candidate directions: {hints}\n"
        "Train queries not covered by the current portfolio but covered by the "
        "frozen target vector: {examples}\n"
        "Last round's rewrites that measurably increased Train union coverage: {feedback}\n"
    ).format(limit=args.max_words_per_step, areas=AREAS[args.dataset], encoder=args.encoder.upper(), typed_note=typed_note, style_note=style_note, sentence=json.dumps(parent["text"], ensure_ascii=False),
             hints=json.dumps(directions, ensure_ascii=False),
             examples=json.dumps(examples, ensure_ascii=False),
             feedback=json.dumps(evidence, ensure_ascii=False))


def score_rewrites(proposals: list[dict], encoder: BGE, fluency: Fluency,
                   providers: list[APIRewriter],
                   train: np.ndarray, target: np.ndarray, residual: np.ndarray,
                   seed: str, args: argparse.Namespace) -> list[dict]:
    filtered = []
    seen = set()
    for item in proposals:
        text = item["text"].strip()
        identity = " ".join(text.casefold().split())
        step_changes = word_change_count(item["parent"]["text"], text)
        if (identity in seen or text == item["parent"]["text"] or
            not valid_text(text, seed) or not typed_placeholder_context_ok(text) or
            not surface_fluency_ok(text) or
            not 1 <= step_changes <= args.max_words_per_step or
            word_change_count(seed, text) > args.max_total_word_changes):
            continue
        seen.add(identity)
        filtered.append(item)
    if not filtered:
        return []
    # No API score enters this calculation. Every surviving string is encoded afresh.
    vectors = encoder.encode([item["text"] for item in filtered], args.batch_size)
    rows = []
    for item, vector in zip(filtered, vectors):
        scores = train @ vector
        rows.append({"text": item["text"], "vector": vector, "scores": scores,
                     "mask": scores >= args.threshold,
                     "target_cosine": float(vector @ target), "ppl": None,
                     "parent_text": item["parent"]["text"],
                     "provider": item["provider"]})
    # Rank by measured Train coverage before spending LM work on PPL.
    rows = select(rows, residual, args.threshold, args.temperature,
                  args.ppl_prefilter)
    seed_ppl = fluency.ppl(seed)
    accepted = []
    for row in rows:
        row["ppl"] = fluency.ppl(row["text"])
        if (np.isfinite(row["ppl"]) and row["ppl"] <= args.max_ppl and
            row["ppl"] <= seed_ppl * args.relative_ppl):
            accepted.append(row)
    if accepted:
        readable = fluency.judge_readability([row["text"] for row in accepted],
                                             AREAS[args.dataset])
        accepted = [row for row, ok in zip(accepted, readable) if ok]
    if accepted:
        texts = [row["text"] for row in accepted]
        judgments = [provider.judge(texts, args.dataset) for provider in providers]
        accepted = [row for i, row in enumerate(accepted)
                    if all(verdicts[i] for verdicts in judgments)]
    return accepted


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.jsonl"
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    representation = "delexed" if args.representation == "desensitized" else "materialized"
    if manifest.get("representation") != representation or args.dataset.replace("dogo", "multidogo") not in manifest.get("dataset", "").casefold():
        raise ValueError("Dataset directory does not match selected dataset/representation")
    train_hash = sha256(train_path)
    if train_hash != manifest["files"]["train.jsonl"]["sha256"]:
        raise ValueError("Train checksum differs from dataset manifest")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    encoder = BGE(args.model_dir, device, args.encoder)
    train_texts = texts_from_jsonl(train_path)
    if len(train_texts) != manifest["counts"]["train"]:
        raise ValueError("Train row count differs from dataset manifest")
    if args.smoke_train_limit:
        train_texts = train_texts[:args.smoke_train_limit]
    model_hash = sha256(args.model_dir / "model.safetensors")
    identity = {"dataset": args.dataset, "representation": representation,
                "train_sha256": train_hash, "train_rows": len(train_texts),
                "encoder": args.encoder, "model_sha256": model_hash,
                "pooling_sha256": sha256(args.model_dir / "1_Pooling" / "config.json"),
                "tokenizer_sha256": sha256(args.model_dir / "tokenizer.json"),
                "sentence_config_sha256": sha256(args.model_dir / "sentence_bert_config.json"),
                "threshold": args.threshold, "k": args.k, "seed": args.seed}
    cache = args.out_dir / "train_vectors.npy"
    cache_meta = args.out_dir / "train_vectors.json"
    if cache.exists() or cache_meta.exists():
        if not cache.exists() or not cache_meta.exists() or json.loads(cache_meta.read_text(encoding="utf-8")) != identity:
            raise ValueError("Train vector cache differs from selected data/model/config")
        train = np.load(cache)
        if train.shape != (len(train_texts), encoder.weight.shape[1]):
            raise ValueError("Cached Train vectors have the wrong shape")
    else:
        train = encoder.encode(train_texts, args.batch_size)
        np.save(cache, train)
        save_json(cache_meta, identity)
    targets, selected = prepare(train, train_texts, threshold=args.threshold,
                                k=args.k, seed=args.seed, device=device,
                                out_dir=args.out_dir / "targets", identity=identity)
    target_path = args.out_dir / "targets" / "target_vectors.npy"
    seeds_path = args.out_dir / "targets" / "seed_selection.json"
    target_hash = sha256(target_path)
    if targets.shape != (args.k, train.shape[1]) or len(selected) != args.k:
        raise ValueError("Target/seed shape differs from chosen K or encoder")
    if args.prepare_only:
        print(f"Prepared {args.k} KMeans→Riemann Train targets in {args.out_dir / 'targets'}; "
              "Validation/Test not loaded", flush=True)
        return
    slot_targets = [int(row["guide_index"]) for row in selected]
    if sorted(slot_targets) != list(range(args.k)):
        raise ValueError("Each frozen target must have one starting sentence")
    specs = load_provider_specs(args.api_config)
    config = {"dataset": args.dataset, "representation": representation,
              "encoder": args.encoder, "k": args.k, "model_sha256": model_hash,
              "allowed_domains": AREAS[args.dataset], "local_readability_judge": True,
              "api_http_timeout_s": 60, "api_hard_timeout_s": 90,
              "api_max_consecutive_failures": 3,
              "domain_judges": [spec["name"] for spec in specs],
              "train_sha256": train_hash, "target_sha256": target_hash,
              "seeds_sha256": sha256(seeds_path), "api_config_sha256": sha256(args.api_config),
              "model_dir": str(args.model_dir), "lm_dir": str(args.lm_dir),
              "threshold": args.threshold, "smoke_train_limit": args.smoke_train_limit,
              "steps_per_slot": args.steps_per_slot, "beam_width": args.beam_width,
              "positions_per_parent": args.positions_per_parent,
              "tokens_per_position": args.tokens_per_position,
              "max_words_per_step": args.max_words_per_step,
              "max_total_word_changes": args.max_total_word_changes,
              "rewrites_per_api": args.rewrites_per_api,
              "prompt_examples": args.prompt_examples, "feedback_examples": args.feedback_examples,
              "focus_size": args.focus_size, "temperature": args.temperature,
              "align_weight": args.align_weight, "max_ppl": args.max_ppl,
              "relative_ppl": args.relative_ppl, "ppl_prefilter": args.ppl_prefilter,
              "swap_passes": args.swap_passes, "batch_size": args.batch_size,
              "seed": args.seed, "selection_mode": "global_greedy_swap"}
    config_path = args.out_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory has a different search configuration")
    save_json(config_path, config)
    fluency = Fluency(args.lm_dir, device)
    providers = [APIRewriter(spec, args.api_config.parent, args.out_dir / "api_cache")
                 for spec in specs]
    original_texts = [row["text"] for row in selected]
    portfolio = []
    archive = {}
    for slot, (text, vector) in enumerate(zip(original_texts,
             encoder.encode(original_texts, args.batch_size))):
        scores = train @ vector
        row = {"text": text, "vector": vector, "scores": scores,
               "mask": scores >= args.threshold,
               "target_cosine": float(vector @ targets[slot_targets[slot]]),
               "ppl": fluency.ppl(text), "target_index": slot_targets[slot]}
        portfolio.append(row)
        archive[text] = row
    baseline = int(union_mask([row["mask"] for row in portfolio]).sum())
    state_path = args.out_dir / "state.json"
    start_slot = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise ValueError("Checkpoint configuration differs")
        start_slot = state["completed_slots"]
        archive = {}
        texts = list(dict.fromkeys(state["archive_texts"] + state["portfolio_texts"]))
        for text, vector in zip(texts, encoder.encode(texts, args.batch_size)):
            scores = train @ vector
            index = state["archive_target_indexes"][text]
            archive[text] = {"text": text, "vector": vector, "scores": scores,
                             "mask": scores >= args.threshold,
                             "target_cosine": float(vector @ targets[index]),
                             "ppl": fluency.ppl(text), "target_index": index}
        portfolio = [archive[text] for text in state["portfolio_texts"]]
    print(f"baseline={baseline}/{len(train)}; device={device}; APIs={len(providers)}; resume={start_slot}", flush=True)
    for slot in range(start_slot, args.max_slots):
        target_index = slot_targets[slot]
        target = targets[target_index]
        guide = train @ target >= args.threshold
        masks = [row["mask"] for row in portfolio]
        residual = residual_without(masks, slot)
        seed = portfolio[slot]
        baseline_gain = int((seed["mask"] & residual).sum())
        beam = [seed]
        seen = {seed["text"]}
        feedback = []
        trace = []
        for step in range(1, args.steps_per_slot + 1):
            proposals = []
            attempted_calls = 0
            successful_providers = set()
            consecutive_api_failures = {provider.name: 0 for provider in providers}
            for parent in beam:
                focus = focus_indices(train, parent["vector"], target, residual,
                                      args.threshold, args.focus_size)
                if not len(focus):
                    continue
                hints = hotflip_hints(encoder, parent["text"], train[focus], target, args)
                if not hints:
                    continue
                example_ids = np.flatnonzero(residual & ~parent["mask"] & guide)
                example_ids = example_ids[np.argsort(-(train[example_ids] @ target),
                                                       kind="stable")]
                examples = [train_texts[i] for i in example_ids[:args.prompt_examples]]
                prompt = prompt_for(parent, hints, examples, feedback, args)
                for provider in providers:
                    attempted_calls += 1
                    try:
                        with api_deadline(90):
                            texts = provider.propose(prompt, args.rewrites_per_api,
                                                     args.rewrites_per_api * args.max_new_tokens + 80)
                    except Exception as exc:
                        print(f"API {provider.name} failed: {type(exc).__name__}: {exc}", flush=True)
                        consecutive_api_failures[provider.name] += 1
                        if consecutive_api_failures[provider.name] >= 3:
                            raise RuntimeError(f"{provider.name} failed three consecutive calls at "
                                               f"slot {slot + 1}, step {step}; checkpoint retained") from exc
                        continue
                    consecutive_api_failures[provider.name] = 0
                    successful_providers.add(provider.name)
                    for text in texts:
                        if text not in seen:
                            proposals.append({"text": text, "parent": parent,
                                              "provider": provider.name})
                            seen.add(text)
            if attempted_calls and len(successful_providers) < 2:
                raise RuntimeError(f"Fewer than two APIs returned proposals at slot {slot + 1}, step {step}")
            rows = score_rewrites(proposals, encoder, fluency, providers, train, target,
                                  residual, seed["text"], args)
            for row in rows:
                row["target_index"] = target_index
                archive.setdefault(row["text"], row)
            by_text = {row["text"]: row for row in beam}
            positive = []
            for row in rows:
                parent = by_text.get(row["parent_text"])
                if parent is None:
                    continue
                increase = int((row["mask"] & residual).sum()) - int((parent["mask"] & residual).sum())
                if increase > 0:
                    positive.append({"before": parent["text"], "after": row["text"],
                                     "new_hits": increase})
            feedback = sorted(positive, key=lambda item: item["new_hits"], reverse=True)[:args.feedback_examples]
            beam = select(beam + rows, residual, args.threshold, args.temperature,
                          args.beam_width)
            best = beam[0]
            current_gain = int((best["mask"] & residual).sum())
            trace.append({"step": step, "api_proposals": len(proposals),
                          "passed_ppl": len(rows), "best_replacement_gain": current_gain,
                          "net_union_change": current_gain - baseline_gain,
                          "beam": [row["text"] for row in beam],
                          "accepted_texts": [row["text"] for row in rows],
                          "positive_feedback": feedback})
            print(f"slot={slot + 1:02d} step={step} proposals={len(proposals)} "
                  f"accepted={len(rows)} net={current_gain - baseline_gain:+d}", flush=True)
            if not proposals:
                break
        best = beam[0]
        portfolio[slot] = best
        save_json(args.out_dir / f"slot_{slot + 1:02d}.json", {
            "slot": slot + 1, "target_index": target_index,
            "starting_text": seed["text"], "selected": serialize_row(best),
            "baseline_replacement_gain": baseline_gain, "trace": trace})
        save_json(state_path, {"config": config, "completed_slots": slot + 1,
                               "portfolio_texts": [row["text"] for row in portfolio],
                               "archive_texts": list(archive),
                               "archive_target_indexes": {text: row["target_index"]
                                                          for text, row in archive.items()}})
    if args.max_slots < args.k:
        print("Partial run: Validation/Test were not loaded.", flush=True)
        return
    pool = list(archive.values())
    greedy = swap_portfolio(greedy_portfolio(pool, args.k), pool, args.swap_passes)
    name, final = "global_greedy_swap", greedy
    final_vectors = encoder.encode([row["text"] for row in final], args.batch_size)
    covered = int(((train @ final_vectors.T).max(axis=1) >= args.threshold).sum())
    train_metrics = {"covered": covered, "total": len(train),
                     "coverage": covered / len(train)}
    frozen = {
        "config": config, "portfolio_source": name, "baseline_train_covered": baseline,
        "selected": [serialize_row(row) for row in final], "metrics": {"train": train_metrics},
        "heldout_loaded": False,
        "note": "Selection used Train only. Test is loaded after this portfolio is frozen."}
    freeze_path = args.out_dir / "train_report.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Existing frozen Train selection differs; use a new output directory")
    save_json(freeze_path, frozen)

    # This is the first access to Test. It cannot affect target generation or selection.
    test_path = args.data_dir / "test.jsonl"
    test_hash = sha256(test_path)
    if test_hash != manifest["files"]["test.jsonl"]["sha256"]:
        raise ValueError("Test checksum differs from dataset manifest")
    with test_path.open(encoding="utf-8") as handle:
        test_rows = [json.loads(line) for line in handle if line.strip()]
    if (len(test_rows) != manifest["counts"]["test"] or
        any(row.get("split") != "test" or row.get("representation") != representation
            for row in test_rows)):
        raise ValueError("Test row count, split, or representation differs from manifest")
    test_vectors = encoder.encode([row["text"] for row in test_rows], args.batch_size)
    test_covered = int(((test_vectors @ final_vectors.T).max(axis=1) >= args.threshold).sum())
    test_metrics = {"covered": test_covered, "total": len(test_rows),
                    "coverage": test_covered / len(test_rows)}
    save_json(args.out_dir / "final_report.json", {
        **frozen, "metrics": {"train": train_metrics, "test": test_metrics},
        "final_coverage_split": "test", "test_sha256": test_hash,
        "heldout_loaded": True,
        "note": "Final coverage is Test cosine-threshold union coverage. Train was used "
                "for optimization and selection; Test was opened only after freezing. "
                "This is not cache Top-1 or attack success."})
    print(f"final Test: {test_covered}/{len(test_rows)} "
          f"({test_metrics['coverage']:.2%}); Train: {covered}/{len(train)} "
          f"({train_metrics['coverage']:.2%})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True,
                        help="dogo is MultiDoGO")
    parser.add_argument("--representation", choices=["desensitized", "non_desensitized"], required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_DIRS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Desktop" / "ykx" / "data")
    parser.add_argument("--models-root", type=Path, default=Path.home() / "Desktop" / "ykx")
    parser.add_argument("--data-dir", type=Path, help="Override DATA_ROOT/dataset/representation")
    parser.add_argument("--model-dir", type=Path, help="Override MODELS_ROOT/model name")
    parser.add_argument("--k", type=int, help="Default: ABCD 5, Banking77 15, MultiDoGO 10")
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--lm-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Generate Train vectors, Riemann targets and natural seeds; no API/PPL search")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, choices=THRESHOLDS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--smoke-train-limit", type=int, default=0)
    parser.add_argument("--steps-per-slot", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--positions-per-parent", type=int, default=8)
    parser.add_argument("--tokens-per-position", type=int, default=4)
    parser.add_argument("--rewrites-per-api", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-words-per-step", type=int, default=3)
    parser.add_argument("--max-total-word-changes", type=int, default=10)
    parser.add_argument("--prompt-examples", type=int, default=6)
    parser.add_argument("--feedback-examples", type=int, default=4)
    parser.add_argument("--focus-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--align-weight", type=float, default=0.10)
    parser.add_argument("--max-ppl", type=float, default=150.0)
    parser.add_argument("--relative-ppl", type=float, default=1.5)
    parser.add_argument("--ppl-prefilter", type=int, default=20)
    parser.add_argument("--swap-passes", type=int, default=4)
    args = parser.parse_args()
    args.k = DATASETS[args.dataset] if args.k is None else args.k
    args.max_slots = args.k if args.max_slots is None else args.max_slots
    args.data_dir = args.data_dir or args.data_root / args.dataset.replace("dogo", "multidogo") / args.representation
    args.model_dir = args.model_dir or args.models_root / MODEL_DIRS[args.encoder]
    if not 1 <= args.max_slots <= args.k or args.k < 1 or args.temperature <= 0 or args.smoke_train_limit < 0:
        parser.error("Invalid slot count or threshold/temperature")
    if args.smoke_train_limit and args.max_slots == args.k and not args.prepare_only:
        parser.error("Truncated Train requires a partial smoke run")
    if not args.prepare_only and (args.api_config is None or args.lm_dir is None):
        parser.error("Search requires --api-config and --lm-dir")
    if not args.data_dir.is_dir() or not args.model_dir.is_dir():
        parser.error("Selected data or encoder model directory does not exist")
    for name in ("beam_width", "positions_per_parent", "tokens_per_position", "rewrites_per_api",
                 "max_words_per_step", "max_total_word_changes", "ppl_prefilter"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    run(args)


if __name__ == "__main__":
    main()

