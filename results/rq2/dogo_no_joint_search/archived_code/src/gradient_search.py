"""Residual search with reserved natural proposals, quality gates, and diverse beams."""

from __future__ import annotations

import time

from .common import normalized_text
from .encoder import encode_texts
from .objectives import coverage_jaccard, ranking


def _diversity(row, selected):
    if not selected:
        return 1.0
    return min(1.0 - coverage_jaccard(row["_coverage_mask"], other["_coverage_mask"]) for other in selected)


def choose_beams(archive, new_rows, method, settings):
    valid = [row for row in archive if row["language_feasible"]]
    ordered = sorted(valid, key=lambda row: ranking(row, method), reverse=True)
    count, slots = settings["beam_size"], settings["exploration_slots"]
    natural_slots = settings["natural_beam_slots"]
    beams = ordered[: max(1, count - slots - natural_slots)]
    identities = {normalized_text(row["text"]) for row in beams}
    natural_order = ordered
    if settings.get("prefer_rewrite_beam", False):
        natural_order = ([row for row in ordered if row["origin"] == "lm_rewrite"] +
                         [row for row in ordered if row["origin"] != "lm_rewrite"])
    for row in natural_order:
        if natural_slots <= 0 or len(beams) >= count - slots:
            break
        identity = normalized_text(row["text"])
        if row["origin"] in {"seed", "lm_rewrite"} and identity not in identities:
            beams.append(row)
            identities.add(identity)
            natural_slots -= 1
    fresh = [row for row in new_rows if normalized_text(row["text"]) not in identities]
    while len(beams) < count and fresh:
        def exploration_key(row):
            overlaps = [coverage_jaccard(row["_coverage_mask"], other["_coverage_mask"])
                        for other in beams]
            below_cap = not overlaps or max(overlaps) <= settings["max_coverage_jaccard"]
            return below_cap, _diversity(row, beams), ranking(row, method)

        candidate = max(fresh, key=exploration_key)
        beams.append(candidate)
        identities.add(normalized_text(candidate["text"]))
        fresh = [row for row in fresh if normalized_text(row["text"]) not in identities]
    for row in ordered:
        identity = normalized_text(row["text"])
        if len(beams) >= count:
            break
        if identity not in identities:
            beams.append(row)
            identities.add(identity)
    return beams


def language_prefilter(rows, method, settings):
    """Reserve language-gate slots for each generating model."""
    ordered = sorted(rows, key=lambda row: ranking(row, method), reverse=True)
    limit = settings["language_prefilter"]
    rewrites = []
    per_model = settings.get("rewrite_prefilter_per_model", 0)
    if per_model:
        models = dict.fromkeys(row.get("generator_model", "unknown") for row in ordered
                               if row["origin"] == "lm_rewrite")
        for model in models:
            rewrites.extend([row for row in ordered
                             if row["origin"] == "lm_rewrite" and
                             row.get("generator_model", "unknown") == model][:per_model])
    rewrite_ids = {normalized_text(row["text"]) for row in rewrites}
    rewrites.extend(row for row in ordered if row["origin"] == "lm_rewrite"
                    and normalized_text(row["text"]) not in rewrite_ids)
    rewrites = rewrites[:settings["rewrite_prefilter"]]
    selected = list(rewrites)
    selected_ids = {normalized_text(row["text"]) for row in selected}
    for row in ordered:
        identity = normalized_text(row["text"])
        if len(selected) >= limit:
            break
        if identity not in selected_ids:
            selected.append(row)
            selected_ids.add(identity)
    return selected


def search_target(method, objective, seeds, engine, settings, gate, train_rows, rewriter=None,
                  excluded_texts=(), quality_judge=None):
    started = time.perf_counter()
    rewrite_scope = gate.settings.get("rewrite_scope", "anchor_intent")
    cross_intent = rewrite_scope in {"banking_domain", "residual_focus"}
    archive = []
    identities = {normalized_text(text) for text in excluded_texts}
    trace = []
    encoder_texts = gradient_calls = completed_iterations = 0
    rejected_rules = rejected_ppl = rejected_readability = rejected_api_quality = rewrite_calls = 0
    edit_counts = {str(order): 0 for order in settings["edit_orders"]}

    def add(text, vector, origin, iteration, language, **metadata):
        identity = normalized_text(text)
        if identity in identities or not language["language_feasible"]:
            return None
        identities.add(identity)
        mask = objective.coverage_mask(vector)
        row = {
            "text": text, "vector": vector, "_coverage_mask": mask,
            "origin": origin, "iteration": iteration, "token_count": engine.token_count(text),
            **objective.metrics(vector), **language, **metadata,
        }
        archive.append(row)
        return row

    seed_intents = [train_rows[seed["train_index"]].get("category") for seed in seeds]
    seed_languages = gate.evaluate(
        [seed["text"] for seed in seeds], [seed["text"] for seed in seeds], seed_intents,
        skip_readability=[True] * len(seeds),
    )
    for seed, language, intent in zip(seeds, seed_languages, seed_intents):
        add(seed["text"], seed["vector"], "seed", 0, language, train_index=seed["train_index"],
            anchor_text=seed["text"], intent_label=intent, anchor_intent_label=intent)
    if not archive:
        raise ValueError("No seed passes the hard language gate")
    beams, status = choose_beams(archive, archive, method, settings), "completed"
    rewrite_only_next = False
    focus_groups = (
        objective.residual_focus_groups(
            train_rows, settings["residual_focus_centers"],
            gate.settings["rewrite_example_count"], settings["residual_focus_pool"],
        ) if rewriter is not None and rewrite_scope == "residual_focus" else []
    )
    focus_cursor = 0

    for iteration in range(1, settings["iterations"] + 1):
        remaining = settings["max_encoder_texts"] - encoder_texts
        cap = min(settings["max_candidates_per_step"], remaining)
        if cap < 1:
            status = "encoder_budget"
            break
        proposals, proposed_ids = [], set()
        rewrite_round = rewriter is not None and iteration % gate.settings["rewrite_every"] == 0
        gradient_cap = (0 if rewrite_only_next or settings.get("disable_gradient", False) else
                        (max(0, cap - settings["rewrite_reserve"]) if rewrite_round else cap))
        rewrite_only_next = False
        for parent in beams:
            if len(proposals) >= gradient_cap:
                break
            if gradient_calls >= settings["max_gradient_calls"]:
                status = "gradient_budget"
                break
            quota = max(1, gradient_cap // max(1, len(beams)))
            candidates, loss = engine.proposals(
                parent["text"], objective, method, quota, identities | proposed_ids
            )
            gradient_calls += 1
            trace.append({
                "event": "gradient", "iteration": iteration, "parent": parent["text"],
                "loss": loss, "candidate_count": len(candidates),
            })
            for candidate in candidates:
                identity = normalized_text(candidate["text"])
                if identity not in identities and identity not in proposed_ids and len(proposals) < gradient_cap:
                    proposed_ids.add(identity)
                    proposals.append({**candidate, "parent_text": parent["text"], "origin": "gradient",
                                      "anchor_text": parent["anchor_text"],
                                      "intent_label": None if cross_intent else parent.get("intent_label"),
                                      "anchor_intent_label": parent.get("anchor_intent_label")})

        if rewrite_round and len(proposals) < cap:
            for parent in beams[:settings["rewrite_parent_count"]]:
                if rewrite_scope == "residual_focus" and focus_groups:
                    examples = focus_groups[focus_cursor % len(focus_groups)]
                    focus_cursor += 1
                    trace.append({
                        "event": "residual_focus", "iteration": iteration,
                        "focus_train_indices": [row["train_index"] for row in examples],
                        "focus_categories": sorted({str(row.get("category")) for row in examples}),
                    })
                else:
                    examples = objective.near_miss_examples(
                        parent["vector"], train_rows, gate.settings["rewrite_example_count"]
                    )
                for candidate in rewriter.propose(
                    parent["text"], examples, anchor=parent["anchor_text"],
                    intent_label=parent.get("anchor_intent_label", parent.get("intent_label"))
                ):
                    identity = normalized_text(candidate["text"])
                    if identity not in identities and identity not in proposed_ids and len(proposals) < cap:
                        proposed_ids.add(identity)
                        proposals.append({**candidate, "parent_text": parent["anchor_text"],
                                          "anchor_text": parent["anchor_text"],
                                          "intent_label": None if cross_intent else parent.get("intent_label"),
                                          "anchor_intent_label": parent.get("anchor_intent_label")})
                rewrite_calls += 1

        cheap = []
        for candidate in proposals:
            ok, reasons = gate.cheap_check(candidate["text"], candidate["parent_text"])
            if ok:
                cheap.append(candidate)
            else:
                rejected_rules += 1
                trace.append({"event": "rule_reject", "iteration": iteration,
                              "text": candidate["text"], "reasons": reasons})
        if not cheap:
            if (rewriter is not None and iteration < settings["iterations"] and
                    (iteration + 1) % gate.settings["rewrite_every"] == 0):
                trace.append({"event": "await_rewrite", "iteration": iteration,
                              "reason": "no_rule_feasible_candidates"})
                rewrite_only_next = True
                continue
            status = "no_rule_feasible_candidates"
            break

        vectors = encode_texts(
            engine.encoder, [row["text"] for row in cheap], settings["embedding_batch_size"]
        )
        encoder_texts += len(cheap)
        prescored = [
            {**candidate, "vector": vector, **objective.metrics(vector)}
            for candidate, vector in zip(cheap, vectors)
        ]
        prescored = language_prefilter(prescored, method, settings)
        languages = gate.evaluate(
            [row["text"] for row in prescored], [row["parent_text"] for row in prescored],
            [row.get("intent_label") for row in prescored],
        )
        if quality_judge is not None:
            quality_indices = [index for index, language in enumerate(languages)
                               if language["language_feasible"]]
            judgments = quality_judge.judge([prescored[index] for index in quality_indices])
            for index, passed in zip(quality_indices, judgments):
                languages[index]["api_quality_feasible"] = bool(passed)
                if not passed:
                    languages[index]["language_feasible"] = False
                    languages[index]["language_failures"].append("api_quality")
                    rejected_api_quality += 1
        metric_keys = set(objective.metrics(prescored[0]["vector"])) if prescored else set()
        fresh = []
        for candidate, language in zip(prescored, languages):
            if not language["ppl_feasible"]:
                rejected_ppl += 1
            if not language["readability_feasible"]:
                rejected_readability += 1
            metadata = {key: value for key, value in candidate.items()
                        if key not in {"text", "vector", "origin"} and key not in metric_keys}
            row = add(candidate["text"], candidate["vector"], candidate["origin"], iteration,
                      language, **metadata)
            if row is not None:
                fresh.append(row)
                if row.get("edit_order") in settings["edit_orders"]:
                    edit_counts[str(row["edit_order"])] += 1
        if not fresh:
            if (rewriter is not None and iteration < settings["iterations"] and
                    (iteration + 1) % gate.settings["rewrite_every"] == 0):
                trace.append({"event": "await_rewrite", "iteration": iteration,
                              "reason": "no_language_feasible_candidates"})
                rewrite_only_next = True
                continue
            status = "no_language_feasible_candidates"
            break

        beams = choose_beams(archive, fresh, method, settings)
        best = max(archive, key=lambda row: ranking(row, method))
        completed_iterations = iteration
        trace.append({
            "event": "iteration_complete", "iteration": iteration,
            "best_text": best["text"], **objective.metrics(best["vector"]),
            "guide_train_total": objective.guide_total, "encoder_texts": encoder_texts,
            "gradient_calls": gradient_calls, "beam_texts": [row["text"] for row in beams],
        })
        print(
            f"  {method} iteration={iteration}: marginal={best['marginal_gain']}, "
            f"union={best['union_covered']}/{len(objective.train)}, "
            f"retained={best['guide_retained']}/{objective.guide_total}, archive={len(archive)}",
            flush=True,
        )
        if status != "completed":
            break

    best = max(archive, key=lambda row: ranking(row, method))
    return archive, best, {
        "status": status, "completed_iterations": completed_iterations,
        "gradient_calls": gradient_calls, "encoder_texts": encoder_texts,
        "rewrite_calls": rewrite_calls, "rejected_rules": rejected_rules,
        "rejected_ppl": rejected_ppl, "rejected_readability": rejected_readability,
        "rejected_api_quality": rejected_api_quality,
        "search_seconds": time.perf_counter() - started,
        "archive_count": len(archive), "candidate_edit_orders": edit_counts,
        "initial_seed_count": sum(row["origin"] == "seed" for row in archive),
        "ppl_computed": gate.ppl_enabled, "language_gate": True,
        "test_loaded": False, "validation_loaded_in_search": False,
    }, trace
