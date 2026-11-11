"""Task adaptation of AdvDec retrieval beam search for 30 natural customer questions.

The official repository is preserved separately. This file replaces its hardcoded
Llama chat format / Contriever / MS MARCO paths with Qwen, BGE and frozen Train
clusters while retaining token-by-token top-k beam expansion and cosine plus
soft readability scoring. Test is never opened here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path('/root/autodl-tmp/advdec_baseline_20261031')
HOTFLIP = Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031')
sys.path.insert(0, str(HOTFLIP/'code'))
from hybrid_search import BGE

DATASETS = {'banking77': 15, 'abcd': 5, 'dogo': 10}
AREAS = {'banking77': 'banking support',
         'abcd': 'account access, orders, product defects, shipping, subscriptions, or site support',
         'dogo': 'airline, fast food, finance, insurance, media, or software support'}
PLACEHOLDER = re.compile(r'<[A-Z][A-Z0-9_]*>')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def prompt_for_cluster(dataset, cluster, texts, indices, train_vectors, seed):
    mean = train_vectors[indices].mean(axis=0)
    ranked = indices[np.argsort(-(train_vectors[indices] @ mean), kind='stable')]
    examples = [seed['text']]
    for i in ranked:
        text = texts[int(i)]
        if text not in examples:
            examples.append(text)
        if len(examples) == 3:
            break
    instruction = (
        f'Write exactly one concise, grammatical English customer-support question in {AREAS[dataset]}. '
        'Never output a literal name, ID, number, address, phone number, or email. '
        'Use typed placeholders such as <ACCOUNT_NUMBER_1> instead of specific personal details. '
        'Do not answer the question. Do not add a preface, a list, or a second sentence. '
        'The customer issues to target are illustrated by these Train examples:\n'
        + '\n'.join(f'{n}. {text}' for n, text in enumerate(examples, 1))
        + '\nCustomer question:'
    )
    return instruction, examples, mean


class Decoder:
    def __init__(self, model_dir, encoder_dir, device, proposal_batch, readability_batch):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            low_cpu_mem_usage=False).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.encoder = BGE(encoder_dir, device, 'bge')
        self.proposal_batch = proposal_batch
        self.readability_batch = readability_batch
        yes = self.tokenizer.encode('Yes', add_special_tokens=False)
        no = self.tokenizer.encode('No', add_special_tokens=False)
        assert len(yes) == len(no) == 1, (yes, no)
        self.yes_id, self.no_id = yes[0], no[0]
        self.proposal_calls = 0
        self.readability_calls = 0
        self.embedding_texts = 0

    def propose(self, prefix_ids, beams, top_k):
        candidates = []
        for offset in range(0, len(beams), self.proposal_batch):
            batch = beams[offset:offset+self.proposal_batch]
            ids = torch.tensor([prefix_ids + x['token_ids'] for x in batch],
                               dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=ids, use_cache=False).logits[:, -1, :].float()
            self.proposal_calls += 1
            logits = logits.clone()
            logits[:, self.tokenizer.all_special_ids] = -torch.inf
            vals, token_ids = torch.topk(logits, k=min(top_k*4, logits.shape[1]), dim=-1)
            for j, parent in enumerate(batch):
                count = 0
                for token in token_ids[j].tolist():
                    new_ids = parent['token_ids'] + [token]
                    text = self.tokenizer.decode(new_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False).strip()
                    if (not text or '\n' in text or '\r' in text or '<|' in text or
                            has_literal_numbers_or_malformed_placeholders(text)):
                        continue
                    candidates.append({'token_ids': new_ids, 'text': text})
                    count += 1
                    if count == top_k:
                        break
        return candidates

    def score(self, candidates, cluster_mean, cosine_weight, readability_weight, readability_cap):
        texts = [x['text'] for x in candidates]
        vectors = self.encoder.encode(texts, 32)
        self.embedding_texts += len(texts)
        cosine = vectors @ cluster_mean
        for x, value in zip(candidates, cosine):
            x['mean_train_cosine'] = float(value)
        for offset in range(0, len(candidates), self.readability_batch):
            batch = candidates[offset:offset+self.readability_batch]
            messages = [[{'role': 'user', 'content':
                          f'Is this text unintelligible? "{x["text"]}" Just answer Yes or No.'}]
                        for x in batch]
            prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            features = self.tokenizer(prompts, return_tensors='pt', padding=True,
                                      truncation=True, max_length=256).to(self.device)
            with torch.inference_mode():
                logits = self.model(**features, use_cache=False).logits[:, -1, :].float()
            self.readability_calls += 1
            yes = logits[:, self.yes_id].cpu().numpy()
            no = logits[:, self.no_id].cpu().numpy()
            for x, y, n in zip(batch, yes, no):
                # Match the official retrieval scorer: (No-Yes)/(Yes+No), upper cap 0.05.
                denominator = float(y+n)
                raw = float((n-y)/denominator) if abs(denominator) > 1e-6 else -1e6
                if not math.isfinite(raw):
                    raw = -1e6
                x['readability_score'] = raw
                x['score'] = cosine_weight*x['mean_train_cosine'] + readability_weight*min(raw, readability_cap)
        return candidates


def is_complete(text):
    body = text.strip()
    if len(body.split()) < 5 or body[-1:] not in ('.', '?', '!'):
        return False
    without_placeholders = PLACEHOLDER.sub('PERSON', body)
    return len(re.findall(r'[.?!]', without_placeholders)) == 1 and not has_literal_numbers_or_malformed_placeholders(body)


def has_literal_numbers_or_malformed_placeholders(text):
    cleaned = PLACEHOLDER.sub('', text)
    partial = re.search(r'<[A-Z][A-Z0-9_]*$', cleaned)
    if partial:
        cleaned = cleaned[:partial.start()]
    return bool(re.search(r'\d|[<>]', cleaned))


def run_cluster(decoder, plan, dataset, cluster, smoke=False):
    inputs = HOTFLIP/'inputs'/dataset
    seeds = read(inputs/'seed_selection.json')
    labels = np.load(inputs/'cluster_labels.npy')
    train_vectors = np.load(inputs/'train_vectors.npy')
    with (inputs/'train.jsonl').open(encoding='utf-8') as f:
        train_texts = [json.loads(line)['text'] for line in f if line.strip()]
    assert len(labels) == len(train_vectors) == len(train_texts)
    assert seeds['train_sha256'] == sha(inputs/'train.jsonl')
    assert seeds['model_sha256'] == sha(Path(plan['encoder_dir'])/'model.safetensors')
    indices = np.flatnonzero(labels == cluster)
    assert len(indices) and seeds['selected'][cluster]['cluster_id'] == cluster
    instruction, examples, mean = prompt_for_cluster(dataset, cluster, train_texts, indices,
                                                     train_vectors, seeds['selected'][cluster])
    prefix_ids = decoder.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': instruction}], tokenize=True, add_generation_prompt=True)
    out = ROOT/('smoke' if smoke else 'runs')/dataset/f'cluster_{cluster:02d}'
    out.mkdir(parents=True, exist_ok=True)
    cfg = {'dataset': dataset, 'cluster': cluster, 'seed': plan['seed'],
           'model_repo': plan['generator_model'], 'model_revision': plan['model_revision'],
           'encoder': plan['encoder'], 'official_commit': plan['official_commit'],
           'beam_width': plan['beam_width'] if not smoke else plan['smoke_beam_width'],
           'top_k': plan['top_k'] if not smoke else plan['smoke_top_k'],
           'max_steps': plan['max_steps'] if not smoke else plan['smoke_steps'],
           'cosine_weight': plan['cosine_weight'],
           'readability_weight': plan['readability_weight'],
           'readability_cap': plan['readability_cap'],
           'prompt': instruction, 'train_examples': examples,
           'train_sha256': seeds['train_sha256'],
           'train_vectors_sha256': seeds['train_vectors_sha256'],
           'encoder_weight_sha256': seeds['model_sha256'],
           'model_dir': str(plan['model_dir']), 'device': decoder.device}
    config_path = out/'config.json'
    if config_path.exists():
        assert read(config_path) == cfg
    else:
        save(config_path, cfg)
    final_path = out/'final_cluster.json'
    if final_path.exists():
        assert read(final_path)['config'] == cfg
        return read(final_path)
    state_path = out/'state.json'
    if state_path.exists():
        state = read(state_path)
        assert state['config'] == cfg
        beams = state['beams']
        best_complete = state['best_complete']
        first_step = state['completed_steps']+1
        elapsed_before = state['active_seconds']
        history = state['history']
    else:
        beams = [{'token_ids': [], 'text': '', 'score': 0.0}]
        best_complete = None
        first_step = 1
        elapsed_before = 0.0
        history = []
    started = time.monotonic()
    for step in range(first_step, cfg['max_steps']+1):
        proposed = decoder.propose(prefix_ids, beams, cfg['top_k'])
        if not proposed:
            break
        scored = decoder.score(proposed, mean, cfg['cosine_weight'],
                               cfg['readability_weight'], cfg['readability_cap'])
        scored.sort(key=lambda x: (x['score'], x['mean_train_cosine'], x['text']), reverse=True)
        active = []
        seen = set()
        for candidate in scored:
            signature = candidate['text'].casefold()
            if signature in seen:
                continue
            seen.add(signature)
            if is_complete(candidate['text']):
                if best_complete is None or candidate['score'] > best_complete['score']:
                    best_complete = candidate
            elif len(active) < cfg['beam_width']:
                active.append(candidate)
        beams = active
        history.append({'step': step, 'candidate_count': len(scored),
                        'active_count': len(beams), 'best_complete_score':
                        None if best_complete is None else best_complete['score'],
                        'best_active_text': beams[0]['text'] if beams else None})
        state = {'config': cfg, 'completed_steps': step, 'beams': beams,
                 'best_complete': best_complete, 'active_seconds': elapsed_before+time.monotonic()-started,
                 'history': history}
        save(state_path, state)
        print('ADVDEC_STEP', dataset, cluster, step, cfg['max_steps'],
              len(scored), 'best', repr((best_complete or (beams[0] if beams else {})).get('text','')[:100]), flush=True)
        if not beams:
            break
    chosen = best_complete or (beams[0] if beams else None)
    assert chosen is not None
    result = {'config': cfg, 'selected': chosen, 'completed_steps': state['completed_steps'],
              'complete_sentence': is_complete(chosen['text']),
              'train_cluster_size': len(indices), 'active_seconds': state['active_seconds'],
              'heldout_loaded': False, 'api_calls': 0,
              'proposal_forward_calls_cumulative': decoder.proposal_calls,
              'readability_forward_calls_cumulative': decoder.readability_calls,
              'encoder_texts_cumulative': decoder.embedding_texts,
              'gpu_max_allocated_mib': torch.cuda.max_memory_allocated()/1024/1024 if decoder.device == 'cuda' else None}
    save(final_path, result)
    print('ADVDEC_CLUSTER_COMPLETE', dataset, cluster, repr(chosen['text']), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(DATASETS))
    parser.add_argument('--cluster', type=int)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-small-model', action='store_true')
    args = parser.parse_args()
    plan = read(ROOT/'experiment_plan.json')
    assert sha(Path(__file__)) == plan['code_sha256'], 'Worker source differs from the frozen experiment plan'
    if args.all:
        assert args.dataset is None and args.cluster is None and not args.smoke
    else:
        assert args.dataset in DATASETS and args.cluster is not None
        assert 0 <= args.cluster < DATASETS[args.dataset]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = 'cuda'
    assert torch.cuda.is_available()
    model_dir = Path(plan['smoke_model_dir']) if args.smoke_small_model else Path(plan['model_dir'])
    actual_plan = dict(plan)
    if args.smoke_small_model:
        actual_plan['model_dir'] = str(model_dir)
        actual_plan['generator_model'] = 'Qwen/Qwen2.5-1.5B-Instruct (smoke only)'
        actual_plan['model_revision'] = 'local-smoke-only'
    decoder = Decoder(model_dir, Path(plan['encoder_dir']), device,
                      plan['proposal_batch_size'], plan['readability_batch_size'])
    if args.all:
        for dataset, k in DATASETS.items():
            for cluster in range(k):
                run_cluster(decoder, actual_plan, dataset, cluster, False)
    else:
        run_cluster(decoder, actual_plan, args.dataset, args.cluster, args.smoke)


if __name__ == '__main__':
    main()
