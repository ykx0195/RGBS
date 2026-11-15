"""Offline loader for author's released GPT2/BERT-tokenizer scorer.

Scoring and differentiable loss methods are inherited unchanged. Only loading
uses pinned, verified local assets, and batched scoring uses 16 rows for memory.
"""
from pathlib import Path
import hashlib,json
import torch
from transformers import AutoTokenizer
from src.attacks.fluency_scorer import GPT2FluencyScorer
from src.nanoGPT.model import GPTConfig,GPT

class LocalAuthorFluency(GPT2FluencyScorer):
    def __init__(self,root,attack_tokenizer):
        root=Path(root)
        manifest=json.loads((root/'verified_assets.json').read_text())
        checkpoint=torch.load(root/'ckpt.pt',map_location='cpu',weights_only=True)
        self.model=GPT(GPTConfig(**checkpoint['model_args']))
        state={k.removeprefix('_orig_mod.'):v for k,v in checkpoint['model'].items()}
        self.model.load_state_dict(state,strict=True)
        del state,checkpoint
        official_tokenizer=AutoTokenizer.from_pretrained(str(root/'e5_tokenizer'),local_files_only=True,use_fast=True)
        assert official_tokenizer.get_vocab()==attack_tokenizer.get_vocab(),'Token ID maps differ'
        assert official_tokenizer.all_special_ids==attack_tokenizer.all_special_ids,'Special token IDs differ'
        probes=['A refund has not arrived.','Please check order <ORDER_ID_1>.','Card delivery or refund status?']
        assert official_tokenizer(probes)['input_ids']==attack_tokenizer(probes)['input_ids']
        # The attack proxy restores placeholder spelling; it is token-equivalent.
        # Share the verified tokenizer object to satisfy the official class check.
        self.tokenizer=attack_tokenizer
        self.model=self.model.to('cuda').eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.input_embedding=self.model.transformer.wte
        self.batch_size=16
        self.asset_manifest=manifest
