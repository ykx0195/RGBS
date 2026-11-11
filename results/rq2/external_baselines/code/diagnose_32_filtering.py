"""CPU-only diagnosis of frozen raw selections; no new search or API calls."""
import os
os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import sys
import json
from collections import Counter
import torch
ROOT = Path('/root/autodl-tmp/vldb_supplement_20261102')
sys.path.insert(0, str(ROOT / 'code'))
from hybrid_search import Fluency
from run_configurable import AREAS

torch.set_num_threads(8)
out = ROOT / 'diagnose_32_filtering'
out.mkdir(exist_ok=True)
work = []
for group in ['gaslite_flu_32', 'gaslite_32']:
    for p in sorted((ROOT / 'runs' / group).glob('**/final_report.json')):
        report = json.loads(p.read_text())
        wanted = {r['text'] for r in report['raw_selected']}
        origins = {t: [] for t in wanted}
        for slot in sorted(p.parent.glob('slot_*.json')):
            d = json.loads(slot.read_text())
            for row in d['archive']:
                if row['text'] in wanted:
                    origins[row['text']].append({'slot': d['slot'], 'seed': d['seed']})
        work.append((group, p, report, origins))
flu = Fluency(Path('/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct'), 'cpu')
rows = []
summaries = []
for group, p, report, origins in work:
    ds = report['config']['dataset']
    subset = []
    screen = {r['text'] for r in report['screened_selected']}
    for t in [x['text'] for x in report['raw_selected']]:
        ppl = flu.ppl(t)
        oo = origins[t]
        assert oo, 'Raw selection not found in candidate archive'
        for origin in oo:
            origin['seed_ppl'] = flu.ppl(origin['seed'])
            origin['ppl_limit'] = min(150., 1.5 * origin['seed_ppl'])
        ppl_pass = any(ppl <= x['ppl_limit'] for x in oo)
        readable = bool(flu.judge_readability([t], AREAS[ds])[0]) if ppl_pass else None
        category = 'fails_ppl' if not ppl_pass else ('fails_local_readability' if not readable else 'passes_local_quality')
        row = {'method': group, 'dataset': ds, 'text': t, 'ppl': ppl,
               'origins': oo, 'ppl_pass_any_origin': ppl_pass,
               'readability_if_ppl_pass': readable, 'category': category,
               'in_final_screened_set': t in screen}
        rows.append(row)
        subset.append(row)
    q = report['quality_screen']
    summary = {'method': group, 'dataset': ds, 'raw_selected_count': len(subset),
               'cpu_raw_diagnosis': dict(Counter(x['category'] for x in subset)),
               'original_screen_counts': {k: sum(x[k] for x in q) for k in ['archive', 'considered', 'ppl_and_local_eligible', 'two_judge_accepted']}}
    summaries.append(summary)
    print(json.dumps(summary), flush=True)
    (out / 'partial_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
result = {'protocol': 'CPU float32 recomputation on frozen raw selected texts; no API calls or reoptimization. Test not used. For multiple origin seeds, PPL pass means at least one origin passes. Not an exact reconstruction of shortlist membership, early stopping or original GPU judgments.',
          'summaries': summaries, 'rows': rows}
(out / 'final_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('DONE', len(rows), flush=True)
