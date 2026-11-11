"""Verify physical stores, sequential traces, strict pairs, and Test returned answers."""
import hashlib,json,sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'outputs';done=read(out/'completion.json');assert done['complete']
summary=read(out/'summary.json');assert len(summary)==6
protocol=read(out/'protocol.json');assert protocol['threshold']==.85
assert sha(ROOT/'run_sequential.py')==protocol['script_sha256']
assert sha(ROOT/'protocol.md')==protocol['protocol_sha256']
checks=[]
for s in summary:
    ds=s['dataset'];p=out/ds/s['condition'];freeze=read(ROOT/'inputs'/ds/'train_frozen_selection.json')
    selected=np.load(ROOT/'inputs'/ds/'selected_vectors.npy');n=len(selected)
    for i in range(n):
        for j in range(i):
            assert float(np.dot(selected[i],selected[j]))<.85
            x,y=selected[i].astype(np.float64),selected[j].astype(np.float64)
            assert float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))<.85
    events=read(p/'admission_events.json');entries=read(p/'entries.json')
    assert len(events)==n==s['submitted_candidates']
    assert [e['expected_text'] for e in events]==freeze['texts']
    admitted=[]
    for i,e in enumerate(events):
        assert e['representation_equal'] and e['encoder_input']==freeze['texts'][i]
        assert e['hit'] != e['written']
        if e['written']:
            record=e['provider_record'];rpath=ROOT/record['record_file'];response=read(rpath)
            assert sha(rpath)==record['record_sha256']
            assert response['request']['messages'][-1]['content']==e['encoder_input']
            assert response['response']['choices'][0]['finish_reason']=='stop'
            assert response['answer']==e['returned_answer']
            admitted.append(e['encoder_input'])
        else:
            assert e['selected_text'] in admitted and e['provider_record'] is None
    assert admitted==[e['text'] for e in entries]
    with sqlite3.connect(p/'cache.sqlite') as c:
        stored=[r[0] for r in c.execute('SELECT question FROM gptcache_question ORDER BY id').fetchall()]
    assert stored==admitted
    assert len(admitted)==s['admitted_entries']==s['rows_after_test']
    matrices=np.load(p/'similarities.npz');mask=(matrices['actual'].max(axis=1)>=.85)
    rr=[json.loads(l) for l in (p/'query_results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert mask.tolist()==[r['cache_hit'] for r in rr]==[r['oracle_hit'] for r in rr]
    assert int(mask.sum())==s['actual_hits']==s['offline_expected_hits']
    assert all(r['text']==r['encoder_input'] for r in rr)
    assert all(not r['cache_hit'] or r['returned_answer'] in {e['answer'] for e in entries} for r in rr)
    checks.append({'dataset':ds,'condition':s['condition'],'entries':len(admitted),'early_hits':sum(e['hit'] for e in events),
      'actual_hits':int(mask.sum()),'queries':len(rr),'physical_questions_match':True,'passed':True})
result={'passed':True,'conditions':checks,'verified_queries':sum(c['queries'] for c in checks),
        'all_candidates_admitted':all(c['entries']=={'banking77':15,'abcd':5,'dogo':10}[c['dataset']] for c in checks),
        'strict_pair_constraint_verified':True,'actual_return_coverage_matches_offline':True,'verify_script_sha256':sha(Path(__file__))}
(out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
