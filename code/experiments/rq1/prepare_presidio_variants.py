"""Run a pinned Presidio pipeline and audit its span coverage against gold.

Labels differ across corpora. Detection scores are therefore TYPE-AGNOSTIC
character coverage, not entity-type accuracy or an official PII benchmark score.
"""
import argparse
import csv
import importlib.metadata
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, ConflictResolutionStrategy

DOMAINS=('healthcare','finance','banking','insurance','legal')
METHODS=('generic','typed','typed_numbered','delete','star')

def variants(text,spans):
    result={m:[] for m in METHODS}; count=Counter(); cursor=0
    for s in sorted(spans,key=lambda x:(x['start'],x['end'])):
        a,b,label=int(s['start']),int(s['end']),s['label'].upper()
        assert cursor<=a<b<=len(text)
        count[label]+=1
        values={'generic':'[PII]','typed':f'[{label}]','typed_numbered':f'[{label}_{count[label]}]','delete':'','star':'*'}
        for m in METHODS: result[m].extend((text[cursor:a],values[m]))
        cursor=b
    for pieces in result.values(): pieces.append(text[cursor:])
    return {m:''.join(p) for m,p in result.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    nlp=NlpEngineProvider(nlp_configuration={'nlp_engine_name':'spacy','models':[{'lang_code':'en','model_name':'en_core_web_lg'}]}).create_engine()
    analyzer=AnalyzerEngine(nlp_engine=nlp,supported_languages=['en'],default_score_threshold=0.5)
    engine=AnonymizerEngine(); rows=[]; audits=[]
    for domain in DOMAINS:
        records=[json.loads(s) for s in (args.input_dir/f'{domain}.jsonl').open(encoding='utf-8')]
        totals=Counter(); labels=Counter()
        with (args.output_dir/f'{domain}.jsonl').open('w',encoding='utf-8') as out:
            for index,r in enumerate(records):
                text=r['source_text']
                results=analyzer.analyze(text=text,language='en',score_threshold=0.5)
                raw=[{'start':x.start,'end':x.end,'label':x.entity_type,'score':x.score} for x in results]
                # Use Presidio's own conflict handling; keep preserves original offsets.
                kept=engine.anonymize(text=text,analyzer_results=results,
                    operators={'DEFAULT':OperatorConfig('keep')},
                    conflict_resolution=ConflictResolutionStrategy.REMOVE_INTERSECTIONS,
                    merge_entities_with_spaces=False)
                assert kept.text==text
                spans=sorted([{'start':x.start,'end':x.end,'label':x.entity_type} for x in kept.items],key=lambda x:(x['start'],x['end']))
                v=variants(text,spans)
                # Verify shared rendering policies against built-in Presidio operators.
                from presidio_anonymizer.entities import RecognizerResult
                clean=[RecognizerResult(entity_type=s['label'],start=s['start'],end=s['end'],score=1.0) for s in spans]
                for m,value in [('generic','[PII]'),('typed',None),('delete',''),('star','*')]:
                    if m=='typed': ops={s['label']:OperatorConfig('replace',{'new_value':'['+s['label']+']'}) for s in spans}
                    elif m=='delete': ops={'DEFAULT':OperatorConfig('redact')}
                    else: ops={'DEFAULT':OperatorConfig('replace',{'new_value':value})}
                    rendered=engine.anonymize(text=text,analyzer_results=clean,operators=ops,
                        conflict_resolution=ConflictResolutionStrategy.REMOVE_INTERSECTIONS,merge_entities_with_spaces=False)
                    assert rendered.text==v[m],(domain,r['record_id'],m)
                gold=np.zeros(len(text),dtype=bool);pred=np.zeros(len(text),dtype=bool)
                for s in r['spans']:gold[s['start']:s['end']]=True
                for s in spans:pred[s['start']:s['end']]=True;labels[s['label']]+=1
                # Ignore whitespace in coverage denominators to avoid boundary artifacts.
                content=np.asarray([not c.isspace() for c in text]); gold &= content; pred &= content
                tp=int((gold&pred).sum());fp=int((~gold&pred).sum());fn=int((gold&~pred).sum())
                totals.update(records=1,detected_records=int(bool(spans)),gold_spans=len(r['spans']),predicted_spans=len(spans),tp_chars=tp,fp_chars=fp,fn_chars=fn)
                rows.append({'domain':domain,'record_id':r['record_id'],'gold_spans':len(r['spans']),'detected_spans':len(spans),
                    'gold_nonspace_chars':int(gold.sum()),'detected_nonspace_chars':int(pred.sum()),'tp_chars':tp,'fp_chars':fp,'fn_chars':fn})
                out.write(json.dumps({**r,'predicted_spans':spans,'raw_analyzer_results':raw,
                    **{f'presidio_{m}':v[m] for m in METHODS}},ensure_ascii=False)+'\n')
                if (index+1)%250==0: print('detected',domain,index+1,'/',len(records),flush=True)
        audit={'domain':domain,**dict(totals),'predicted_label_counts':dict(labels)}
        audit['type_agnostic_char_precision']=totals['tp_chars']/max(1,totals['tp_chars']+totals['fp_chars'])
        audit['type_agnostic_char_recall']=totals['tp_chars']/max(1,totals['tp_chars']+totals['fn_chars'])
        audits.append(audit);print('completed',domain,json.dumps(audit),flush=True)
    with (args.output_dir/'detection_record_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={'analyzer':'Presidio AnalyzerEngine, all default supported English entities, score_threshold 0.5; no corpus-specific recognizers',
        'nlp_model':'en_core_web_lg 3.8.0','overlap_resolution':'AnonymizerEngine REMOVE_INTERSECTIONS; merge_entities_with_spaces=False',
        'rendering':'Five common operators; numbering resets per text in left-to-right span order. Other four policies verified against AnonymizerEngine.',
        'coverage':'Type-agnostic non-whitespace character overlap with gold; label taxonomies are not equivalent; this is not an official detector accuracy benchmark.',
        'versions':{p:importlib.metadata.version(p) for p in ('presidio-analyzer','presidio-anonymizer','spacy','en-core-web-lg')},'audits':audits}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('DONE PRESIDIO',len(rows),flush=True)

if __name__=='__main__':main()
