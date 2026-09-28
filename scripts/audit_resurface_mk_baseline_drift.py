#!/usr/bin/env python3
"""CPU-only immutable snapshot of historical/current native MK baseline drift.

No model is loaded, no numerical settings are changed, and no quality gate is
relaxed. Hash-ledger equality is report evidence, not a fresh model-file audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

HISTORICAL_SHA = 'a88f8fe75e82b992b8fc9cebeb6f681847e87aa36ea8bb1771edf979c47d551d'
PPL_SHA = '3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4'
SCRIPT_EVALUATOR_SHA = '7d8ecefb8c28c4a74b9a9f3faed22831849a312cc65781f22404a3200d52c471'
OLD = 'current_axis_readapted_small'
CURRENT = 'current_readapted'
RESTORED = 'restored_current'
GENERATED = {'generated_ids', 'output', 'prediction', 'correct'}


def read(path, expected=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Pinned historical input differs: {path}')
    return json.loads(raw), {'file':str(path.resolve()), 'bytes':len(raw), 'sha256':digest}


def first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def diff_rows(a, b):
    if len(a) != 768 or len(b) != 768 or [x['id'] for x in a] != [x['id'] for x in b]:
        raise ValueError('Paired768 ordered cases required')
    diffs = []
    for old, new in zip(a, b):
        if set(old) != set(new) or {k:v for k,v in old.items() if k not in GENERATED} != {k:v for k,v in new.items() if k not in GENERATED}:
            raise ValueError('A non-generation prompt/cache/scoring field differs')
        changed = sorted(k for k in old if old[k] != new[k])
        if changed:
            index = first_difference(old['generated_ids'], new['generated_ids'])
            diffs.append({'id':old['id'], 'condition':old['condition'], 'N':old['N'], 'template':old['template'],
                'answer':old['answer'], 'prompt_tokens':old['prompt_tokens'],
                'prompt_token_sha256_int64le':old['prompt_token_sha256_int64le'], 'changed_fields':changed,
                'first_divergent_generated_offset_zero_based':index,
                'equal_generated_prefix':old['generated_ids'][:index],
                'historical':{k:old[k] for k in GENERATED}, 'current':{k:new[k] for k in GENERATED}})
    return diffs


def model_proof(old, new):
    expected = old['actual507_before']['unchanged_content_sha256']
    if len(expected) != 507:
        raise ValueError('Expected507 model hashes')
    for arm in (old, new):
        for side in ('before','after'):
            if arm[f'actual507_{side}']['unchanged_content_sha256'] != expected:
                raise ValueError('Report model content hashes differ')
        if arm['identities507_before'] != arm['identities507_after'] or arm['replay_exact'] is not True:
            raise ValueError('Within-arm parameter identity or12 replays differ')
    structural = ('dtype','shape','stride','requires_grad')
    a,b = old['identities507_before'],new['identities507_before']
    if set(a) != set(b) or any(a[k][f] != b[k][f] for k in a for f in structural):
        raise ValueError('Parameter shape/stride/dtype/trainability changed')
    return {'tensor_count':507, 'all_report_content_hashes_exact':True, 'structural_metadata_exact':True,
        'each_arm_identity_storage_version_unchanged':True,
        'historical_and_current12_replays_exact':True,
        'pointer_alignment_mismatch_counts':{str(mod):sum(a[k]['data_ptr']%mod != b[k]['data_ptr']%mod for k in a)
            for mod in (16,128,256,512)}, 'actual_fp16_sha256':expected,
        'scope':'Previously measured actual tensor ledgers; this CPU audit does not reload model binaries.'}


def audit(args):
    old, old_receipt = read(args.historical,HISTORICAL_SHA)
    new, new_receipt = read(args.current)
    ppl, ppl_receipt = read(args.ppl,PPL_SHA)
    if old.get('complete') is not True or new.get('script_sha256') != SCRIPT_EVALUATOR_SHA:
        raise ValueError('Historical completion or frozen evaluator identity differs')
    a,b = old['arms'][OLD],new['arms'][CURRENT]
    if a['complete'] is not True or b['complete'] is not True:
        raise ValueError('Both current baseline arms must be complete')
    diffs = diff_rows(a['rows'],b['rows'])
    proof = model_proof(a,b)
    old_ppl, now_ppl = ppl['development'][OLD],new['ppl'][CURRENT]
    if len(now_ppl['windows']) != 64 or old_ppl['windows'] != now_ppl['windows'] or old_ppl['summary'] != now_ppl['summary']:
        raise ValueError('Historical64-prose per-window/token/chunk/nll/summary differs')
    restored = new.get('arms',{}).get(RESTORED)
    restoration = {'available':False, 'reason':'Restored-current768 MK arm has not completed in this snapshot.'}
    if restored and restored.get('complete') is True:
        rd = diff_rows(b['rows'],restored['rows'])
        restoration = {'available':True, 'generated_rows_exact':not rd,
            'summary_exact':b['summary']==restored['summary'], 'changed_rows':rd,
            'model':model_proof(b,restored)}
    normal = [row for row in diffs if row['condition']=='normal']
    return {'format':'MAMBA2_RESURFACE_MK_BASELINE_DRIFT_AUDIT_V1', 'complete':True,
        'snapshot_utc_unix':time.time(), 'live_evaluation_complete':new.get('complete'),
        'live_evaluation_error':new.get('error'), 'inputs':{'historical':old_receipt,'current_snapshot':new_receipt,'ppl':ppl_receipt},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cpu_only':True,'python':platform.python_version(),'gpu_model_loaded':False,
        'paired_prompt_count':768,'all_prompt_metadata_token_hashes_cache_metadata_exact':True,
        'environment_receipts_exact':old.get('environment')==new.get('environment'),
        'historical_arm_order':old['arm_order'],'new_arm_order':['current_readapted','active_resurface','restored_current'],
        'model':proof,'changed_rows':diffs,'changed_row_count':len(diffs),
        'changed_normal_rows':len(normal),'changed_removed_rows':len(diffs)-len(normal),
        'historical_summary':a['summary'],'current_summary':b['summary'],
        'normal_gained':sum(not row['historical']['correct'] and row['current']['correct'] for row in normal),
        'normal_lost':sum(row['historical']['correct'] and not row['current']['correct'] for row in normal),
        'prose64':{'all_windows_and_fp32_CE_chunks_exact':True,'summary':now_ppl['summary']},
        'same_process_restoration':restoration,
        'quality_gate_changed':False,'quality_pass_claimed':False,
        'interpretation':'A generation-only historical mismatch is observed with equal prompts, parameter content/layout metadata and64 prose CE. Different native numerical paths or cached-generation execution are plausible but unproven; no per-token hidden/logit/cache-value traces exist to establish a cause.'}


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical',type=Path,default=root/'reports/readapted_mk_baseline_v1.json')
    p.add_argument('--current',type=Path,default=root/'reports/resurface_readapted_v1_dev_eval.json')
    p.add_argument('--ppl',type=Path,default=root/'reports/axis_small_readaptation_v1_eval.json')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    result=audit(args)
    with args.report.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'changed_rows':result['changed_row_count'],
        'normal_gained':result['normal_gained'],'normal_lost':result['normal_lost'],
        'restoration_available':result['same_process_restoration']['available']}))


if __name__=='__main__':main()
