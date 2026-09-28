#!/usr/bin/env python3
"""Independent terminal CONFIRM receipt arithmetic/provenance audit, CPU only.

Reads final JSON reports/manifests only. Never imports Torch, model code or
dataset loaders; never opens TRAIN/CONFIRM raw/token payloads or model binaries.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '3ab56d0483f1a0b8e3c531829daa6ab6f7dfcb693288f455f8dc28e4d4baa308'
PROTOCOL = 'f6be64321c214bb7b229ae658b24aa75b8a0f3ec17a9df9059c4303920cdbabf'
FULL_SHA = '06a71c11fc0a12a52add6e7bf5d28b8a8eb9f832b2ec1cb846d9acaf30afe961'
CONFIRM_SHA = '0abb3e1ef2994e2088b8da0f0f605371e980f5afa3ea9df5335ff19d3f5b3dec'
BASE, ACTIVE, RESTORED = 'current_readapted', 'active_resurface', 'restored_current'
MATCH = re.compile(r'(?<!\d)\d{6}(?!\d)')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(name, digest=None):
    path = ROOT / 'reports' / name
    require(path.parent == ROOT / 'reports' and path.suffix == '.json', 'Reports-only JSON input')
    raw = path.read_bytes()
    receipt = {'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    require(digest is None or receipt['sha256'] == digest, 'Pinned JSON differs: ' + name)
    return json.loads(raw), receipt


def terminal(process, receipt):
    require(process.get('status') == 'finished' and type(process.get('exit_code')) is int
            and process['exit_code'] == 0 and process.get('report_complete') is True,
            'Successful terminal process required')
    require(process['report_sha256'] == receipt['sha256'] and process['report_bytes'] == receipt['bytes'],
            'Process/report hash and bytes differ')
    require(process['finished_unix'] >= process['started_unix'], 'Process time order differs')


def cp_bound(k, n, lower):
    """Invert direct binomial polynomials, independent of evaluator lgamma sums."""
    if lower and k == 0:
        return 0.0
    if not lower and k == n:
        return 1.0
    indexes = range(k, n + 1) if lower else range(k + 1)
    coefficients = [(j, math.comb(n, j)) for j in indexes]
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        tail = math.fsum(c * mid ** j * (1 - mid) ** (n - j) for j, c in coefficients)
        move_lower_endpoint = tail < .025 if lower else tail > .025
        if move_lower_endpoint:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired(first, second, condition):
    counts = dict(both_correct=0, both_wrong=0, gained=0, lost=0)
    categories = {(True, True): 'both_correct', (False, False): 'both_wrong',
                  (False, True): 'gained', (True, False): 'lost'}
    for a, b in zip(first, second):
        require(all(a[k] == b[k] for k in ('id', 'condition', 'answer', 'prompt_token_sha256_int64le',
                                         'N', 'template', 'query_position')), 'Paired input differs')
        if a['condition'] == condition:
            counts[categories[a['correct'], b['correct']]] += 1
    discordant = counts['gained'] + counts['lost']
    p = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(counts['gained'], counts['lost']) + 1))
             / (1 << discordant)) if discordant else 1.)
    return {'count': sum(counts.values()), 'source_correct': counts['both_correct'] + counts['lost'],
            'current_correct': counts['both_correct'] + counts['gained'], 'paired': counts,
            'mcnemar_exact_two_sided_p': p,
            'delta_percentage_points': 100 * (counts['gained'] - counts['lost']) / sum(counts.values())}


def audit_arm(arm, plan, replay_ids, expected, identities):
    require(arm['complete'] is True and arm['replay_exact'] is True, 'Incomplete native MK arm')
    rows = arm['rows']
    require(len(rows) == len(plan) == 768, 'Expected768 ordered rows')
    cells = Counter()
    for row, case in zip(rows, plan):
        require(all(row[k] == value and type(row[k]) is type(value) for k, value in case.items()),
                'Case plan metadata differs')
        ids = row['generated_ids']
        require(1 <= len(ids) <= 12 and all(type(x) is int and 0 <= x < 256000 for x in ids),
                'Generated token inventory differs')
        match = MATCH.search(row['output'])
        prediction = match.group() if match else None
        require(row['prediction'] == prediction and type(row['correct']) is bool
                and row['correct'] == (prediction == row['answer']), 'Numeric matching differs')
        require(row['finite_hidden_every_backbone_call'] is True, 'Missing finite native hidden proof')
        cache = {'layers': 56, 'tensors': 112, 'dtype': 'torch.float16', 'finite': True, 'bytes': 122028032}
        require(row['prefill_cache'] == row['final_cache'] == cache and row['cache_bytes'] == cache['bytes'],
                'Native cache coverage differs')
        cells[row['N'], row['template'], row['condition']] += 1
    require(len(cells) == 12 and set(cells.values()) == {64}, 'Unbalanced confirmation cells')
    for condition in ('normal', 'target_removed'):
        chosen = [r for r in rows if r['condition'] == condition]
        summary = arm['summary'][condition]
        correct = sum(r['correct'] for r in chosen)
        require(summary['correct'] == correct and summary['count'] == 384
                and summary['accuracy'] == correct / 384, 'MK summary differs')
    require(len(arm['cells']) == 12, 'Missing per-cell summaries')
    for cell in arm['cells']:
        selected = [r for r in rows if all(r[k] == cell[k] for k in ('N', 'template', 'condition'))]
        require(cell['count'] == 64 and cell['correct'] == sum(r['correct'] for r in selected)
                and cell['query_positions'] == [r['query_position'] for r in selected], 'Per-cell result differs')
    by_id = {r['id']: r for r in rows}
    require([r['id'] for r in arm['replays']] == replay_ids and len(replay_ids) == 12, 'Replay IDs differ')
    for replay in arm['replays']:
        require(replay['exact_equal'] is True and all(replay[k] == by_id[replay['id']][k]
                for k in ('generated_ids', 'output', 'prediction', 'correct')), 'Fresh-cache replay differs')
    for side in ('before', 'after'):
        proof = arm['actual507_' + side]
        require(proof['verified_tensor_count'] == 507 and proof['unchanged_content_sha256'] == expected,
                'Recorded base507 content differs')
        require(arm['identities507_' + side] == identities, 'Recorded base identity changed')
    require(arm['identity_storage_version_unchanged'] is True, 'Identity guard failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-report-sha', required=True)
    parser.add_argument('--confirm-process-sha', required=True)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/resurface_soft_continuation_v1_confirm_independent_audit.json')
    args = parser.parse_args()
    out = args.report.resolve()
    require(out.parent == ROOT/'reports' and not out.exists(), 'Fresh audit under reports required')
    d, dr = read('resurface_soft_continuation_v1_confirm_eval.json', args.confirm_report_sha)
    process, proc_receipt = read('resurface_soft_continuation_v1_confirm_eval_process.json', args.confirm_process_sha)
    terminal(process, dr)
    require(d['complete'] is True and d['stage'] == 'confirm' and d['mode'] == 'native_fixed_soft_continuation',
            'Completed native confirmation required')
    require(d['script_sha256'] == SOURCE and d['protocol_sha256'] == PROTOCOL and d['gate_mode'] == 'soft',
            'Source/protocol/mode differs')
    full, full_receipt = read('resurface_soft_continuation_v1_full_eval.json', FULL_SHA)
    full_process, full_proc_receipt = read('resurface_soft_continuation_v1_full_eval_process.json')
    terminal(full_process, full_receipt)
    require(full['complete'] is True and full['gate']['passed'] is True, 'Full PPL prerequisite failed')
    require(d['prior_stage']['sha256'] == FULL_SHA and d['prior_stage']['stage'] == 'full'
            and d['prior_stage']['gate'] == full['gate'], 'Full-stage receipt binding differs')
    require(d['candidate_identity'] == full['candidate_identity'] and d['export_proof'] == full['export_proof'],
            'Confirmation selected another export/checkpoint')
    prep, prep_receipt = read('resurface_readapted_v1_prepare_confirm.json')
    prep_process, prep_proc_receipt = read('resurface_readapted_v1_prepare_confirm_process.json')
    terminal(prep_process, prep_receipt)
    manifest, manifest_receipt = read('resurface_readapted_v1_confirm_data_manifest.json', CONFIRM_SHA)
    require(prep_process['started_unix'] > full_process['finished_unix']
            and process['started_unix'] > prep_process['finished_unix'], 'Confirmation generation/stage time order differs')
    require(prep_process['required_reports']['reports/resurface_soft_continuation_v1_full_eval.json'] == FULL_SHA,
            'Preparation was not bound to completed full PPL')
    require(prep['complete'] is True and prep['split'] == 'confirm' and prep['gpu_used'] is False
            and prep['cuda_initialized'] is False and prep['exact_raw_and_tokenized_roundtrip'] is True,
            'CPU confirmation preparation proof differs')
    require(prep['manifest'] == manifest and prep['manifest_sha256'] == CONFIRM_SHA
            and prep['manifest_bytes'] == manifest_receipt['bytes'] and d['stage_data']['manifest'] == manifest,
            'Actual confirmation manifest binding differs')
    expected_contract = {'seed': 2026092802, 'key_range': [400000,480000],
        'value_ranges': [[691000,700000],[791000,800000],[891000,900000],[991000,1000000]],
        'samples_per_cell': 64, 'normal_count':384,'removed_count':384,
        'positions':'floor(sample*(N-1)/63)', 'control_key_base':490000,'control_value_base':990000}
    require(manifest['contract'] == expected_contract and manifest['row_count'] == 768
            and manifest['answer_target_tokens'] == 0 and manifest['evaluation_supervised_targets_stored'] is False,
            'Frozen confirmation data contract differs')
    previous_ranges = [[600000,680000],[690000,690064],[681000,690000],[781000,790000],
                       [881000,890000],[981000,990000]]
    require(all(max(a,c) >= min(b,e) for a,b in expected_contract['value_ranges']
                for c,e in previous_ranges + [[990000,990064]]), 'Cross-split/control value pools overlap')
    require(all(max(400000,a) >= min(480000,b) for a,b in [[100000,180000],[190000,190064],
                [300000,380000],[490000,490064]]), 'Cross-split/control key pools overlap')
    plan = d['stage_data']['case_plan']; require(len(plan) == 768, 'Case plan incomplete')
    expected_ids = []
    for size in (16,64):
        for template in range(3):
            for sample in range(64):
                normal = f'resurface-confirm-n{size}-t{template}-s{sample}'
                expected_ids.extend([normal, normal+'-removed'])
    require([r['id'] for r in plan] == expected_ids and len({r['prompt_token_sha256_int64le'] for r in plan}) == 768,
            'Ordered/unique confirmation inputs differ')
    for index in range(0,768,2):
        normal, removed = plan[index:index+2]; sample = int(normal['id'].rsplit('-s',1)[1])
        require(normal['condition'] == 'normal' and removed['condition'] == 'target_removed'
                and all(normal[k] == removed[k] for k in ('key','answer','N','template','query_position'))
                and normal['query_position'] == sample*(normal['N']-1)//63, 'Paired query metadata differs')
        require(400000 <= normal['key'] < 480000 and any(lo <= int(normal['answer']) < hi
                for lo,hi in expected_contract['value_ranges']), 'Query key/value outside frozen pools')
    expected = full['final_current507_audit']['unchanged_content_sha256']
    identities = d['arms'][BASE]['identities507_before']; require(len(expected) == len(identities) == 507, 'Base coverage differs')
    require(set(d['arms']) == {BASE,ACTIVE,RESTORED} and 'ppl' not in d, 'Confirmation stage task scope differs')
    for arm in (BASE,ACTIVE,RESTORED):
        audit_arm(d['arms'][arm], plan, d['stage_data']['plan']['replay_case_ids'], expected, identities)
    for field in ('rows','summary','cells','replays'):
        require(d['arms'][BASE][field] == d['arms'][RESTORED][field], 'Complete restored-current output differs')
    require(d['mk_restoration_exact'] is True, 'Missing complete restoration proof')
    comparison = {condition: paired(d['arms'][BASE]['rows'],d['arms'][ACTIVE]['rows'],condition)
                  for condition in ('normal','target_removed')}
    for condition, computed in comparison.items():
        require(all(d['mk_comparison'][condition][k] == v for k,v in computed.items()), 'Independent paired arithmetic differs')
    counts = comparison['normal']['paired']; lower = cp_bound(counts['gained'],384,True); upper = cp_bound(counts['lost'],384,False)
    bound = lower-upper; scipy_note = {'available': False}
    try:
        from scipy.stats import beta
    except ImportError:
        pass
    else:
        sl = 0. if counts['gained']==0 else float(beta.ppf(.025,counts['gained'],385-counts['gained']))
        su = 1. if counts['lost']==384 else float(beta.ppf(.975,counts['lost']+1,384-counts['lost']))
        require(abs(sl-lower)<2e-12 and abs(su-upper)<2e-12, 'Independent SciPy beta inversion differs')
        scipy_note = {'available':True,'lower':sl,'upper':su,'agrees_with_direct_binomial_inversion':True}
    gate = d['gate']; quality = {'normal_gain_positive':counts['gained']>counts['lost'],
        'mcnemar_p_below_0_05':comparison['normal']['mcnemar_exact_two_sided_p']<.05,
        'paired_lower_bound_positive':bound>0.,
        'control_no_increase':comparison['target_removed']['current_correct']<=comparison['target_removed']['source_correct']}
    quality['passed'] = all(quality.values())
    require(all(gate[k] is v for k,v in quality.items()), 'Strict confirmation gate differs')
    require(abs(gate['gained_lower_one_sided97_5']-lower)<2e-12
            and abs(gate['lost_upper_one_sided97_5']-upper)<2e-12
            and abs(gate['paired_accuracy_improvement_conservative95_lower']-bound)<2e-12,
            'Reported conservative bound differs')
    for field in ('initial_current507_audit','final_current507_audit'):
        require(d[field]['verified_tensor_count']==507 and d[field]['unchanged_content_sha256']==expected, 'Final base content differs')
    require(d['final_current507_identity']==identities and d['adapter_before']==d['adapter_after'], 'Final base/adapter identity differs')
    require(d['adapter_before']['tensor_count']==224 and len(d['adapter_before']['identities'])==224
            and d['adapter_before']['tensor_sha256']==full['adapter_before']['tensor_sha256'], 'Actual adapter224 content differs')
    require(d['environment']==d['environment_reference']==full['environment'], 'Native environment changed')
    for snapshot in ('numerical_before','numerical_after'):
        require({k:v for k,v in d[snapshot].items() if k!='already_loaded_autotuner_metadata'} ==
                {k:v for k,v in full[snapshot].items() if k!='already_loaded_autotuner_metadata'}, 'Numerical flags changed')
    ledger=d['checked_input_sha256']
    require(d['final_bound_input_recheck']=={'file_count':len(ledger),'checked_input_sha256':ledger}, 'Final bound-file proof differs')
    for name,receipt in manifest['files'].items():
        require(ledger.get('/home/horde/Mamba2-8B-E8W5/training_data/resurface_readapted_v1/confirm/'+name)==receipt['sha256'],
                'Scored confirmation file not bound to prepared split')
    result={'format':'MAMBA2_RESURFACE_CONFIRM_INDEPENDENT_RECEIPT_AUDIT_V1','complete':True,
        'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':{'confirm':dr,'confirm_process':proc_receipt,'full':full_receipt,'full_process':full_proc_receipt,
                  'prepare':prep_receipt,'prepare_process':prep_proc_receipt,'manifest':manifest_receipt},
        'candidate_identity':d['candidate_identity'],'comparison':comparison,'gate':quality,
        'conservative95_lower':bound,'gained_lower97_5':lower,'lost_upper97_5':upper,'scipy_crosscheck':scipy_note,
        'time_order':{'full_finished_unix':full_process['finished_unix'],'prepare_started_unix':prep_process['started_unix'],
            'prepare_finished_unix':prep_process['finished_unix'],'confirm_started_unix':process['started_unix']},
        'checks':{'all3x768_inputs_cells_replays_cache':True,'restored_MK_exact':True,'base507_adapter224_recorded_identities':True,
                  'same_full_PPL_candidate_and_native_environment':True,'final_recorded_file_audit_complete':True,
                  'frozen_numeric_pool_metadata_disjoint':True,'preparation_after_full_terminal_success':True},
        'recorded_bound_file_count':len(ledger),'terminal_exit_code':0,
        'limits':['CPU report/manifest arithmetic audit; no GPU or model binary reread.',
                  'No TRAIN/CONFIRM raw/token payload opened. Prompt binding/absence proof comes from pinned preparation and evaluator receipts.',
                  'Process chronology establishes recorded preparation after full PPL; it cannot prove absence of undocumented prior access.',
                  'Same three template families; results do not establish unseen-template or universal recall quality.',
                  'Original strict historical-MK replay failure remains preserved; no publication authorized.']}
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'report':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                      'comparison':comparison,'conservative95_lower':bound,'gate':quality},indent=2))


if __name__ == '__main__':
    main()
