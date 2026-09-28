#!/usr/bin/env python3
"""User-directed PPL-only continuation of the immutable norm experiment.

Evaluates original FP16 source, original two-sweep E8/W5 parent, and the final
norm-compensation export on complete validation in one process. The earlier
combined-gate failure remains valid; MK no longer blocks this PPL research run.
No training, test-split scoring, publication, or frozen-helper edits occur here.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.norm_overlay import sha, verify_overlay, apply_overlay
from mamba_e8w5.runtime import (SentencePieceTokenizer, load_source_model, load_quantized_model,
                                environment_receipt, gpu_memory_receipt, token_digest)
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from audit_decoded import audit_loaded_model

PRIOR_REPORT_SHA = '72350eaa1989601f4412ba9e31c178e22dba95ec0b85f5837f9f08b29b265dad'
NORM_MANIFEST_SHA = '1fa9d35ff56a08e2f7ddf258bfa1fb61bc2e83ae2af404ab2f68a781821ac4d8'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def window_plan(windows):
    result = []
    next_start = targets = 0
    for first, tokens in windows:
        count = len(tokens) - 1
        if first != next_start or not 0 < count <= 2048:
            raise ValueError('Validation windows must cover every target exactly once')
        result.append({'start': first, 'input_tokens': count, 'target_tokens': count,
                       'token_sha256_int64le': token_digest(tokens.numpy())})
        next_start += count
        targets += count
    if targets != 264764:
        raise ValueError('Complete validation must score exactly264764 targets')
    return result


def check_arm(result, plan):
    if result['target_tokens'] != 264764 or result['execution'] != 'prefill' or result['logits_chunk_tokens'] != 64:
        raise ValueError('Full-validation execution or target count differs')
    if len(result['windows']) != len(plan):
        raise ValueError('Full-validation window count differs')
    for row, expected in zip(result['windows'], plan):
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('Full-validation token identity differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-dir', 'parent-dir', 'overlay-dir', 'training-report', 'training-checkpoint', 'smoke-report', 'report'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path, default=Path('calibration/v1/manifest.json'))
    parser.add_argument('--calibration-tokens', type=Path, default=Path('calibration/v1/calibration_tokens.pt'))
    parser.add_argument('--training-protocol', type=Path, default=Path('docs/NORM_COMPENSATION_PROTOCOL.md'))
    parser.add_argument('--prior-evaluation', type=Path, default=Path('reports/norm_compensation_v1_eval.json'))
    parser.add_argument('--continuation-note', type=Path, default=Path('docs/PPL_PRIORITY_CONTINUATION.md'))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    if sha(args.prior_evaluation) != PRIOR_REPORT_SHA:
        raise ValueError('Prior failed combined-gate receipt changed')
    previous = json.loads(args.prior_evaluation.read_text())
    if not previous['complete'] or previous['dev_comparison']['gate_passed'] is not False:
        raise ValueError('Continuation must preserve the recorded prior outcome')
    if sha(args.overlay_dir/'manifest.json') != NORM_MANIFEST_SHA:
        raise ValueError('Continuation must use the fixed final norm candidate')
    parent, overlay, receipt = verify_overlay(args.parent_dir, args.overlay_dir,
        calibration_manifest=args.calibration_manifest, calibration_tokens=args.calibration_tokens,
        protocol=args.training_protocol, training_report=args.training_report,
        training_checkpoint=args.training_checkpoint, smoke_report=args.smoke_report)
    for key, path in {'runtime_source_sha256': ROOT/'mamba_e8w5/runtime.py',
                      'evaluation_source_sha256': ROOT/'mamba_e8w5/evaluation.py',
                      'codec_source_sha256': ROOT/'mamba_e8w5/codec.py',
                      'norm_overlay_source_sha256': ROOT/'mamba_e8w5/norm_overlay.py'}.items():
        if sha(path) != previous[key]:
            raise ValueError(f'Frozen evaluation implementation changed: {key}')
    report = {'format': 'MAMBA2_NORM_PPL_CONTINUATION_V1', 'complete': False,
        'script_sha256': sha(__file__), 'continuation_note_sha256': sha(args.continuation_note),
        'run_scope_note_sha256': sha(ROOT/'docs/NORM_PPL_CONTINUATION.md'),
        'prior_evaluation_sha256': PRIOR_REPORT_SHA, 'overlay_receipt': receipt,
        'user_directed_scope_change': {
            'reason': 'User explicitly prioritizes PPL research; MK can be addressed later with Resurface.',
            'previous_combined_gate_result': 'FAIL; preserved unchanged',
            'current_scope': 'Complete validation PPL for fixed source/parent/norm arms; MK is recorded but not blocking.',
            'training_changed': False, 'test_split_used': False, 'publication_authorized': False},
        'prior_MK_not_rerun': {'parent': previous['parent_dev']['mk']['summary'],
                             'norm_candidate': previous['candidate_dev']['mk']['summary'],
                             'paired_cases': previous['dev_comparison']['paired_MK'],
                             'source_receipt_sha256': PRIOR_REPORT_SHA},
        'data_limitations': ['Validation has already been consulted in diagnosis and development; this is not untouched test evidence.',
                             'PPL-only continuation does not reverse the recorded MK regression or establish combined-quality acceptance.',
                             'Original source is a declared reference input, not a norm-candidate inference dependency.'],
        'arm_order': ['source_fp16', 'parent_e8w5', 'norm_e8w5'], 'same_process': True,
        'environment': environment_receipt(), 'frozen_code_sha256': {
            name: sha(ROOT/'mamba_e8w5'/name) for name in ('runtime.py','evaluation.py','codec.py','norm_overlay.py')},
        'results': {}}
    write_json(args.report, report)
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', WIKITEXT_REVISION)
    if dataset != previous['dataset'] or len(ids) != 264765:
        raise ValueError('Validation dataset/token stream differs from the frozen identity')
    windows = ppl_windows(ids, 2048, None)
    plan = window_plan(windows)
    report['dataset'] = dataset
    report['window_plan'] = plan
    report['window_plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    report['protocol'] = {'split': 'validation', 'target_tokens': 264764, 'maximum_targets_per_window': 2048,
        'number_of_windows': len(plan), 'execution': 'prefill', 'logits_chunk_tokens': 64,
        'loss': 'unchanged evaluation.py: FP16 head GEMM then FP32 summed cross entropy',
        'state': 'fresh zero state for every window; native SSD scan internal precision',
        'automatic_special_tokens': False}
    write_json(args.report, report)
    torch.cuda.reset_peak_memory_stats()
    print('[continuation] Loading and evaluating original FP16 source', flush=True)
    source = load_source_model(args.source_dir)
    report['source_package_receipt'] = source._package_receipt
    if len(dict(source.named_parameters())) != 507 or sum(p.numel() for p in source.parameters()) != 8236999680:
        raise ValueError('Source parameter coverage differs')
    result = evaluate_ppl(source, windows, execution='prefill', logits_chunk=64)
    check_arm(result, plan)
    report['results']['source_fp16'] = result
    write_json(args.report, report)
    del source
    gc.collect()
    torch.cuda.empty_cache()
    print('[continuation] Loading and auditing original two-sweep E8/W5 parent', flush=True)
    model = load_quantized_model(args.parent_dir)
    report['parent_package_receipt'] = model._package_receipt
    report['parent_loaded_tensor_audit'] = audit_loaded_model(model, parent, args.parent_dir)
    print('[continuation] Evaluating original two-sweep E8/W5 parent', flush=True)
    result = evaluate_ppl(model, windows, execution='prefill', logits_chunk=64)
    check_arm(result, plan)
    report['results']['parent_e8w5'] = result
    write_json(args.report, report)
    print('[continuation] Reloading and evaluating final FP16 norm export', flush=True)
    report['norm_overlay_application'] = apply_overlay(model, args.overlay_dir, overlay)
    result = evaluate_ppl(model, windows, execution='prefill', logits_chunk=64)
    check_arm(result, plan)
    report['results']['norm_e8w5'] = result
    report['candidate_package_receipt'] = {key: receipt[key] for key in
        ('parent_manifest_sha256','overlay_manifest_sha256','resolved_file_ledger_sha256','logical_candidate_data_bytes')}
    source, base, norm = (report['results'][name] for name in report['arm_order'])
    paired = []
    for expected, s, b, n in zip(plan, source['windows'], base['windows'], norm['windows']):
        paired.append({**expected, 'source_nll': s['nll'], 'parent_nll': b['nll'], 'norm_nll': n['nll'],
                       'source_ppl': s['ppl'], 'parent_ppl': b['ppl'], 'norm_ppl': n['ppl'],
                       'norm_minus_parent_mean_nll': (n['nll']-b['nll'])/expected['target_tokens'],
                       'norm_minus_source_mean_nll': (n['nll']-s['nll'])/expected['target_tokens']})
    report['paired_windows'] = paired
    report['comparison'] = {'source_ppl': source['ppl'], 'parent_ppl': base['ppl'], 'norm_ppl': norm['ppl'],
        'norm_vs_parent_ppl_relative_change': norm['ppl']/base['ppl']-1,
        'parent_vs_source_ppl_relative_change': base['ppl']/source['ppl']-1,
        'norm_vs_source_ppl_relative_change': norm['ppl']/source['ppl']-1,
        'norm_minus_parent_mean_nll': (norm['nll']-base['nll'])/264764,
        'norm_minus_source_mean_nll': (norm['nll']-source['nll'])/264764,
        'source_plus5_percent_PPL_reference_met': norm['ppl'] <= 1.05*source['ppl'],
        'note': 'PPL-only research report; prior combined gate remains failed and MK repair remains separate.'}
    if sha(args.prior_evaluation) != PRIOR_REPORT_SHA or sha(args.overlay_dir/'manifest.json') != NORM_MANIFEST_SHA:
        raise ValueError('Immutable prior receipt/candidate changed during continuation')
    report.update(complete=True, elapsed_seconds=time.perf_counter()-started, gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    print(json.dumps({'complete': True, 'comparison': report['comparison'],
                      'report_sha256': sha(args.report)}), flush=True)


if __name__ == '__main__':
    main()
