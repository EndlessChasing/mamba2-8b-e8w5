#!/usr/bin/env python3
"""Independent full-validation evaluation of the final all-small FP16 export.

Four fixed arms run in one process with unchanged evaluation.py math. Training
and previous receipts remain immutable. No checkpoint selection, MK scoring,
test-set access, container promotion or publication is performed here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5 import norm_overlay, small_overlay
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from mamba_e8w5.runtime import (SentencePieceTokenizer, load_source_model,
    load_quantized_model, environment_receipt, gpu_memory_receipt, token_digest)
from audit_decoded import audit_loaded_model, tensor_sha_fp16

sha = norm_overlay.sha
REFERENCE_REPORT_SHA = '298e423566e1c6dba39570f72b3c80b3185aa3fa8f6a8d4597b1192a42cde68d'
PROTOCOL_SHA = 'f57229f979f8bc94def0780f87aa48ae0b83f7d30dc29fef527b46aaa2be0c31'
CONTINUATION_SHA = 'e57512650cc1f9b89c043ab9fb3092e51ab0e045cf571ebc454c83df12d22391'
TARGETS = 264764
ARMS = ('source_fp16', 'parent_e8w5', 'norm_e8w5', 'small_e8w5')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def window_plan(windows):
    plan = []
    start = 0
    for first, tokens in windows:
        count = len(tokens) - 1
        if first != start or count <= 0 or count > 2048:
            raise ValueError('Validation windows must cover every target exactly once')
        plan.append({'start': first, 'input_tokens': count, 'target_tokens': count,
                     'token_sha256_int64le': token_digest(tokens.numpy())})
        start += count
    if start != TARGETS or len(plan) != 130:
        raise ValueError('Expected complete validation: 264764 targets and 130 windows')
    return plan


def check_arm(result, plan):
    """Check matched input rows and independently recompute weighted aggregate."""
    if (result['target_tokens'] != TARGETS or result['execution'] != 'prefill'
            or result['logits_chunk_tokens'] != 64 or len(result['windows']) != len(plan)):
        raise ValueError('Full-validation protocol differs')
    for row, expected in zip(result['windows'], plan):
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('Full-validation token identity differs')
        if not math.isfinite(row['nll']) or row['nll'] < 0:
            raise ValueError('Nonfinite or negative window NLL')
        if not math.isclose(row['ppl'], math.exp(row['nll']/row['target_tokens']), rel_tol=1e-12):
            raise ValueError('Per-window PPL differs from exp(mean NLL)')
    total = math.fsum(row['nll'] for row in result['windows'])
    if not math.isclose(result['nll'], total, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Aggregate NLL differs from window sum')
    if not math.isclose(result['ppl'], math.exp(total/TARGETS), rel_tol=1e-12):
        raise ValueError('Aggregate PPL is not target-weighted exp(mean NLL)')


def comparisons(results, plan):
    for result in results.values():
        check_arm(result, plan)
    if set(results) != set(ARMS):
        raise ValueError('All four same-process arms are required')
    source, parent, norm, candidate = (results[name] for name in ARMS)
    paired = []
    for index, expected in enumerate(plan):
        rows = {name: results[name]['windows'][index] for name in ARMS}
        paired.append({**expected,
            'nll': {name: row['nll'] for name, row in rows.items()},
            'ppl': {name: row['ppl'] for name, row in rows.items()},
            'small_minus_norm_mean_nll': (rows['small_e8w5']['nll']-rows['norm_e8w5']['nll'])/expected['target_tokens'],
            'small_minus_source_mean_nll': (rows['small_e8w5']['nll']-rows['source_fp16']['nll'])/expected['target_tokens']})
    result = {'ppl': {name: results[name]['ppl'] for name in ARMS},
        'mean_nll': {name: results[name]['nll']/TARGETS for name in ARMS},
        'small_vs_norm_ppl_relative_change': candidate['ppl']/norm['ppl']-1,
        'small_vs_parent_ppl_relative_change': candidate['ppl']/parent['ppl']-1,
        'source_relative_ppl_gaps': {name: results[name]['ppl']/source['ppl']-1 for name in ARMS[1:]},
        'small_minus_norm_mean_nll': (candidate['nll']-norm['nll'])/TARGETS,
        'window_counts_against_norm': {
            'improved': sum(row['small_minus_norm_mean_nll'] < 0 for row in paired),
            'unchanged': sum(row['small_minus_norm_mean_nll'] == 0 for row in paired),
            'worsened': sum(row['small_minus_norm_mean_nll'] > 0 for row in paired)},
        'meaningful_improvement_reference': {'relative_ppl_reduction_required': 0.01,
            'against': 'norm_e8w5', 'met': candidate['ppl'] <= 0.99*norm['ppl']},
        'source_plus5_percent_reference': {'maximum_ppl': 1.05*source['ppl'],
            'met': candidate['ppl'] <= 1.05*source['ppl']},
        'note': 'PPL-only validation research. MK is not measured here; no combined-quality or publication acceptance is implied.'}
    return result, paired


def audit_parameter_coverage(model):
    parameters = dict(model.named_parameters())
    count = sum(value.numel() for value in parameters.values())
    if len(parameters) != 507 or count != 8236999680:
        raise ValueError('Loaded architecture parameter coverage differs')
    return {'tensors': len(parameters), 'parameters': count}


@torch.no_grad()
def audit_selected_values(model, values):
    """Independent canonical-byte comparison of actual loaded/exported values."""
    parameters = dict(model.named_parameters())
    hashes = {}
    for name, expected in values.items():
        actual = parameters[name]
        if actual.shape != expected.shape or actual.dtype != torch.float16 or expected.dtype != torch.float16:
            raise ValueError(f'Loaded/export FP16 shape or dtype differs: {name}')
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError(f'Nonfinite loaded/exported parameter: {name}')
        actual_hash, expected_hash = tensor_sha_fp16(actual), tensor_sha_fp16(expected)
        if actual_hash != expected_hash:
            raise ValueError(f'Loaded FP16 bytes differ from export: {name}')
        hashes[name] = actual_hash
    return {'verified_tensor_count': len(hashes), 'loaded_and_export_fp16_sha256': hashes}


@torch.no_grad()
def audit_frozen_values(model, expected_hashes):
    """Rehash actual GPU tensors; retaining parameter objects alone is insufficient."""
    hashes = {}
    for name, expected in expected_hashes.items():
        actual = tensor_sha_fp16(model.get_parameter(name))
        if actual != expected:
            raise ValueError(f'Frozen E8/W5 tensor changed: {name}')
        hashes[name] = actual
    return {'verified_tensor_count': len(hashes), 'unchanged_content_sha256': hashes}


def audit_checkpoint_export(checkpoint_path, exported, binding, expected_digest):
    """Second, evaluator-owned final-master rounding check, independent of trainer."""
    if sha(checkpoint_path) != expected_digest:
        raise ValueError('Final checkpoint file hash differs')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (state.get('format') != 'MAMBA2_SMALL_TRAINING_CHECKPOINT_V1'
            or state.get('successful_updates') != 1024 or state.get('binding') != binding):
        raise ValueError('Only the bound final1024 checkpoint is eligible')
    masters = state['masters']
    if set(masters) != set(exported):
        raise ValueError('Checkpoint master/export coverage differs')
    hashes = {}
    for name, value in exported.items():
        master = masters[name]
        if master.dtype != torch.float32 or master.shape != value.shape or not torch.isfinite(master).all():
            raise ValueError(f'Invalid FP32 master: {name}')
        rounded_hash = tensor_sha_fp16(master.half())
        if rounded_hash != tensor_sha_fp16(value):
            raise ValueError(f'Final rounded master differs from export: {name}')
        hashes[name] = rounded_hash
    return {'checkpoint_sha256': expected_digest, 'successful_updates': 1024,
            'verified_tensor_count': len(hashes), 'rounded_fp16_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-dir', 'parent-dir', 'initialization-dir', 'overlay-dir',
                'training-report', 'training-checkpoint', 'smoke-report', 'report'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('training_data/small_compensation_v1'))
    parser.add_argument('--protocol', type=Path, default=Path('docs/SMALL_TENSOR_COMPENSATION_PROTOCOL.md'))
    parser.add_argument('--continuation-note', type=Path, default=Path('docs/PPL_PRIORITY_CONTINUATION.md'))
    parser.add_argument('--reference-report', type=Path, default=Path('reports/norm_compensation_v1_ppl_continuation.json'))
    parser.add_argument('--norm-training-report', type=Path, default=Path('reports/norm_compensation_v1_train.json'))
    parser.add_argument('--norm-training-checkpoint', type=Path, default=Path('artifacts/norm_compensation_v1_train_work/checkpoint_attempt0128_step0128.pt'))
    parser.add_argument('--norm-smoke-report', type=Path, default=Path('reports/norm_compensation_v1_smoke.json'))
    parser.add_argument('--verify-only', action='store_true', help='CPU integrity only; no dataset/model loading or PPL')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    if sha(args.reference_report) != REFERENCE_REPORT_SHA or sha(args.protocol) != PROTOCOL_SHA:
        raise ValueError('Fixed reference receipt or declared training protocol changed')
    if sha(args.continuation_note) != CONTINUATION_SHA:
        raise ValueError('User-directed PPL scope note changed')
    previous = json.loads(args.reference_report.read_text())
    if previous.get('complete') is not True:
        raise ValueError('Incomplete norm reference receipt')
    frozen = previous['frozen_code_sha256']
    for name, expected in frozen.items():
        if sha(ROOT/'mamba_e8w5'/name) != expected:
            raise ValueError(f'Frozen evaluation implementation changed: {name}')
    parent, overlay, receipt = small_overlay.verify_overlay(args.parent_dir, args.overlay_dir,
        initialization_dir=args.initialization_dir, data_dir=args.data_dir, protocol=args.protocol,
        training_report=args.training_report, training_checkpoint=args.training_checkpoint,
        smoke_report=args.smoke_report)
    _, norm, norm_receipt = norm_overlay.verify_overlay(args.parent_dir, args.initialization_dir,
        calibration_manifest=ROOT/'calibration/v1/manifest.json',
        calibration_tokens=ROOT/'calibration/v1/calibration_tokens.pt',
        protocol=ROOT/'docs/NORM_COMPENSATION_PROTOCOL.md',
        training_report=args.norm_training_report, training_checkpoint=args.norm_training_checkpoint,
        smoke_report=args.norm_smoke_report)
    if sha(args.initialization_dir/'manifest.json') != small_overlay.INITIAL_MANIFEST_SHA256:
        raise ValueError('Norm initialization manifest differs')
    exported = torch.load(args.overlay_dir/'other_fp16.pt', map_location='cpu', weights_only=True)
    if len(exported) != 393 or sum(value.numel() for value in exported.values()) != 3580928:
        raise ValueError('Expected exactly393 existing small FP16 tensors')
    rounding = audit_checkpoint_export(args.training_checkpoint, exported, overlay['binding'],
                                      overlay['training_receipt']['checkpoint_sha256'])
    input_files = {'parent_manifest': args.parent_dir/'manifest.json',
        'norm_manifest': args.initialization_dir/'manifest.json', 'small_manifest': args.overlay_dir/'manifest.json',
        'small_export': args.overlay_dir/'other_fp16.pt', 'training_report': args.training_report,
        'training_checkpoint': args.training_checkpoint, 'smoke_report': args.smoke_report,
        'training_data_manifest': args.data_dir/'manifest.json', 'training_tokens': args.data_dir/'training_tokens.pt',
        'protocol': args.protocol, 'continuation_note': args.continuation_note,
        'reference_report': args.reference_report, 'evaluator': Path(__file__)}
    identities = {key: {'path': str(path.resolve()), 'bytes': path.stat().st_size,
                        'sha256': sha(path)} for key, path in input_files.items()}
    report = {'format': 'MAMBA2_SMALL_COMPENSATION_FULL_VALIDATION_V1', 'complete': False,
        'mode': 'cpu_integrity_only' if args.verify_only else 'four_arm_full_validation',
        'pid': os.getpid(), 'script_sha256': sha(__file__), 'inputs': identities,
        'small_overlay_receipt': receipt, 'norm_overlay_receipt': norm_receipt,
        'independent_checkpoint_export_audit': rounding,
        'frozen_code_sha256': frozen, 'small_overlay_source_sha256': sha(small_overlay.__file__),
        'loaded_tensor_audit_source_sha256': sha(ROOT/'scripts/audit_decoded.py'),
        'arm_order': list(ARMS), 'same_process': True,
        'scope': {'training_changed': False, 'final_successful_updates': 1024,
            'checkpoint_selection_performed': False, 'test_split_used': False,
            'MK_evaluated': False, 'MK_is_continuation_gate': False, 'publication_authorized': False},
        'historical_MK_not_rerun': previous['prior_MK_not_rerun'],
        'limitations': ['Validation has already informed development; it is not untouched test evidence.',
            'PPL does not establish recall preservation or repair the prior MK regression.',
            'Original source is a declared reference input, not a compressed candidate inference dependency.',
            'Evaluation expands weights to FP16; this is not compressed runtime residency.',
            'Recorded PPL evaluation timings are run diagnostics, not a throughput benchmark.',
            'No old Huffman container size is claimed for changed FP16 values.'],
        'storage': {'logical_candidate_data_bytes': receipt['logical_candidate_data_bytes'],
            'actual_small_archive_bytes': (args.overlay_dir/'other_fp16.pt').stat().st_size,
            'actual_small_manifest_bytes': (args.overlay_dir/'manifest.json').stat().st_size,
            'small_fp16_tensor_capacity_bytes': sum(value.numel()*value.element_size() for value in exported.values()),
            'physical_parent_plus_both_overlay_directories_bytes': sum(path.stat().st_size
                for directory in (args.parent_dir, args.initialization_dir, args.overlay_dir) for path in directory.iterdir()),
            'scope': 'Raw model artifacts and manifests only; source, tokenizer, software, reports, calibration, training and optimizer files excluded.'},
        'results': {}}
    write_json(args.report, report)
    if args.verify_only:
        report.update(complete=True, elapsed_seconds=time.perf_counter()-started)
        write_json(args.report, report)
        print(json.dumps({'complete': True, 'mode': report['mode'], 'report_sha256': sha(args.report)}), flush=True)
        return
    report['environment'] = environment_receipt()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', WIKITEXT_REVISION)
    if dataset != previous['dataset'] or len(ids) != TARGETS+1:
        raise ValueError('Validation dataset/token stream differs')
    windows = ppl_windows(ids, 2048, None)
    plan = window_plan(windows)
    if plan != previous['window_plan']:
        raise ValueError('Complete validation windows differ from fixed reference')
    report.update(dataset=dataset, window_plan=plan,
        window_plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        protocol={'split': 'validation', 'target_tokens': TARGETS, 'maximum_targets_per_window': 2048,
            'number_of_windows': len(plan), 'execution': 'prefill', 'logits_chunk_tokens': 64,
            'loss': 'unchanged evaluation.py: FP16 head GEMM followed by FP32 summed cross entropy',
            'state': 'fresh zero state each window; native SSD scan internal precision',
            'automatic_special_tokens': False})
    write_json(args.report, report)
    torch.cuda.reset_peak_memory_stats()

    def score(name, model):
        print(f'[all-small evaluation] Scoring {name}', flush=True)
        result = evaluate_ppl(model, windows, execution='prefill', logits_chunk=64)
        check_arm(result, plan)
        report['results'][name] = result
        write_json(args.report, report)

    print('[all-small evaluation] Loading original FP16 source', flush=True)
    source = load_source_model(args.source_dir)
    report['source_coverage'] = audit_parameter_coverage(source)
    report['source_package_receipt'] = source._package_receipt
    score('source_fp16', source)
    del source
    gc.collect()
    torch.cuda.empty_cache()
    print('[all-small evaluation] Loading and independently auditing E8/W5 parent', flush=True)
    model = load_quantized_model(args.parent_dir)
    report['parent_package_receipt'] = model._package_receipt
    parent_audit = audit_loaded_model(model, parent, args.parent_dir)
    report['parent_loaded_tensor_audit'] = parent_audit
    frozen_hashes = {entry['name']: entry['decoded_fp16_sha256'] for entry in parent_audit['tensors']
                     if entry['family'] in ('e8', 'w5')}
    if len(frozen_hashes) != 114:
        raise ValueError('Frozen projection/vocabulary coverage differs')
    score('parent_e8w5', model)
    print('[all-small evaluation] Reloading norm-v1 FP16 export', flush=True)
    report['norm_overlay_application'] = norm_overlay.apply_overlay(model, args.initialization_dir, norm)
    norm_values = torch.load(args.initialization_dir/'other_fp16.pt', map_location='cpu', weights_only=True)
    report['norm_loaded_tensor_audit'] = {'coverage': audit_parameter_coverage(model),
        'small': audit_selected_values(model, norm_values), 'frozen': audit_frozen_values(model, frozen_hashes)}
    score('norm_e8w5', model)
    print('[all-small evaluation] Reloading final1024 all-small FP16 export', flush=True)
    model = small_overlay.apply_overlay(model, args.overlay_dir, overlay)
    report['small_overlay_application'] = model._small_overlay_receipt
    report['small_loaded_tensor_audit'] = {'coverage': audit_parameter_coverage(model),
        'small': audit_selected_values(model, exported), 'frozen': audit_frozen_values(model, frozen_hashes)}
    score('small_e8w5', model)
    report['comparison'], report['paired_windows'] = comparisons(report['results'], plan)
    for key, path in input_files.items():
        if sha(path) != identities[key]['sha256']:
            raise ValueError(f'Audited input changed during evaluation: {key}')
    report.update(complete=True, elapsed_seconds=time.perf_counter()-started, gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    print(json.dumps({'complete': True, 'comparison': report['comparison'], 'report_sha256': sha(args.report)}), flush=True)


if __name__ == '__main__':
    main()
