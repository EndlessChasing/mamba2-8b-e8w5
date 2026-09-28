#!/usr/bin/env python3
"""One-window execution probe for the preserved output-calibration CE failure.

The four actions are fixed. CE differences are recorded exactly, never accepted
under a relaxed tolerance. No fitting, temperature change or quality decision.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import diagnose_output_calibration as single
import diagnose_low_rank_generalization as paired

native = single.native
SINGLE_SHA = '116c69a5151a2497209408ed4fef7561fcae38624b20e6bdf659b40b3126fd64'
PAIRED_SHA = 'e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879'
FAILED_REPORT_SHA = '38c31d728bc44dba9882b3e2344aeae1feec746a48d500dcbe27264b6b3a374c'
FAILED_PROCESS_SHA = '2b311d9d729e176b9d2a03d04277cebce4ce8d36300ad6a259cfa78c03f77aee'
ACTIONS = ['new_source_alone', 'old_paired_after_base_load',
           'new_source_then_base', 'old_paired_repeated']


def tensor_layout(value):
    return {'shape': list(value.shape), 'stride': list(value.stride()),
        'storage_offset': value.storage_offset(), 'dtype': str(value.dtype),
        'device': str(value.device), 'contiguous': value.is_contiguous()}


@contextmanager
def trace_layouts(models):
    """Record Python metadata only; no extra tensor math, copies or scalar reads."""
    trace = {'head_calls': [], 'cross_entropy_calls': [],
        'scope': 'Read-only shape/stride metadata hooks; no tensor values or synchronization added.'}
    handles = []
    original_ce = F.cross_entropy

    def head_hook(name):
        def hook(_module, args, result):
            trace['head_calls'].append({'model': name, 'input': tensor_layout(args[0]),
                'output': tensor_layout(result)})
        return hook

    def ce_hook(logits, target, *args, **kwargs):
        # Both frozen scorers supply reduction by keyword; reject another signature.
        if args:
            raise ValueError('Unexpected positional CE arguments in fixed scorers')
        trace['cross_entropy_calls'].append({'logits': tensor_layout(logits),
            'targets': tensor_layout(target), 'reduction': kwargs.get('reduction', 'mean')})
        return original_ce(logits, target, **kwargs)

    try:
        for name, model in models.items():
            handles.append(model.lm_head.register_forward_hook(head_hook(name)))
        F.cross_entropy = ce_hook
        yield trace
    finally:
        F.cross_entropy = original_ce
        for handle in handles:
            handle.remove()


def ce_record(chunks, nll, count):
    if len(chunks) != 32 or count != 2047 or not all(math.isfinite(v) for v in chunks):
        raise ValueError('Fixed window requires31x64+63 targets and32 finite CE chunks')
    if not single.equal_doubles([math.fsum(chunks)], [nll]):
        raise ValueError('Recorded CE sum does not match its chunk values')
    return {'target_tokens': count, 'chunk_target_counts': [64]*31+[63],
        'ce_chunk_sums': chunks, 'nll': nll, 'mean_nll': nll/count,
        'ppl': math.exp(nll/count)}


def compare(actual, reference):
    first, second = actual['ce_chunk_sums'], reference['ce_chunk_sums']
    exact = [single.equal_doubles([a], [b]) for a, b in zip(first, second)]
    deltas = [a-b for a, b in zip(first, second)]
    return {'target_tokens_equal': actual['target_tokens'] == reference['target_tokens'],
        'chunk_bitwise_equal': exact, 'all_chunks_bitwise_equal': all(exact),
        'different_chunks': [i for i, value in enumerate(exact) if not value],
        'actual_minus_reference_chunk_nll': deltas,
        'maximum_absolute_chunk_difference': max(abs(v) for v in deltas),
        'first_chunk_difference': deltas[0],
        'total_nll_bitwise_equal': single.equal_doubles([actual['nll']], [reference['nll']]),
        'actual_minus_reference_nll': actual['nll']-reference['nll'],
        'tolerance_used': False}


def run(args, report):
    pins = {ROOT/'scripts/diagnose_output_calibration.py': SINGLE_SHA,
        ROOT/'scripts/diagnose_low_rank_generalization.py': PAIRED_SHA,
        ROOT/'reports/output_calibration_v1.json': FAILED_REPORT_SHA,
        ROOT/'reports/output_calibration_v1_process.json': FAILED_PROCESS_SHA}
    for path, digest in pins.items():
        if native.sha(path) != digest:
            raise ValueError(f'Preserved failed input/scorer changed: {path}')
    failed = json.loads((ROOT/'reports/output_calibration_v1.json').read_text())
    if (failed.get('complete') is not False
            or failed.get('error') != 'Original beta1 CE does not reproduce exactly: source_fp16/0'):
        raise ValueError('Not the fixed source-window0 execution failure')
    prior, _manifest, data, windows, plans, base_hashes, _candidate, checked, integrity = single.verify_inputs()
    checked.update({str(path): digest for path, digest in pins.items()})
    checked[str(Path(__file__).resolve())] = native.sha(__file__)
    window, plan = windows[0], plans[0]
    if window.shape != (2048,) or plan['index'] != 0 or plan['target_tokens'] != 2047:
        raise ValueError('Expected only observed reserved window0')
    old_base = prior['reserved_results']['best_small_e8w5'][0]
    old_candidate = prior['reserved_results']['teacher_kl_e8w5'][0]
    if not single.equal_doubles(old_base['teacher_ce_chunk_sums'], old_candidate['teacher_ce_chunk_sums']):
        raise ValueError('Previously paired source controls differ')
    anchors = {
        'source_fp16': ce_record(old_base['teacher_ce_chunk_sums'], old_base['source_nll'], 2047),
        'best_small_e8w5': ce_record(old_base['student_ce_chunk_sums'], old_base['student_nll'], 2047)}
    report.update(binding={'prior_evaluation_sha256': single.EVALUATION_SHA,
        'failed_diagnostic_sha256': FAILED_REPORT_SHA, 'checked_input_sha256': checked},
        verified_fixed_export=integrity, dataset=data['dataset'], window=plan,
        window_cpu_layout=tensor_layout(window), prior_anchors=anchors, actions={}, measurements={})
    native.write_json(args.report, report)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    report['environment'] = native.environment_receipt()
    source = native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
    if source._package_receipt != prior['source_package_receipt']:
        raise ValueError('Source loader/checkpoint differs from measured endpoint')
    report['source_package_receipt'] = source._package_receipt
    report['source_coverage'] = native.audit_parameter_coverage(source)
    source_hashes = {name: native.tensor_sha_fp16(value) for name, value in source.named_parameters()}
    if len(source_hashes) != 507:
        raise ValueError('Expected507 source tensors')
    report['source507_before_sha256'] = source_hashes
    source_identity = single.model_identity(source)

    def save_single(action, name, model, models):
        with trace_layouts(models) as trace:
            row, _ce = single.score_window(model, window, chunk_tokens=64)
        item = ce_record(row['ce_chunk_sums'], row['nll'], row['target_tokens'])
        key = action+'/'+name
        report['measurements'][key] = {'model': name, **item}
        report['actions'].setdefault(action, []).append({'model': name, 'measurement_key': key,
            'layouts': trace, 'gpu_memory': native.gpu_memory_receipt()})
        native.write_json(args.report, report)

    def save_paired(action, base):
        with trace_layouts({'source_fp16': source, 'best_small_e8w5': base}) as trace:
            row = paired.score_window(source, base, window, chunk_tokens=64)
        keys = []
        for name, chunks, nll in [('source_fp16', row['teacher_ce_chunk_sums'], row['source_nll']),
                                  ('best_small_e8w5', row['student_ce_chunk_sums'], row['student_nll'])]:
            key = action+'/'+name
            report['measurements'][key] = {'model': name, **ce_record(chunks, nll, row['target_tokens'])}
            keys.append(key)
        report['actions'][action] = {'measurement_keys': keys, 'layouts': trace,
            'gpu_memory': native.gpu_memory_receipt()}
        native.write_json(args.report, report)

    print('[anchor probe]1/4 new source scorer, source alone', flush=True)
    save_single(ACTIONS[0], 'source_fp16', source, {'source_fp16': source})
    report['source507_after_alone_audit'] = native.audit_frozen_values(source, source_hashes)
    model = native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    small = json.loads((ROOT/'artifacts/small_compensation_v1/manifest.json').read_text())
    native.small_overlay.apply_overlay(model, ROOT/'artifacts/small_compensation_v1', small)
    model.eval().requires_grad_(False)
    report['baseline507_before_audit'] = native.audit_frozen_values(model, base_hashes)
    report['baseline_coverage'] = native.audit_parameter_coverage(model)
    base_identity = single.model_identity(model)
    print('[anchor probe]2/4 old frozen paired scorer after base loading/decoding', flush=True)
    save_paired(ACTIONS[1], model)
    print('[anchor probe]3/4 new source scorer then new base scorer', flush=True)
    models = {'source_fp16': source, 'best_small_e8w5': model}
    save_single(ACTIONS[2], 'source_fp16', source, models)
    save_single(ACTIONS[2], 'best_small_e8w5', model, models)
    print('[anchor probe]4/4 repeat old frozen paired scorer', flush=True)
    save_paired(ACTIONS[3], model)
    if single.model_identity(source) != source_identity or single.model_identity(model) != base_identity:
        raise ValueError('Probe changed parameter objects/versions')
    report['source507_after_audit'] = native.audit_frozen_values(source, source_hashes)
    report['baseline507_after_audit'] = native.audit_frozen_values(model, base_hashes)
    report['comparisons_with_prior'] = {key: compare(value, anchors[value['model']])
        for key, value in report['measurements'].items()}
    pairs = {}
    values = list(report['measurements'].items())
    for i, (first_name, first) in enumerate(values):
        for second_name, second in values[i+1:]:
            if first['model'] == second['model']:
                pairs[second_name+' minus '+first_name] = compare(second, first)
    report['pairwise_same_model_comparisons'] = pairs
    for path, digest in checked.items():
        if native.sha(path) != digest:
            raise ValueError(f'Preserved bound input changed: {path}')
    report.update(complete=True, gpu_memory=native.gpu_memory_receipt(),
        all_prior_ce_anchors_bitwise_equal=all(v['all_chunks_bitwise_equal']
            and v['total_nll_bitwise_equal'] for v in report['comparisons_with_prior'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report = args.report.resolve()
    if args.report.parent != ROOT/'reports':
        raise ValueError('Probe output must be a new report directly under reports/')
    if args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(8)
    started = time.perf_counter()
    report = {'format': 'MAMBA2_OUTPUT_CALIBRATION_ANCHOR_PROBE_V1', 'complete': False,
        'posthoc': True, 'window_indices': [0], 'actions_in_order': ACTIONS,
        'script_sha256': native.sha(__file__), 'inverse_temperature_beta': 1.,
        'training': False, 'temperature_fitting': False, 'temperature_sweep': False,
        'candidate_advancement': False, 'quality_gate': None, 'validation_or_test_computation': False,
        'limitations': ['Fixed one-window execution probe; CE discrepancies are recorded, never accepted under a tolerance.',
            'The original failed diagnostic remains failed and immutable.',
            'Action order changes execution/allocator history; outcomes describe these paths, not a proven causal mechanism.',
            'Metadata-only hooks record native tensor shape/stride/dtype; no additional tensor reductions in hooks.',
            'No final candidate is evaluated here; only the fixed source and accepted all-small endpoint.']}
    try:
        run(args, report)
    except BaseException as error:
        report.update(complete=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        native.write_json(args.report, report)
    print(json.dumps({'complete': True, 'report': str(args.report), 'sha256': native.sha(args.report),
        'all_prior_ce_anchors_bitwise_equal': report['all_prior_ce_anchors_bitwise_equal'],
        'comparison_summary': {key: {k: row[k] for k in ('different_chunks', 'first_chunk_difference',
            'maximum_absolute_chunk_difference', 'actual_minus_reference_nll')}
            for key, row in report['comparisons_with_prior'].items()}}), flush=True)


if __name__ == '__main__':
    main()
