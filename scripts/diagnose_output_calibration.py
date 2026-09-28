#!/usr/bin/env python3
"""Posthoc beta=1 output diagnostic on the already observed64 TRAIN windows.

Three fixed native models; no fitting, temperature search, model selection,
advancement, validation/test loading or serialized vocabulary logits.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_teacher_kl_compensation as native

EVALUATION_SHA = 'd8acf27dfa500d18a6d4b25dc6d95e3edaa10a0ccc27f49369cb02ef2649e59a'
MANIFEST_SHA = 'cd1c260c3e92146618322431d845101ea7b84468c5fa759769f843a899445300'
EVALUATOR_SHA = 'b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071'
DOCUMENT_SHA = 'baa87fa1549e29bf757cca8e18f1ab0feadc18daebffa79402c75c87035bc05d'
ARMS = ('source_fp16', 'best_small_e8w5', 'teacher_kl_e8w5')
RANK_BINS = ((1, 1), (2, 5), (6, 10), (11, 100), (101, 1000), (1001, None))
SURPRISAL_EDGES = (0., 1., 2., 4., 8., 16., math.inf)
PROTOCOL = {
    'posthoc': True, 'split': 'train', 'already_observed_windows': 64,
    'target_tokens': 131008, 'stored_window_tokens': 2048, 'targets_per_window': 2047,
    'fresh_state_per_window': True, 'arm_order': list(ARMS),
    'inverse_temperature_beta': 1., 'temperature_fitting': False, 'temperature_sweep': False,
    'checkpoint_selection': False, 'candidate_advancement': False, 'quality_gate': None,
    'validation_or_test_computation': False, 'logits_chunk_tokens': 64,
    'native_head': 'FP16 GEMM, followed by FP32 logits/loss/moments',
    'ce_anchor': 'unchanged FP32 F.cross_entropy(reduction=sum), each chunk byte-equal to prior',
    'entropy': '-sum_v p_v log(p_v)',
    'beta_gradient': 'per-target CE - predictive entropy; independent centered-logit expectation check',
    'beta_curvature': 'sum_v p_v*(log(p_v)-E_p[log(p)])**2',
    'gradient_identity_atol': 0.0003, 'gradient_identity_rtol': 0.00003,
    'ce_reduction_atol': 0.0002, 'ce_reduction_rtol': 0.000003,
    'gradient_sign_counts': 'strict FP32 sign; descriptive numerical near-zero count abs(g)<=1e-5 also recorded',
    'rank_definition': 'rank_lo=1+count(z>z_y); rank_hi=count(z>=z_y); ties retained',
    'top1_correct': 'actual argmax token index equals observed target; first-index tie handling',
    'rank_bins': ['1', '2..5', '6..10', '11..100', '101..1000', '>1000'],
    'source_surprisal_bins_nats': ['[0,1)', '[1,2)', '[2,4)', '[4,8)', '[8,16)', '[16,inf)'],
    'source_bin_selection': 'fixed prior to outcomes; source per-target CE only',
    'precision': 'TF32 off; highest FP32 matmul precision; autocast disabled',
    'serialization': 'per-window/chunk summaries and per-target hashes only; no logits or per-target arrays',
}


def protocol_sha():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def equal_doubles(first, second):
    return len(first) == len(second) and all(struct.pack('<d', x) == struct.pack('<d', y)
        for x, y in zip(first, second))


def array_sha(value):
    value = value.detach().contiguous().cpu()
    kind = '<i8' if value.dtype == torch.int64 else '<f8'
    return hashlib.sha256(value.numpy().astype(kind, copy=False).tobytes()).hexdigest()


def output_statistics(logits, targets):
    """Analytic derivatives at exactly beta1; no autograd or beta optimization."""
    if (logits.ndim != 2 or targets.shape != logits.shape[:1] or targets.dtype != torch.int64
            or logits.dtype not in (torch.float32, torch.float64) or logits.device != targets.device
            or targets.numel() == 0 or not torch.isfinite(logits).all()):
        raise ValueError('Invalid finite native logit/target geometry')
    logp = F.log_softmax(logits, dim=-1)
    probability = logp.exp()
    ce = F.cross_entropy(logits, targets, reduction='none')
    mean_logp = (probability * logp).sum(-1)
    entropy = -mean_logp
    gradient = ce - entropy
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    target_shifted = shifted.gather(1, targets[:, None]).squeeze(1)
    independent = (probability * shifted).sum(-1) - target_shifted
    curvature = (probability * (logp - mean_logp[:, None]).square()).sum(-1)
    if (not all(torch.isfinite(value).all() for value in (ce, entropy, gradient, independent, curvature))
            or torch.any(ce < 0) or torch.any(entropy < 0) or torch.any(curvature < 0)):
        raise FloatingPointError('Invalid beta1 moments')
    if not torch.allclose(gradient, independent,
            atol=PROTOCOL['gradient_identity_atol'], rtol=PROTOCOL['gradient_identity_rtol']):
        raise ValueError('CE-H and independent centered-logit derivative disagree')
    target_logits = logits.gather(1, targets[:, None])
    rank_lo = 1 + (logits > target_logits).sum(-1)
    rank_hi = (logits >= target_logits).sum(-1)
    correct = (logits.argmax(-1) == targets).to(torch.int64)
    if torch.any(rank_lo > rank_hi) or torch.any(rank_hi > logits.shape[1]):
        raise ValueError('Invalid strict/tied target rank')
    return {'ce': ce, 'entropy': entropy, 'gradient': gradient, 'curvature': curvature,
        'independent_gradient': independent, 'rank_lo': rank_lo, 'rank_hi': rank_hi,
        'top1_correct': correct,
        'probability_sum_error': (probability.sum(-1) - 1).abs()}


def rank_counts(ranks):
    return {name: int(((ranks >= lo) & (ranks <= hi if hi is not None else True)).sum())
        for name, (lo, hi) in zip(PROTOCOL['rank_bins'], RANK_BINS)}


def summarize_vectors(vectors, ce_chunks):
    count = vectors['ce'].numel()
    sums = {key: float(vectors[key].double().sum())
        for key in ('ce', 'entropy', 'gradient', 'curvature', 'independent_gradient')}
    g = vectors['gradient']
    lower, upper = vectors['rank_lo'], vectors['rank_hi']
    nll = math.fsum(ce_chunks)
    if not math.isclose(nll, sums['ce'],
            abs_tol=PROTOCOL['ce_reduction_atol'] * len(ce_chunks), rel_tol=PROTOCOL['ce_reduction_rtol']):
        raise ValueError('Per-target CE and unchanged chunk-reduced CE disagree')
    return {'target_tokens': count, 'nll': nll, 'mean_nll': nll/count, 'ppl': math.exp(nll/count),
        'ce_chunk_sums': ce_chunks, 'per_target_sums': sums,
        'per_target_means': {key: value/count for key, value in sums.items()},
        'per_target_ce_minus_chunk_nll': sums['ce'] - nll,
        'gradient_sign_counts': {'negative': int((g < 0).sum()), 'zero': int((g == 0).sum()),
            'positive': int((g > 0).sum()), 'abs_le_1e_5': int((g.abs() <= 1e-5).sum())},
        'gradient_identity_maximum_absolute_error': float((g-vectors['independent_gradient']).abs().max()),
        'probability_sum_maximum_absolute_error': float(vectors['probability_sum_error'].max()),
        'top1_correct': int(vectors['top1_correct'].sum()),
        'rank_lo_bins': rank_counts(lower), 'rank_hi_bins': rank_counts(upper),
        'target_tied_count': int((upper > lower).sum()),
        'target_tie_count_sum_including_self': int((upper-lower+1).sum()),
        'target_in_top_tie_count': int(((lower == 1) & (upper > 1)).sum()),
        'target_is_maximum_count': int((lower == 1).sum()),
        'largest_target_tie_size': int((upper-lower+1).max()),
        'per_target_statistic_sha256': {key: array_sha(value) for key, value in vectors.items()},
        'per_target_hash_encoding': 'CPU contiguous little-endian float64 for moments, int64 for ranks/correctness'}


@torch.inference_mode()
def score_window(model, window, chunk_tokens=64):
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError('Positive integer chunk size required')
    device = model.backbone.embedding.weight.device
    ids, targets = window[None, :-1].to(device), window[1:].to(device)
    if ids.shape[1] == 0:
        raise ValueError('At least one next-token target required')
    chunks, collected = [], {}
    with torch.autocast(device_type=device.type, enabled=False):
        hidden = model.backbone(ids)
        if hidden.dtype != torch.float16 or not torch.isfinite(hidden).all():
            raise FloatingPointError('Expected finite native FP16 hidden states')
        for first in range(0, targets.numel(), chunk_tokens):
            end = min(first+chunk_tokens, targets.numel())
            logits16 = model.lm_head(hidden[:, first:end])
            if logits16.dtype != torch.float16:
                raise ValueError('Head GEMM must remain native FP16')
            logits = logits16.float().flatten(0, 1)
            truth = targets[first:end]
            # Preserve exactly the historical loss kernel/reduction, before new moment operations.
            chunks.append(float(F.cross_entropy(logits, truth, reduction='sum')))
            moments = output_statistics(logits, truth)
            for key, value in moments.items():
                collected.setdefault(key, []).append(value.detach().cpu())
            del logits16, logits, moments, truth
    vectors = {key: torch.cat(values) for key, values in collected.items()}
    summary = summarize_vectors(vectors, chunks)
    summary['chunk_target_counts'] = [min(chunk_tokens, targets.numel()-first)
        for first in range(0, targets.numel(), chunk_tokens)]
    # Only CE is retained across model passes, for paired source-surprisal strata.
    return summary, vectors['ce'].double()


def aggregate(rows):
    count = sum(row['target_tokens'] for row in rows)
    nll = math.fsum(row['nll'] for row in rows)
    sums = {key: math.fsum(row['per_target_sums'][key] for row in rows)
        for key in ('ce', 'entropy', 'gradient', 'curvature', 'independent_gradient')}
    result = {'windows': len(rows), 'target_tokens': count, 'nll': nll,
        'mean_nll': nll/count, 'ppl': math.exp(nll/count), 'per_target_sums': sums,
        'per_target_means': {key: value/count for key, value in sums.items()},
        'gradient_sign_counts': {key: sum(row['gradient_sign_counts'][key] for row in rows)
            for key in ('negative', 'zero', 'positive', 'abs_le_1e_5')},
        'window_mean_gradient_sign_counts': {
            'negative': sum(row['per_target_sums']['gradient'] < 0 for row in rows),
            'zero': sum(row['per_target_sums']['gradient'] == 0 for row in rows),
            'positive': sum(row['per_target_sums']['gradient'] > 0 for row in rows)},
        'gradient_identity_maximum_absolute_error': max(row['gradient_identity_maximum_absolute_error'] for row in rows),
        'probability_sum_maximum_absolute_error': max(row['probability_sum_maximum_absolute_error'] for row in rows),
        'top1_correct': sum(row['top1_correct'] for row in rows),
        'rank_lo_bins': {key: sum(row['rank_lo_bins'][key] for row in rows) for key in PROTOCOL['rank_bins']},
        'rank_hi_bins': {key: sum(row['rank_hi_bins'][key] for row in rows) for key in PROTOCOL['rank_bins']},
        'target_tied_count': sum(row['target_tied_count'] for row in rows),
        'target_tie_count_sum_including_self': sum(row['target_tie_count_sum_including_self'] for row in rows),
        'target_in_top_tie_count': sum(row['target_in_top_tie_count'] for row in rows),
        'target_is_maximum_count': sum(row['target_is_maximum_count'] for row in rows),
        'largest_target_tie_size': max(row['largest_target_tie_size'] for row in rows)}
    result['top1_accuracy'] = result['top1_correct']/count
    if (sum(result['rank_lo_bins'].values()) != count or sum(result['rank_hi_bins'].values()) != count
            or sum(result['gradient_sign_counts'][key] for key in ('negative', 'zero', 'positive')) != count):
        raise ValueError('Rank/sign counts do not cover every target')
    return result


def surprisal_bins(ce_by_arm):
    source, base, candidate = (ce_by_arm[name] for name in ARMS)
    if source.ndim != 1 or base.shape != source.shape or candidate.shape != source.shape:
        raise ValueError('Paired per-target arrays differ')
    rows = []
    for name, lo, hi in zip(PROTOCOL['source_surprisal_bins_nats'], SURPRISAL_EDGES, SURPRISAL_EDGES[1:]):
        mask = (source >= lo) & (source < hi)
        count = int(mask.sum())
        totals = {arm: float(value[mask].double().sum()) for arm, value in ce_by_arm.items()}
        delta = candidate[mask]-base[mask]
        rows.append({'source_surprisal_interval_nats': name, 'lower_inclusive': lo,
            'upper_exclusive': hi if math.isfinite(hi) else None, 'target_tokens': count,
            'per_target_nll_sum': totals,
            'per_target_mean_nll': {arm: value/count if count else None for arm, value in totals.items()},
            'candidate_minus_baseline_nll_sum': float(delta.sum()),
            'candidate_minus_baseline_mean_nll': float(delta.sum())/count if count else None,
            'target_nll_improved': int((delta < 0).sum()), 'target_nll_unchanged': int((delta == 0).sum()),
            'target_nll_worsened': int((delta > 0).sum())})
    if sum(row['target_tokens'] for row in rows) != source.numel():
        raise ValueError('Fixed source bins omit targets')
    for arm, values in ce_by_arm.items():
        if not math.isclose(math.fsum(row['per_target_nll_sum'][arm] for row in rows),
                float(values.double().sum()), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('Source-bin NLL totals differ from unbinned target sums')
    if not math.isclose(math.fsum(row['candidate_minus_baseline_nll_sum'] for row in rows),
            float((candidate-base).double().sum()), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError('Source-bin paired deltas differ from unbinned target sum')
    return rows


def verify_inputs():
    evaluation = ROOT/'reports/teacher_kl_compensation_v1_eval.json'
    folder = ROOT/'artifacts/teacher_kl_compensation_v1'
    document = ROOT/'docs/OUTPUT_CALIBRATION_DIAGNOSTIC_PROTOCOL.md'
    if (native.sha(evaluation) != EVALUATION_SHA or native.sha(folder/'manifest.json') != MANIFEST_SHA
            or native.sha(ROOT/'scripts/evaluate_teacher_kl_compensation.py') != EVALUATOR_SHA
            or native.sha(document) != DOCUMENT_SHA):
        raise ValueError('Fixed evaluated model/provenance changed')
    prior = json.loads(evaluation.read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    if (prior.get('complete') is not True or prior.get('script_sha256') != EVALUATOR_SHA
            or prior['integrity']['teacher_kl_manifest_sha256'] != MANIFEST_SHA
            or manifest.get('complete') is not True or manifest['binding'] != prior['integrity']['binding']
            or manifest['training_receipt'] != prior['integrity']['training_receipt']
            or prior['reserved_gate']['passed'] is not False or prior['full_validation']['performed'] is not False):
        raise ValueError('Completed negative experiment identity differs')
    old_roots = {Path(name).parents[1] for name, digest in prior['integrity']['checked_input_sha256'].items()
        if Path(name).name == 'evaluate_teacher_kl_compensation.py' and digest == EVALUATOR_SHA}
    if len(old_roots) != 1:
        raise ValueError('Cannot resolve original repository root')
    old_root = next(iter(old_roots))
    checked = {}
    for name, digest in prior['integrity']['checked_input_sha256'].items():
        path = ROOT/Path(name).relative_to(old_root)
        if path.is_symlink() or not path.is_file() or native.sha(path) != digest:
            raise ValueError(f'Bound evaluated input changed: {path}')
        checked[str(path)] = digest
    checked.update({str(evaluation): EVALUATION_SHA, str(document): DOCUMENT_SHA,
        str(Path(__file__).resolve()): native.sha(__file__)})
    if (manifest['factors_inventory'] != native.expected_inventory()
            or {p.name for p in folder.iterdir()} != set(manifest['files'])|{'manifest.json'}):
        raise ValueError('Unexpected strict factor inventory')
    # Reuse the frozen CPU final224-master/strict112-file verifier, without any bank/trainer forward.
    verify_args = argparse.Namespace(protocol=ROOT/'docs/TEACHER_KL_COMPENSATION_PROTOCOL.md',
        baseline_report=ROOT/'reports/small_compensation_v1_eval.json',
        parent_dir=ROOT/'artifacts/e8w5_v1', small_dir=ROOT/'artifacts/small_compensation_v1',
        initialization_dir=ROOT/'artifacts/norm_compensation_v1',
        data_dir=ROOT/'training_data/teacher_kl_compensation_v1', overlay_dir=folder,
        training_report=ROOT/'reports/teacher_kl_compensation_v1_train.json',
        smoke_report=ROOT/'reports/teacher_kl_compensation_v1_smoke.json',
        training_checkpoint=ROOT/'artifacts/teacher_kl_compensation_v1_train_work/checkpoint_attempt448_step448_final.pt')
    _, _, verified_manifest, _, fresh_receipt, data, heldout = native.verify_export(verify_args)
    if (verified_manifest != manifest or fresh_receipt['independent_final448_factor_audit']
            != prior['integrity']['independent_final448_factor_audit']):
        raise ValueError('Fresh checkpoint/factor verification differs from measured final export')
    plan = [{'index': i, **row, 'target_tokens': row['targets']} for i, row in enumerate(data['heldout_windows'])]
    if plan != prior['reserved_plan'] or len(plan) != 64 or sum(row['target_tokens'] for row in plan) != 131008:
        raise ValueError('Observed reserved token identities differ')
    baseline = prior['baseline507_actual_content_audit']['unchanged_content_sha256']
    candidate = prior['candidate507_final_content_audit']['unchanged_content_sha256']
    if (len(baseline) != 507 or len(candidate) != 507
            or baseline != prior['integrity']['frozen_baseline507_hashes']
            or baseline != manifest['frozen_base_hash_ledger']):
        raise ValueError('Actual measured baseline507 identity differs')
    expected = dict(baseline)
    if set(prior['candidate112_merged_fp16']) != set(native.LABELS):
        raise ValueError('Expected112 previously measured projection hashes')
    for label, entry in native.expected_inventory().items():
        row = prior['candidate112_merged_fp16'][label]
        if row['source_key'] != entry['source_key'] or row['factor_file_sha256'] != manifest['files'][entry['file']]['sha256']:
            raise ValueError('Factor/native mapping differs')
        expected[entry['source_key']] = row['merged_fp16_sha256']
    if expected != candidate:
        raise ValueError('Candidate changed tensors outside its112 projections')
    return prior, manifest, data, heldout, plan, baseline, candidate, checked, fresh_receipt


def model_identity(model):
    return {name: (id(value), value.data_ptr(), value._version) for name, value in model.named_parameters()}


def run(args, report):
    prior, manifest, data, heldout, plan, baseline_hashes, candidate_hashes, checked, integrity = verify_inputs()
    report.update(binding={'prior_evaluation_sha256': EVALUATION_SHA, 'candidate_manifest_sha256': MANIFEST_SHA,
        'native_evaluator_source_sha256': EVALUATOR_SHA, 'protocol_document_sha256': DOCUMENT_SHA,
        'training_binding': manifest['binding'],
        'training_receipt': manifest['training_receipt'], 'checked_input_sha256': checked},
        integrity=integrity, dataset=data['dataset'], window_plan=plan, rows={}, summaries={})
    native.write_json(args.report, report)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    report['environment'] = native.environment_receipt()
    source = native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
    if source._package_receipt != prior['source_package_receipt']:
        raise ValueError('Verified FP16 source checkpoint receipt changed')
    report['source_package_receipt'] = source._package_receipt
    report['source_coverage'] = native.audit_parameter_coverage(source)
    source_hashes = {name: native.tensor_sha_fp16(value) for name, value in source.named_parameters()}
    if len(source_hashes) != 507:
        raise ValueError('Source must expose507 actual FP16 tensors')
    report['source507_initial_fp16_sha256'] = source_hashes
    source_identity = model_identity(source)
    ce_by_arm = {}

    def score_arm(arm, model):
        if model.lm_head.weight.shape[0] != 256000:
            raise ValueError('The full256000-token vocabulary is required')
        rows = report['rows'][arm] = []
        ce_parts = []
        before = model_identity(model)
        for index, (window, row_plan) in enumerate(zip(heldout, plan)):
            row, ce = score_window(model, window)
            old_base = prior['reserved_results']['best_small_e8w5'][index]
            old_candidate = prior['reserved_results']['teacher_kl_e8w5'][index]
            if arm == 'source_fp16':
                old_chunks, old_nll = old_base['teacher_ce_chunk_sums'], old_base['source_nll']
                if not equal_doubles(old_chunks+[old_nll], old_candidate['teacher_ce_chunk_sums']+[old_candidate['source_nll']]):
                    raise ValueError('Previously repeated source CE differs')
            else:
                old = old_base if arm == 'best_small_e8w5' else old_candidate
                old_chunks, old_nll = old['student_ce_chunk_sums'], old['student_nll']
            if (row['target_tokens'] != 2047 or row['chunk_target_counts'] != [64]*31+[63]
                    or not equal_doubles(row['ce_chunk_sums']+[row['nll']], old_chunks+[old_nll])):
                raise ValueError(f'Original beta1 CE does not reproduce exactly: {arm}/{index}')
            row.update(row_plan, prior_ce_chunk_and_total_bitwise_equal=True)
            rows.append(row)
            ce_parts.append(ce)
            if (index+1) % 8 == 0:
                print(f'[output calibration] {arm} {index+1}/64 CE={row["mean_nll"]:.6f}', flush=True)
                native.write_json(args.report, report)
        if model_identity(model) != before:
            raise ValueError('Inference changed parameter objects/versions')
        report['summaries'][arm] = aggregate(rows)
        expected = prior['reserved_summary']
        if (not equal_doubles([report['summaries'][arm]['nll'], report['summaries'][arm]['ppl']],
                [expected['nll'][arm], expected['ppl'][arm]])):
            raise ValueError('Original aggregate CE/PPL does not reproduce exactly')
        ce_by_arm[arm] = torch.cat(ce_parts)
        report['summaries'][arm]['per_target_ce_sha256_float64le'] = array_sha(ce_by_arm[arm])
        native.write_json(args.report, report)

    score_arm('source_fp16', source)
    if model_identity(source) != source_identity:
        raise ValueError('Source identity changed')
    report['source507_post_diagnostic_audit'] = native.audit_frozen_values(source, source_hashes)
    del source
    gc.collect(); torch.cuda.empty_cache()
    model = native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    small = json.loads((ROOT/'artifacts/small_compensation_v1/manifest.json').read_text())
    native.small_overlay.apply_overlay(model, ROOT/'artifacts/small_compensation_v1', small)
    model.eval().requires_grad_(False)
    report['baseline507_actual_audit'] = native.audit_frozen_values(model, baseline_hashes)
    report['baseline_coverage'] = native.audit_parameter_coverage(model)
    score_arm('best_small_e8w5', model)
    report['baseline507_post_diagnostic_audit'] = native.audit_frozen_values(model, baseline_hashes)
    projections = {entry['source_key'] for entry in native.expected_inventory().values()}
    inherited = {name: value for name, value in model.named_parameters() if name not in projections}
    if len(inherited) != 395:
        raise ValueError('Expected395 unchanged native tensors')
    merged = {}
    for label, entry in native.expected_inventory().items():
        path = native.checked_file(ROOT/'artifacts/teacher_kl_compensation_v1', entry['file'], manifest['files'][entry['file']])
        B, A = native.read_factors(path, expected_shape=entry['shape'], expected_rank=4)
        name = entry['source_key']
        base = model.get_parameter(name)
        if native.tensor_sha_fp16(base) != baseline_hashes[name]:
            raise ValueError('Base projection content changed')
        value = native.merge_native_projection(base, B, A)
        digest = native.tensor_sha_fp16(value)
        if digest != candidate_hashes[name]:
            raise ValueError(f'Merge differs from already evaluated final candidate: {label}')
        owner, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(value, requires_grad=False))
        merged[label] = digest
    del value, base, B, A
    gc.collect(); torch.cuda.empty_cache()
    if any(model.get_parameter(name) is not value for name, value in inherited.items()):
        raise ValueError('Merge replaced an inherited parameter')
    report['candidate112_merged_fp16_sha256'] = merged
    report['candidate395_unchanged_audit'] = native.audit_frozen_values(model,
        {name: baseline_hashes[name] for name in inherited})
    report['candidate507_actual_audit'] = native.audit_frozen_values(model, candidate_hashes)
    report['candidate_coverage'] = native.audit_parameter_coverage(model)
    score_arm('teacher_kl_e8w5', model)
    report['candidate507_final_audit'] = native.audit_frozen_values(model, candidate_hashes)
    report['source_surprisal_bins'] = surprisal_bins(ce_by_arm)
    report['paired_windows'] = [{**row, 'candidate_minus_baseline_mean_nll':
        report['rows'][ARMS[2]][i]['mean_nll']-report['rows'][ARMS[1]][i]['mean_nll'],
        'per_target_means': {arm: report['rows'][arm][i]['per_target_means'] for arm in ARMS}}
        for i, row in enumerate(plan)]
    report['per_target_arrays_retained_for_pairing_bytes'] = sum(x.numel()*x.element_size() for x in ce_by_arm.values())
    report['per_target_arrays_serialized'] = False
    for path, digest in checked.items():
        if native.sha(path) != digest:
            raise ValueError(f'Bound diagnostic input changed: {path}')
    report.update(complete=True, all192_window_ce_anchors_bitwise_equal=True,
        gpu_memory=native.gpu_memory_receipt())


def self_test():
    """CPU-only analytic beta derivatives, labels, offset, ties and short tail."""
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise ValueError('Self-test must not initialize CUDA')
    torch.manual_seed(928)
    logits = torch.randn(7, 19, dtype=torch.float64)*1.4
    targets = torch.arange(7, dtype=torch.int64)
    stats = output_statistics(logits, targets)
    first, second = [], []
    for row, target in zip(logits, targets):
        beta = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        ce = F.cross_entropy((beta*row)[None], target[None])
        gradient, = torch.autograd.grad(ce, beta, create_graph=True)
        curvature, = torch.autograd.grad(gradient, beta)
        first.append(gradient.detach()); second.append(curvature.detach())
    torch.testing.assert_close(stats['gradient'], torch.stack(first), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(stats['curvature'], torch.stack(second), atol=1e-12, rtol=1e-12)
    shifted = output_statistics(logits+torch.arange(7, dtype=torch.float64)[:, None]*100., targets)
    for key in ('ce', 'entropy', 'gradient', 'curvature', 'independent_gradient'):
        torch.testing.assert_close(stats[key], shifted[key], atol=2e-12, rtol=2e-12)
    for key in ('rank_lo', 'rank_hi', 'top1_correct'):
        assert torch.equal(stats[key], shifted[key])
    altered = output_statistics(logits, targets.flip(0))
    assert not torch.equal(stats['gradient'], altered['gradient'])
    assert torch.equal(stats['entropy'], altered['entropy']) and torch.equal(stats['curvature'], altered['curvature'])
    ties = output_statistics(torch.tensor([[2.,2.,0.],[0.,2.,2.]]), torch.tensor([1,1]))
    assert ties['rank_lo'].tolist() == [1,1] and ties['rank_hi'].tolist() == [2,2]
    assert ties['top1_correct'].tolist() == [0,1]
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.embedding = torch.nn.Embedding(19,4,dtype=torch.float16)
            self.backbone.forward = self.backbone.embedding.forward
            self.lm_head = torch.nn.Linear(4,19,bias=False,dtype=torch.float16)
    toy = Toy().eval().requires_grad_(False)
    row, ce = score_window(toy, torch.arange(8), chunk_tokens=3)
    assert row['chunk_target_counts'] == [3,3,1] and row['target_tokens'] == 7
    with torch.inference_mode():
        dense = toy.lm_head(toy.backbone(torch.arange(7)[None])).float().flatten(0,1)
        direct = output_statistics(dense, torch.arange(1,8))
    torch.testing.assert_close(ce, direct['ce'].double(), atol=0, rtol=0)
    for key in ('ce','entropy','gradient','curvature'):
        assert math.isclose(row['per_target_means'][key], float(direct[key].double().mean()), rel_tol=1e-7, abs_tol=1e-7)
    assert aggregate([row])['target_tokens'] == 7
    sample = torch.tensor([0.,.5,1.,2.,4.,8.,16.,20.],dtype=torch.float64)
    bins = surprisal_bins(dict(zip(ARMS,(sample,sample+.5,sample+.25))))
    assert [r['target_tokens'] for r in bins] == [2,1,1,1,1,2]
    assert all(r['candidate_minus_baseline_mean_nll'] == -.25 for r in bins)
    return {'complete': True, 'passed': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'checks': ['beta1 first/second derivatives versus CPU float64 autograd',
            'per-row logit offset invariance', 'label-dependent gradient and label-independent entropy/curvature',
            'strict and tied target rank versus actual argmax', 'native FP16 head short3+3+1 target-weighted statistics',
            'fixed source-surprisal bin boundaries and paired target means'],
        'script_sha256': native.sha(__file__), 'diagnostic_protocol_sha256': protocol_sha(),
        'protocol_document_sha256': DOCUMENT_SHA,
        'actual_fp32_gradient_identity_tolerance': {
            'atol': PROTOCOL['gradient_identity_atol'], 'rtol': PROTOCOL['gradient_identity_rtol']},
        'actual_fp32_ce_reduction_tolerance': {
            'atol_per_chunk': PROTOCOL['ce_reduction_atol'], 'rtol': PROTOCOL['ce_reduction_rtol']},
        'toy_derivative_tolerance': {'atol': 1e-12, 'rtol': 1e-12}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.report is not None:
            parser.error('Self-test prints its receipt; do not combine with --report')
        print(json.dumps(self_test(), indent=2)); return
    if args.report is None:
        parser.error('--report is required')
    args.report = args.report.resolve()
    if args.report.parent != ROOT/'reports':
        raise ValueError('New diagnostic report must be a direct child of repository reports/')
    if args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(8)
    started = time.perf_counter()
    report = {'format': 'MAMBA2_OUTPUT_CALIBRATION_DIAGNOSTIC_V1', 'complete': False,
        'posthoc': True, 'candidate_advancement': False, 'temperature_fitting': False,
        'temperature_sweep': False, 'quality_gate': None, 'validation_or_test_computation': False,
        'script_sha256': native.sha(__file__), 'protocol': PROTOCOL, 'protocol_sha256': protocol_sha(),
        'limitations': ['Already observed TRAIN windows; descriptive diagnostic, not new heldout evidence.',
            'Gradient/curvature concern only global positive inverse temperature at beta1; no causal attribution.',
            'Positive temperature cannot change token ranking or greedy argmax.',
            'No fitted/proposed beta, Newton step, candidate promotion or unchanged-quality claim.',
            'FP32 per-target sums may round differently from the exact historical chunk CE anchors.']}
    try:
        run(args, report)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        native.write_json(args.report, report)
    print(json.dumps({'complete':True, 'report':str(args.report), 'sha256':native.sha(args.report),
        'summaries':report['summaries']}), flush=True)


if __name__ == '__main__':
    main()
