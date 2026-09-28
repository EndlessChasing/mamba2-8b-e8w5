#!/usr/bin/env python3
"""Four predeclared projection/small-tensor interventions, always with W5 vocab.

No training or model export. The two source-projection arms are diagnostic
oracles, not achieved same-storage repairs. Frozen evaluation math is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5 import small_overlay
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from mamba_e8w5.runtime import (SentencePieceTokenizer, load_source_model, load_quantized_model,
    environment_receipt, gpu_memory_receipt, token_digest)
from evaluate_small_compensation import (sha, write_json, window_plan, check_arm,
    audit_parameter_coverage, audit_selected_values, audit_frozen_values, audit_checkpoint_export)
from audit_decoded import audit_loaded_model, tensor_sha_fp16

PRIOR_SHA = '957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
SMALL_MANIFEST_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_EVALUATOR_SHA = '2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be'
ARMS = ('original_small_e8', 'trained_small_e8', 'original_small_source', 'trained_small_source')
VOCAB = ('backbone.embedding.weight', 'lm_head.weight')
TARGETS = 264764


def assign_references(model, references):
    if set(dict(model.named_parameters())) != set(references):
        raise ValueError('Reference swap must cover exactly every model parameter')
    for name, value in references.items():
        module, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(module), leaf, value)
    if any(model.get_parameter(name) is not value for name, value in references.items()):
        raise ValueError('Parameter reference assignment differs')


@torch.inference_mode()
def endpoint_probe(model, tokens):
    device = next(model.parameters()).device
    hidden = model.backbone(tokens[None].to(device))
    logits = model.lm_head(hidden[:, -8:])
    if hidden.dtype != torch.float16 or logits.dtype != torch.float16:
        raise ValueError('Endpoint probe must use FP16 native computation')
    if not torch.isfinite(hidden).all() or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite endpoint output')
    return {'input_tokens': len(tokens), 'input_sha256_int64le': token_digest(tokens.numpy()),
        'hidden_shape': list(hidden.shape), 'hidden_fp16_sha256': tensor_sha_fp16(hidden),
        'logits_shape': list(logits.shape), 'logits_fp16_sha256': tensor_sha_fp16(logits),
        'last_logit_tokens': 8, 'finite': True}


def conditional_effects(cases):
    a, b, c, d = (cases[name] for name in ARMS)
    effects = {}
    for label, first, second in (
        ('restore_projections_given_original_small', a, c),
        ('restore_projections_given_trained_small', b, d),
        ('train_small_given_e8_projections', a, b),
        ('train_small_given_source_projections', c, d)):
        effects[label] = {'mean_nll_delta': (second['nll']-first['nll'])/TARGETS,
            'ppl_ratio': second['ppl']/first['ppl'],
            'ppl_relative_change': second['ppl']/first['ppl']-1}
    interaction = (d['nll']-b['nll']-c['nll']+a['nll'])/TARGETS
    return {'ppl': {name: value['ppl'] for name, value in cases.items()},
        'mean_nll': {name: value['nll']/TARGETS for name, value in cases.items()},
        'conditional_effects': effects, 'interaction_mean_nll': interaction,
        'interaction_definition': '(trained/source - trained/E8) - (original/source - original/E8)',
        'interaction_sign': 'Positive means projection restoration removes less loss after small-tensor adaptation.',
        'oracles_are_achieved_E8_repairs': False}


def paired_effects(cases, plan):
    paired = []
    for index, row in enumerate(plan):
        nll = {name: cases[name]['windows'][index]['nll'] for name in ARMS}
        ppl = {name: cases[name]['windows'][index]['ppl'] for name in ARMS}
        a, b, c, d = (nll[name] for name in ARMS)
        count = row['target_tokens']
        paired.append({**row, 'nll': nll, 'ppl': ppl,
            'restore_projections_given_original_small_mean_nll': (c-a)/count,
            'restore_projections_given_trained_small_mean_nll': (d-b)/count,
            'train_small_given_e8_projections_mean_nll': (b-a)/count,
            'train_small_given_source_projections_mean_nll': (d-c)/count,
            'restore_projections_given_original_small_ppl_relative_change': ppl[ARMS[2]]/ppl[ARMS[0]]-1,
            'restore_projections_given_trained_small_ppl_relative_change': ppl[ARMS[3]]/ppl[ARMS[1]]-1,
            'train_small_given_e8_projections_ppl_relative_change': ppl[ARMS[1]]/ppl[ARMS[0]]-1,
            'train_small_given_source_projections_ppl_relative_change': ppl[ARMS[3]]/ppl[ARMS[2]]-1,
            'interaction_mean_nll': (d-b-c+a)/count})
    keys = [key for key in paired[0] if key.endswith('_mean_nll')]
    counts = {key: {'negative': sum(row[key] < 0 for row in paired),
                   'zero': sum(row[key] == 0 for row in paired),
                   'positive': sum(row[key] > 0 for row in paired)} for key in keys}
    return paired, counts


def self_test():
    """Small CPU reference-swap/interaction check; not model or quality evidence."""
    source = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    other = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    original = dict(source.named_parameters())
    coded = dict(other.named_parameters())
    before = {name: value.detach().clone() for name, value in original.items()}
    assign_references(source, coded)
    x = torch.tensor([[1., 2.]])
    if not torch.equal(source(x), other(x)):
        raise AssertionError('Direct and swapped synthetic outputs differ')
    mixed = {name: coded[name] if name.endswith('weight') else original[name] for name in original}
    assign_references(source, mixed)
    assign_references(source, original)
    if any(source.get_parameter(name) is not value or not torch.equal(value, before[name])
           for name, value in original.items()):
        raise AssertionError('Synthetic reference/content restoration failed')
    cases = {name: {'nll': TARGETS*loss, 'ppl': __import__('math').exp(loss)}
             for name, loss in zip(ARMS, (2., 1.9, 1.5, 1.7))}
    effect = conditional_effects(cases)
    if abs(effect['interaction_mean_nll']-.3) > 1e-12:
        raise AssertionError('Interaction sign/arithmetic differs')
    return {'cpu_only': True, 'reference_swap_and_restoration': 'passed',
            'direct_swapped_output': 'exact', 'conditional_interaction_arithmetic': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--source-dir', type=Path, default=Path('models/source'))
    parser.add_argument('--parent-dir', type=Path, default=Path('artifacts/e8w5_v1'))
    parser.add_argument('--overlay-dir', type=Path, default=Path('artifacts/small_compensation_v1'))
    parser.add_argument('--initialization-dir', type=Path, default=Path('artifacts/norm_compensation_v1'))
    parser.add_argument('--data-dir', type=Path, default=Path('training_data/small_compensation_v1'))
    parser.add_argument('--prior-report', type=Path, default=Path('reports/small_compensation_v1_eval.json'))
    parser.add_argument('--training-report', type=Path, default=Path('reports/small_compensation_v1_train.json'))
    parser.add_argument('--smoke-report', type=Path, default=Path('reports/small_compensation_v1_smoke.json'))
    parser.add_argument('--protocol', type=Path, default=Path('docs/SMALL_PROJECTION_INTERACTION_PROTOCOL.md'))
    parser.add_argument('--report', type=Path, default=Path('reports/small_projection_interaction_v1.json'))
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.self_test:
        print(json.dumps(self_test()), flush=True)
        return
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    if sha(args.prior_report) != PRIOR_SHA or sha(args.overlay_dir/'manifest.json') != SMALL_MANIFEST_SHA:
        raise ValueError('Fixed prior receipt or all-small candidate differs')
    if sha(ROOT/'scripts/evaluate_small_compensation.py') != SMALL_EVALUATOR_SHA:
        raise ValueError('Imported frozen audit/window helper differs')
    prior = json.loads(args.prior_report.read_text())
    if prior.get('complete') is not True:
        raise ValueError('Incomplete prior full-validation receipt')
    for name, expected in prior['frozen_code_sha256'].items():
        if sha(ROOT/'mamba_e8w5'/name) != expected:
            raise ValueError(f'Frozen implementation differs: {name}')
    if sha(small_overlay.__file__) != prior['small_overlay_source_sha256']:
        raise ValueError('Small overlay verifier differs')
    if sha(ROOT/'scripts/audit_decoded.py') != prior['loaded_tensor_audit_source_sha256']:
        raise ValueError('Independent loaded-tensor audit helper differs')
    if sha(args.training_report) != prior['inputs']['training_report']['sha256']:
        raise ValueError('Training report identity differs')
    train = json.loads(args.training_report.read_text())
    checkpoint = Path(train['final_checkpoint']['file'])
    parent, overlay, receipt = small_overlay.verify_overlay(args.parent_dir, args.overlay_dir,
        initialization_dir=args.initialization_dir, data_dir=args.data_dir,
        protocol=ROOT/'docs/SMALL_TENSOR_COMPENSATION_PROTOCOL.md',
        training_report=args.training_report, training_checkpoint=checkpoint, smoke_report=args.smoke_report)
    exported = torch.load(args.overlay_dir/'other_fp16.pt', map_location='cpu', weights_only=True)
    rounding = audit_checkpoint_export(checkpoint, exported, overlay['binding'],
                                      overlay['training_receipt']['checkpoint_sha256'])
    small_names = set(exported)
    if len(small_names) != 393 or sum(value.numel() for value in exported.values()) != 3580928:
        raise ValueError('Small-tensor inventory differs')
    bound_paths = {'parent_manifest': args.parent_dir/'manifest.json',
        'small_manifest': args.overlay_dir/'manifest.json', 'small_export': args.overlay_dir/'other_fp16.pt',
        'prior_report': args.prior_report, 'protocol': args.protocol, 'script': Path(__file__),
        'training_report': args.training_report, 'checkpoint': checkpoint, 'smoke_report': args.smoke_report}
    hashes = {key: sha(path) for key, path in bound_paths.items()}
    report = {'format': 'MAMBA2_SMALL_PROJECTION_INTERACTION_V1', 'complete': False,
        'scope': 'Four predeclared full-validation interventions; W5 vocabularies fixed; no MK/test/training/export/publication.',
        'script_sha256': sha(__file__), 'protocol_sha256': sha(args.protocol), 'input_sha256': hashes,
        'prior_report_sha256': PRIOR_SHA, 'small_overlay_receipt': receipt,
        'independent_checkpoint_export_audit': rounding, 'frozen_code_sha256': prior['frozen_code_sha256'],
        'imported_audit_helper_sha256': SMALL_EVALUATOR_SHA,
        'loaded_tensor_audit_source_sha256': prior['loaded_tensor_audit_source_sha256'],
        'small_overlay_source_sha256': prior['small_overlay_source_sha256'],
        'arm_order': list(ARMS), 'same_process': True, 'fixed_vocabularies': 'parent W5 embedding and W5 lm_head in every arm',
        'environment': environment_receipt(), 'cases': {},
        'limitations': ['Source-projection interventions are FP16 storage oracles, not achieved E8 repairs.',
            'Validation has already informed development; this is not untouched test evidence.',
            'PPL does not establish MK recall; no MK runs or recall claims are included.',
            'Source PPL from the previous report is historical; no true-source fifth arm is added.',
            'GPU allocation is expanded FP16 reference residency; timings are not throughput measurements.']}
    write_json(args.report, report)
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', WIKITEXT_REVISION)
    if dataset != prior['dataset'] or len(ids) != TARGETS+1:
        raise ValueError('Validation token stream differs')
    windows = ppl_windows(ids, 2048, None)
    plan = window_plan(windows)
    if plan != prior['window_plan']:
        raise ValueError('Validation windows differ')
    report.update(dataset=dataset, window_plan=plan,
        window_plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        scoring_protocol={'target_tokens': TARGETS, 'windows': 130, 'maximum_targets_per_window': 2048,
            'final_window_targets': 572, 'execution': 'prefill', 'logits_chunk_tokens': 64,
            'state': 'zero state per window; native SSD scan internal precision',
            'loss': 'frozen evaluation.py FP16 head products then FP32 summed cross entropy'})
    write_json(args.report, report)
    torch.cuda.reset_peak_memory_stats()
    print('[interaction] Loading source and original quantized parent', flush=True)
    source = load_source_model(args.source_dir)
    quantized = load_quantized_model(args.parent_dir)
    report['source_package_receipt'] = source._package_receipt
    report['parent_package_receipt'] = quantized._package_receipt
    original, coded = dict(source.named_parameters()), dict(quantized.named_parameters())
    if set(original) != set(coded):
        raise ValueError('Source and quantized parameter names differ')
    projections = {name for name in original if name.endswith(('mixer.in_proj.weight', 'mixer.out_proj.weight'))}
    if len(projections) != 112 or small_names | projections | set(VOCAB) != set(original):
        raise ValueError('112+393+2 parameter coverage differs')
    if small_names & projections or small_names & set(VOCAB) or projections & set(VOCAB):
        raise ValueError('Intervention groups overlap')
    report['source_coverage'] = audit_parameter_coverage(source)
    report['parent_loaded_tensor_audit'] = audit_loaded_model(quantized, parent, args.parent_dir)
    large_hashes = {entry['name']: entry['decoded_fp16_sha256']
        for entry in report['parent_loaded_tensor_audit']['tensors'] if entry['family'] in ('e8', 'w5')}
    if len(large_hashes) != 114:
        raise ValueError('Large decoded hash coverage differs')
    source_small = {name: original[name].detach().cpu() for name in small_names}
    report['original_small_source_parent_equality'] = audit_selected_values(quantized, source_small)
    source_projection_hashes = {name: tensor_sha_fp16(original[name]) for name in sorted(projections)}
    if any(original[name].dtype != torch.float16 for name in projections):
        raise ValueError('Source projections must be FP16')
    report['source_projection_fp16_sha256'] = source_projection_hashes
    probe_tokens = windows[0][1][:128]
    direct_parent = endpoint_probe(quantized, probe_tokens)
    try:
        small_overlay.apply_overlay(quantized, args.overlay_dir, overlay)
        trained = {name: quantized.get_parameter(name) for name in small_names}
        report['small_overlay_application'] = quantized._small_overlay_receipt
        report['trained_small_loaded_audit'] = audit_selected_values(quantized, exported)
        report['large_tensors_unchanged_after_overlay'] = audit_frozen_values(quantized, large_hashes)
        direct_trained = endpoint_probe(quantized, probe_tokens)

        def references(arm):
            use_trained = arm.startswith('trained_')
            use_source = arm.endswith('_source')
            result = {}
            for name in original:
                if name in VOCAB:
                    result[name] = coded[name]
                elif name in projections:
                    result[name] = original[name] if use_source else coded[name]
                else:
                    result[name] = trained[name] if use_trained else original[name]
            return result

        report['endpoint_parity'] = {}
        for arm, direct in ((ARMS[0], direct_parent), (ARMS[1], direct_trained)):
            assign_references(source, references(arm))
            swapped = endpoint_probe(source, probe_tokens)
            if swapped != direct:
                raise ValueError(f'Direct versus swapped endpoint differs: {arm}')
            report['endpoint_parity'][arm] = {'direct': direct, 'swapped': swapped, 'exact': True}
        write_json(args.report, report)
        print('[interaction] Both direct-versus-swapped E8 endpoints exactly equal', flush=True)
        for arm in ARMS:
            intended = references(arm)
            assign_references(source, intended)
            coverage = audit_parameter_coverage(source)
            if any(source.get_parameter(name) is not coded[name] for name in VOCAB):
                raise ValueError('W5 vocabularies changed across arms')
            actual_small = audit_selected_values(source, exported if arm.startswith('trained_') else source_small)
            print(f'[interaction] Scoring {arm}; W5 vocabularies fixed', flush=True)
            result = evaluate_ppl(source, windows, execution='prefill', logits_chunk=64)
            check_arm(result, plan)
            report['cases'][arm] = result
            report.setdefault('arm_integrity', {})[arm] = {'coverage': coverage,
                'all507_parameter_references_equal_declared_mapping': True,
                'W5_embedding_and_head_reference_identity': True, 'small_tensor_audit': actual_small}
            write_json(args.report, report)
            print(json.dumps({'arm': arm, 'ppl': result['ppl'], 'mean_nll': result['nll']/TARGETS}), flush=True)
    finally:
        assign_references(source, original)
        assign_references(quantized, coded)
    report['references_restored'] = {'source507': all(source.get_parameter(k) is v for k,v in original.items()),
                                    'parent507': all(quantized.get_parameter(k) is v for k,v in coded.items())}
    if not all(report['references_restored'].values()):
        raise ValueError('Original model references were not restored')
    report['restored_parent114_content_audit'] = audit_frozen_values(quantized, large_hashes)
    report['restored_source112_content_audit'] = audit_frozen_values(source, source_projection_hashes)
    report['comparison'] = conditional_effects(report['cases'])
    report['paired_windows'], report['window_sign_counts'] = paired_effects(report['cases'], plan)
    report['historical_anchor_comparisons'] = {
        arm: {'historical_ppl': prior['results'][old]['ppl'], 'current_ppl': report['cases'][arm]['ppl'],
            'relative_change': report['cases'][arm]['ppl']/prior['results'][old]['ppl']-1}
        for arm, old in ((ARMS[0], 'parent_e8w5'), (ARMS[1], 'small_e8w5'))}
    report['historical_true_source_reference'] = {'ppl': prior['results']['source_fp16']['ppl'],
        'source_report_sha256': PRIOR_SHA, 'same_process_as_this_diagnostic': False}
    for key, path in bound_paths.items():
        if sha(path) != hashes[key]:
            raise ValueError(f'Bound input changed during evaluation: {key}')
    report.update(complete=True, elapsed_seconds=time.perf_counter()-started, gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    print(json.dumps({'complete': True, 'comparison': report['comparison'], 'report_sha256': sha(args.report)}), flush=True)


if __name__ == '__main__':
    main()
