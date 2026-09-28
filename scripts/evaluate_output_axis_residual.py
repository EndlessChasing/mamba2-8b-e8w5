#!/usr/bin/env python3
"""Independent native evaluation of the fixed 56-output axis-residual overlay.

The input-axis/W4-embedding/W5-head model is the paired baseline. Quantizers
and training forwards are never invoked. Complete validation is conditional.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import evaluate_input_axis_residual as prior_stage

native, codec, frozen = prior_stage.native, prior_stage.codec, prior_stage.frozen
PROTOCOL_SHA = '3ac0a78c4b93dcc84f2402b39847fd93c5daf1705cbca360c2cdfa84e7e545e4'
INPUT_EVALUATOR_SHA = 'c8dece7ec209b74ef65244086c7c245f08a59ea344124559f964e44c8a2540f0'
INPUT_MANIFEST_SHA = '0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85'
INPUT_EVALUATION_SHA = '1dd6be0ddb4481e765da2658926ecea92183521ac9862b914fc3d3ba2a80a7eb'
VERIFIER_SHA = 'b135a1c50aec71cc9948e35231496f1439dd038676723a0ccaa299dfb33e8a97'
BUILDER_SHA = '786bf33186848cd620d83107a98b9082f3ae81d4e0a6a1b3d0ba6ae776b1fc65'
LABELS = tuple(f'layer{i}.out_proj' for i in range(56))
OUT_KEYS = tuple(f'backbone.layers.{i}.mixer.out_proj.weight' for i in range(56))
BASE = 'axis_in_w4_w5'
PRIMARY = 'axis_in_out_w4_w5'
REPEAT = 'axis_in_w4_w5_repeat'
DEV_ARMS = (BASE, PRIMARY)


def expected_hashes(accepted, paired, output):
    if (len(accepted) != 507 or set(paired) != set(accepted)
            or set(output) != set(OUT_KEYS) or not set(output).issubset(accepted)):
        raise ValueError('Incorrect complete507/output56 tensor coverage')
    base_changed = sorted(k for k in accepted if accepted[k] != paired[k])
    required_base = set(prior_stage.IN_KEYS) | {'backbone.embedding.weight'}
    if set(base_changed) != required_base:
        raise ValueError('Paired baseline must change exactly56 inputs and W4 embedding')
    candidate = {**paired, **output}
    paired_changed = sorted(k for k in paired if paired[k] != candidate[k])
    all_changed = sorted(k for k in accepted if accepted[k] != candidate[k])
    if set(paired_changed) != set(OUT_KEYS) or set(all_changed) != required_base | set(OUT_KEYS):
        raise ValueError('Expected exactly56 paired changes /113 changes relative to accepted')
    return {BASE: paired, PRIMARY: candidate, REPEAT: paired}, {
        BASE: base_changed, PRIMARY: all_changed, REPEAT: base_changed}, paired_changed


def development_gate(arms):
    if set(arms) != set(DEV_ARMS):
        raise ValueError('Both fixed arms must finish before the primary gate')
    baseline, candidate = (arms[name]['summary']['ppl'] for name in DEV_ARMS)
    if not all(math.isfinite(x) and x > 0 for x in (baseline, candidate)):
        raise ValueError('Invalid development PPL')
    return {'primary_arm': PRIMARY, 'baseline_arm': BASE, 'required_reduction': .01,
            'baseline_ppl': baseline, 'primary_ppl': candidate,
            'maximum_primary_ppl': .99*baseline, 'passed': candidate <= .99*baseline}


def verify_inputs(args):
    check = prior_stage.checked_file
    check(ROOT/'scripts/evaluate_input_axis_residual.py', INPUT_EVALUATOR_SHA)
    check(ROOT/'docs/OUTPUT_AXIS_RESIDUAL_PROTOCOL.md', PROTOCOL_SHA)
    input_dir = ROOT/'artifacts/input_axis_residual_v1'
    check(input_dir/'manifest.json', INPUT_MANIFEST_SHA)
    prior_path = ROOT/'reports/input_axis_residual_v1_eval.json'
    check(prior_path, INPUT_EVALUATION_SHA)
    if VERIFIER_SHA.startswith('PENDING') or BUILDER_SHA.startswith('PENDING'):
        raise ValueError('Output verifier/builder hashes must be frozen before evaluation')
    verifier_path = ROOT/'mamba_e8w5/output_axis_residual.py'
    builder_path = ROOT/'scripts/quantize_output_axis_residual.py'
    check(verifier_path, VERIFIER_SHA)
    check(builder_path, BUILDER_SHA)
    verifier = importlib.import_module('mamba_e8w5.output_axis_residual')
    (input_verifier, parent, small, input_manifest, data, windows, plan, accepted,
     input_axis, w4_hashes, input_resolved, prior_full, w4, input_proof) = prior_stage.verify_inputs(
         SimpleNamespace(overlay_dir=input_dir))
    prior_result = json.loads(prior_path.read_text())
    paired = input_resolved[prior_stage.PRIMARY][0]
    if (prior_result.get('complete') is not True or prior_result.get('baseline_repeat_exact') is not True
            or prior_result['script_sha256'] != INPUT_EVALUATOR_SHA
            or prior_result['integrity']['overlay_manifest_sha256'] != INPUT_MANIFEST_SHA
            or prior_result['integrity']['resolved507_fp16_sha256_by_arm'][BASE] != paired
            or prior_result['development_plan'] != plan or prior_result['dataset'] != data['dataset']
            or prior_result['full_validation'].get('performed') is not True):
        raise ValueError('Completed input experiment/model/data identity differs')
    manifest = verifier.verify_overlay(args.overlay_dir)
    if (manifest.get('format') != 'MAMBA2_OUTPUT_AXIS_RESIDUAL_V1'
            or manifest.get('complete') is not True or manifest.get('model_config') != prior_stage.MODEL_CONFIG
            or manifest.get('parent_manifest_sha256') != native.PARENT_SHA
            or manifest.get('small_manifest_sha256') != native.SMALL_SHA
            or manifest.get('input_manifest_sha256') != INPUT_MANIFEST_SHA
            or manifest.get('input_evaluation_sha256') != INPUT_EVALUATION_SHA):
        raise ValueError('Unexpected output-axis model provenance')
    binding = manifest['binding']
    for key, value in {'protocol_sha256': PROTOCOL_SHA,
        'input_manifest_sha256': INPUT_MANIFEST_SHA, 'input_evaluation_sha256': INPUT_EVALUATION_SHA,
        'source_checkpoint_sha256': native.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256': native.TOKENIZER_SHA256,
        'calibration_manifest_sha256': prior_stage.CALIBRATION_SHA,
        'codec_source_sha256': native.sha(ROOT/'mamba_e8w5/codec.py'),
        'runtime_source_sha256': native.sha(ROOT/'mamba_e8w5/runtime.py'),
        'builder_source_sha256': BUILDER_SHA, 'verifier_source_sha256': VERIFIER_SHA,
        'input_builder_source_sha256': prior_stage.BUILDER_SHA,
        'input_verifier_source_sha256': prior_stage.VERIFIER_SHA,
        'quip_sha256': parent['binding']['quip_sha256']}.items():
        if binding.get(key) != value:
            raise ValueError(f'Output-axis binding differs:{key}')
    calibration_path = ROOT/'calibration/v1/manifest.json'
    check(calibration_path, prior_stage.CALIBRATION_SHA)
    calibration = json.loads(calibration_path.read_text())
    # Reconstruct the output builder ledger independently; never accept an
    # arbitrary self-declared checked-input map as model provenance.
    # This input map has just been independently reconstructed and compared by
    # the frozen input evaluator above; it is not accepted from the output file.
    required = {Path(path): digest for path, digest in
                input_manifest['binding']['checked_input_sha256'].items()}
    required.update({ROOT/'docs/OUTPUT_AXIS_RESIDUAL_PROTOCOL.md': PROTOCOL_SHA,
        input_dir/'manifest.json': INPUT_MANIFEST_SHA, prior_path: INPUT_EVALUATION_SHA,
        ROOT/'scripts/quantize_input_axis_residual.py': prior_stage.BUILDER_SHA,
        ROOT/'mamba_e8w5/input_axis_residual.py': prior_stage.VERIFIER_SHA,
        builder_path: BUILDER_SHA, verifier_path: VERIFIER_SHA})
    required.update({input_dir/name: row['sha256'] for name, row in input_manifest['files'].items()})
    required[ROOT/'artifacts/vocab_w4_v1/embedding.uniform'] = w4['files']['embedding.uniform']['sha256']
    inherited = {name: {**row, 'origin': 'parent'} for name, row in parent['files'].items()
                 if name not in {label+'.e8' for label in LABELS}}
    inherited.update({name: {**row, 'origin': 'input_axis_residual_v1'} for name, row in input_manifest['files'].items()})
    inherited['other_fp16.pt'] = {**small['files']['other_fp16.pt'], 'origin': 'small_compensation_v1'}
    inherited['embedding.uniform'] = {'bytes': w4['files']['embedding.uniform']['actual_file_bytes'],
        'sha256': w4['files']['embedding.uniform']['sha256'], 'origin': 'vocab_w4_v1'}
    if len(inherited) != 61 or manifest.get('inherited_files') != inherited:
        raise ValueError('Output overlay must inherit exactly61 resolved input/W4/W5/small files')
    output_axis = {}
    for layer, (label, name) in enumerate(zip(LABELS, OUT_KEYS)):
        row = manifest['matrices'][label]
        if (row['source_key'] != name or row['shape'] != [4096, 8192]
                or row['seed'] != 1001+2*layer or row['index_bits'] != 20
                or row['values_per_index'] != 8 or row['source_dtype'] != 'torch.bfloat16'):
            raise ValueError(f'Fixed output geometry/recipe differs:{label}')
        hessian = calibration['matrices'][label]
        if (hessian['file'] != label+'.pt' or hessian['shape'] != [8192, 8192]
                or hessian['sha256'] != parent['matrices'][label]['hessian_sha256']
                or row['hessian_sha256'] != hessian['sha256']):
            raise ValueError(f'Original output Hessian differs:{label}')
        required[ROOT/'calibration/v1'/hessian['file']] = hessian['sha256']
        output_axis[name] = row['decoded_fp16_sha256']
    if binding['checked_input_sha256'] != {str(path): digest for path, digest in required.items()}:
        raise ValueError('Output builder ledger differs from independently required provenance')
    resolved, changed, paired_changed = expected_hashes(accepted, paired, output_axis)
    checked = dict(input_proof['checked_input_sha256'])
    required.update({ROOT/'scripts/evaluate_input_axis_residual.py': INPUT_EVALUATOR_SHA,
        Path(__file__).resolve(): native.sha(__file__),
        args.overlay_dir/'manifest.json': native.sha(args.overlay_dir/'manifest.json')})
    required.update({args.overlay_dir/name: row['sha256'] for name, row in manifest['files'].items()})
    for path, digest in required.items():
        if str(path) in checked and checked[str(path)] != digest:
            raise ValueError(f'Conflicting independently bound identity:{path}')
        check(path, digest)
        checked[str(path)] = digest
    base_bytes = input_proof['resolved_raw_data_bytes_by_arm'][prior_stage.PRIMARY]
    extra = sum(manifest['files'][label+'.e8']['bytes']-parent['files'][label+'.e8']['bytes'] for label in LABELS)
    if base_bytes != 3021487451:
        raise ValueError('Measured paired-baseline raw capacity differs')
    proof = {'overlay_manifest_sha256': native.sha(args.overlay_dir/'manifest.json'),
        'input_manifest_sha256': INPUT_MANIFEST_SHA, 'input_evaluation_sha256': INPUT_EVALUATION_SHA,
        'protocol_sha256': PROTOCOL_SHA, 'input_evaluator_sha256': INPUT_EVALUATOR_SHA,
        'builder_source_sha256': BUILDER_SHA, 'verifier_source_sha256': VERIFIER_SHA,
        'scorer_source_sha256': prior_stage.SCORER_SHA, 'checked_input_sha256': checked,
        'resolved507_fp16_sha256_by_arm': resolved, 'changed_parameters_relative_accepted': changed,
        'paired56_changed_parameters': paired_changed, 'base_package': input_proof['base_package'],
        'overlay_binding': binding, 'metadata_bytes': (args.overlay_dir/'manifest.json').stat().st_size,
        'overlay_data_bytes': sum(row['bytes'] for row in manifest['files'].values()),
        'resolved_raw_data_bytes_by_arm': {BASE: base_bytes, PRIMARY: base_bytes+extra},
        'storage_note': 'Actual raw replacement data; excludes manifests/tokenizer/software. No new Huffman distribution or compressed residency claim.',
        'complete_distribution_built': False}
    return verifier, input_verifier, parent, small, data, windows, plan, accepted, input_axis, w4_hashes, output_axis, resolved, prior_full, prior_result, proof


@torch.no_grad()
def install_outputs(model, directory, cb, expected, verifier=None):
    """Stream one replacement at a time; do not retain56 original dense outputs."""
    actual = {}
    for label, name in zip(LABELS, OUT_KEYS):
        path = directory/f'{label}.e8'
        payload = verifier.read_axis_e8(path, expected_shape=(4096, 8192))[0] if verifier else codec.read_e8(path)
        decoded = codec.decode_e8(payload, cb, device='cuda')
        digest = native.tensor_sha_fp16(decoded)
        if digest != expected[name]:
            raise ValueError(f'Independent stored output decode differs:{label}')
        prior_stage.set_parameter(model, name, decoded)
        actual[name] = digest
        del payload, decoded
    return actual


def run(args, report):
    if args.verify_only:
        report['self_test'] = self_test()
    (verifier, input_verifier, parent, small, data, windows, plan, accepted, input_axis,
     w4_hashes, output_axis, resolved, prior_full, prior_result, proof) = verify_inputs(args)
    report.update(integrity=proof, dataset=data['dataset'], development_plan=plan,
        development={}, full_validation={'performed': False, 'reason': 'Primary gate not yet evaluated'})
    native.write_json(args.report, report)
    if args.verify_only:
        if torch.cuda.is_initialized():
            raise ValueError('Verification must remain CPU only')
        report.update(complete=True, mode='verify_only', cuda_initialized=False)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    report['environment'] = native.environment_receipt()
    report['reproducibility_before'] = prior_stage.reproducibility_snapshot()
    model = native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    native.small_overlay.apply_overlay(model, ROOT/'artifacts/small_compensation_v1', small)
    model.eval().requires_grad_(False)
    report['initial_accepted507_audit'] = native.audit_frozen_values(model, accepted)
    cb, _ = codec.load_reference_primitives()
    if cb.grid_packed_abs.cpu().numpy().astype('<i4').tobytes() != (ROOT/'artifacts/e8w5_v1/e8_codebook.bin').read_bytes():
        raise ValueError('Independent E8 codebook differs')
    cb = cb.to('cuda').requires_grad_(False)
    report['independent56_fixed_input_hashes'] = prior_stage.install_inputs(
        model, ROOT/'artifacts/input_axis_residual_v1', cb, input_axis, input_verifier)
    prior_stage.install_w4(model, 0, w4_hashes)
    modified = False

    def restore():
        nonlocal modified
        if modified:
            report['restored56_output_hashes'] = install_outputs(model, ROOT/'artifacts/e8w5_v1', cb, resolved[BASE])
            modified = False

    try:
        for arm in DEV_ARMS:
            if arm == PRIMARY:
                modified = True
                report['independent56_output_axis_hashes'] = install_outputs(model, args.overlay_dir, cb, output_axis, verifier)
            entry = report['development'][arm] = {
                'changed_parameters_relative_accepted': proof['changed_parameters_relative_accepted'][arm],
                'changed_parameters_relative_paired': [] if arm == BASE else proof['paired56_changed_parameters']}
            prior_stage.score_arm(model, windows, plan, resolved[arm], entry, args.report, report, arm)
        restore()
        entry = report['development'][REPEAT] = {'changed_parameters_relative_paired': []}
        prior_stage.score_arm(model, windows, plan, resolved[BASE], entry, args.report, report, REPEAT)
        report['baseline_repeat_exact'] = prior_stage.exact_repeat(report['development'][BASE], entry)
        if not report['baseline_repeat_exact']:
            raise ValueError('Same-process complete64-window baseline repeat differs')
        report['development_comparison'] = prior_stage.paired_comparison(report['development'], (PRIMARY,), BASE)
        report['development_gate'] = development_gate({name: report['development'][name] for name in DEV_ARMS})
        previous = prior_result['development'][BASE]['summary']['ppl']
        report['historical_development_baseline_drift'] = {'prior_ppl': previous,
            'current_ppl': entry['summary']['ppl'], 'relative_ppl_change': entry['summary']['ppl']/previous-1,
            'scope': 'Descriptive only; advancement compares same-process paired arms.'}
        native.write_json(args.report, report)
        if not report['development_gate']['passed']:
            report['full_validation']['reason'] = 'Fixed primary did not improve development PPL by1%; validation not loaded.'
            report['complete'] = True
            return
        # First validation-data load occurs only after the complete paired gate.
        tokenizer = native.SentencePieceTokenizer(ROOT/'models/source')
        ids, dataset = native.load_wikitext_tokens(tokenizer, 'validation', native.WIKITEXT_REVISION)
        full_windows = native.ppl_windows(ids, 2048, None)
        full_plan = native.window_plan(full_windows)
        if (dataset != prior_full['dataset'] or full_plan != prior_full['window_plan']
                or full_plan != prior_result['full_validation']['window_plan']):
            raise ValueError('Established validation dataset/window identities differ')
        tokens = [value for _, value in full_windows]
        full = report['full_validation']
        full.update(performed=True, reason='Fixed primary development gate passed', dataset=dataset,
                    window_plan=full_plan, arms={})
        source = native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
        full['source_package_receipt'] = source._package_receipt
        source_hashes = {name: native.tensor_sha_fp16(value) for name, value in source.named_parameters()}
        entry = full['arms']['source_fp16'] = {}
        prior_stage.score_arm(source, tokens, full_plan, source_hashes, entry, args.report, report, 'source_fp16', full=True)
        del source
        gc.collect()
        torch.cuda.empty_cache()
        entry = full['arms'][BASE] = {}
        prior_stage.score_arm(model, tokens, full_plan, resolved[BASE], entry, args.report, report, BASE, full=True)
        modified = True
        full['reloaded56_output_axis_hashes'] = install_outputs(model, args.overlay_dir, cb, output_axis, verifier)
        entry = full['arms'][PRIMARY] = {}
        prior_stage.score_arm(model, tokens, full_plan, resolved[PRIMARY], entry, args.report, report, PRIMARY, full=True)
        full['comparison'] = prior_stage.paired_comparison(full['arms'], (PRIMARY,), BASE)
        source_ppl, base_ppl, primary_ppl = [full['arms'][name]['summary']['ppl'] for name in ('source_fp16', BASE, PRIMARY)]
        full['gates'] = {'meaningful_improvement_reference': {'required_reduction': .01, 'met': primary_ppl <= .99*base_ppl},
            'source_plus5_percent_reference': {'maximum_ppl': 1.05*source_ppl, 'met': primary_ppl <= 1.05*source_ppl},
            'candidate_vs_source_ppl_relative_change': primary_ppl/source_ppl-1}
        full['historical_baseline_drift'] = {
            'source_relative_ppl': source_ppl/prior_result['full_validation']['arms']['source_fp16']['summary']['ppl']-1,
            'paired_relative_ppl': base_ppl/prior_result['full_validation']['arms'][BASE]['summary']['ppl']-1,
            'scope': 'Descriptive cross-process drift; no historical equality gate.'}
        report['complete'] = True
    finally:
        restore()
        report['final_paired507_restored_audit'] = native.audit_frozen_values(model, resolved[BASE])
        report['reproducibility_after'] = prior_stage.reproducibility_snapshot()
        report['gpu_memory'] = native.gpu_memory_receipt()
        for path, digest in proof['checked_input_sha256'].items():
            prior_stage.checked_file(path, digest)


def self_test():
    inherited = prior_stage.self_test()
    names = ['backbone.embedding.weight', 'lm_head.weight', 'backbone.norm_f.weight']
    for i in range(56):
        names += [f'backbone.layers.{i}.{suffix}' for suffix in (
            'norm.weight', 'mixer.norm.weight', 'mixer.dt_bias', 'mixer.A_log', 'mixer.D',
            'mixer.conv1d.weight', 'mixer.conv1d.bias', 'mixer.in_proj.weight', 'mixer.out_proj.weight')]
    accepted = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    paired = dict(accepted)
    for name in list(prior_stage.IN_KEYS)+['backbone.embedding.weight']:
        paired[name] = hashlib.sha256(('paired'+name).encode()).hexdigest()
    output = {name: hashlib.sha256(('output'+name).encode()).hexdigest() for name in OUT_KEYS}
    resolved, changed, delta = expected_hashes(accepted, paired, output)
    assert len(resolved[PRIMARY]) == 507 and len(changed[BASE]) == 57 and len(changed[PRIMARY]) == 113 and len(delta) == 56
    assert all(resolved[PRIMARY][name] == value for name, value in paired.items() if name not in OUT_KEYS)
    for invalid in ({**output, 'unexpected.weight': 'invalid'}, {k: v for k, v in output.items() if k != OUT_KEYS[0]}):
        try:
            expected_hashes(accepted, paired, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Unlisted or missing output projection accepted')
    arms = {BASE: {'summary': {'ppl': 100.}}, PRIMARY: {'summary': {'ppl': 99.}}}
    assert development_gate(arms)['passed']
    arms[PRIMARY]['summary']['ppl'] = 99.0001
    assert not development_gate(arms)['passed']
    try:
        development_gate({BASE: arms[BASE]})
    except ValueError:
        pass
    else:
        raise AssertionError('Gate accepted incomplete fixed arms')
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'cuda_initialized': False, 'checks': [
        'Independent507 inventory:57/113 changes relative accepted, exactly56 paired output changes',
        'Reject extra/missing output projection;451 paired-baseline tensors remain fixed',
        'Paired1% boundary and incomplete-arm rejection'], 'reused_frozen_self_test': inherited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay-dir', type=Path, default=ROOT/'artifacts/output_axis_residual_v1')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/output_axis_residual_v1_eval.json')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    args.report = args.report.resolve()
    args.overlay_dir = args.overlay_dir.resolve()
    if args.report.parent != ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('Use a fresh report directly under reports/')
    if args.verify_only and args.self_test:
        parser.error('Select self-test or verify-only')
    torch.set_num_threads(8)
    report = {'format': 'MAMBA2_OUTPUT_AXIS_RESIDUAL_EVALUATION_V1', 'complete': False,
        'script_sha256': native.sha(__file__), 'protocol_sha256': PROTOCOL_SHA,
        'development_arm_order': list(DEV_ARMS)+[REPEAT], 'primary_arm': PRIMARY,
        'fitting_performed': False, 'mk_performed': False, 'candidate_advancement': False,
        'publication_performed': False, 'limitations': [
            'Development windows were previously observed; complete validation has informed development.',
            'The output overlay adds raw capacity; no new Huffman distribution is measured.',
            'Native dense FP16 execution is not compressed runtime residency.',
            'Only the fixed paired primary gate can enable full validation.']}
    started = time.monotonic()
    try:
        if args.self_test:
            report.update(self_test=self_test(), complete=True, mode='self_test', cuda_initialized=False)
        else:
            run(args, report)
    except BaseException as error:
        report.update(complete=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        native.write_json(args.report, report)
    print(json.dumps({'complete': report['complete'], 'report': str(args.report),
        'development_gate': report.get('development_gate'),
        'full_validation_performed': report.get('full_validation', {}).get('performed', False)}), flush=True)


if __name__ == '__main__':
    main()
