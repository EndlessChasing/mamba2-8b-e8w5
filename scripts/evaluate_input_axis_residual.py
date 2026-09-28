#!/usr/bin/env python3
"""Independently load the fixed 56 input-axis files and evaluate the declared arms.

No quantizer or training forward is invoked. Only the predeclared primary may
enable complete validation; the other controls are development-only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import diagnose_vocab_w4 as frozen
from diagnose_output_calibration_v2 import reproducibility_snapshot
from mamba_e8w5.runtime import MODEL_CONFIG

native, codec = frozen.native, frozen.codec
PROTOCOL_SHA = 'bf4af74313812a5f20373e9cfa68654b2cbe39a66841788ae4c66e965aa80088'
SCORER_SHA = 'c1dd8343a204923d6799f5b4eed31a6f0ffee1bc0774b2deaa15f34e0d54091d'
W4_REPORT_SHA = '3ae1bb986ddfa7f2e514fb2451885a83feee5b6c5cc3a1924f248b4a259e353b'
CALIBRATION_SHA = '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
VERIFIER_SHA = 'baff8c56c875258830ac00076f880105b7933a173f57ba0bf38e27cbf1436c06'
BUILDER_SHA = 'b06c947c0b034c645ea66f70432eb929df7c7791c31bc4f720d42fe65d960aa7'
LABELS = tuple(f'layer{i}.in_proj' for i in range(56))
IN_KEYS = tuple(f'backbone.layers.{i}.mixer.in_proj.weight' for i in range(56))
BASE = 'accepted_e8_w5_w5'
PRECISION = 'axis_in_w5_w5'
PRIMARY = 'axis_in_w4_w5'
CAPACITY = 'axis_in_w4_w4'
REPEAT = 'accepted_e8_w5_w5_repeat'
DEV_ARMS = (BASE, PRECISION, PRIMARY, CAPACITY)
ARM_FLAGS = {BASE: (False, False, False), PRECISION: (True, False, False),
             PRIMARY: (True, True, False), CAPACITY: (True, True, True),
             REPEAT: (False, False, False)}
CHANGED_COUNTS = {BASE: 0, PRECISION: 56, PRIMARY: 57, CAPACITY: 58, REPEAT: 0}


def expected_hashes(base, axis, vocab, arm):
    """Resolve a complete507 ledger with no unlisted replacement allowed."""
    if (len(base) != 507 or set(axis) != set(IN_KEYS)
            or set(vocab) != set(frozen.VOCAB)
            or not set(axis).union(vocab).issubset(base) or arm not in ARM_FLAGS):
        raise ValueError('Incorrect full-model/replacement tensor coverage')
    enhanced, embedding, head = ARM_FLAGS[arm]
    result = dict(base)
    if enhanced:
        result.update(axis)
    for name, replace in zip(frozen.VOCAB, (embedding, head)):
        if replace:
            result[name] = vocab[name]
    changed = sorted(name for name in result if result[name] != base[name])
    if len(changed) != CHANGED_COUNTS[arm]:
        raise ValueError(f'Expected exactly{CHANGED_COUNTS[arm]} changed tensors for{arm}')
    return result, changed


def development_gate(arms):
    if set(arms) != set(DEV_ARMS):
        raise ValueError('All four fixed arms must finish before the primary gate')
    values = {name: arm['summary']['ppl'] for name, arm in arms.items()}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError('Invalid development PPL')
    return {'primary_arm': PRIMARY, 'baseline_arm': BASE, 'required_reduction': .01,
            'primary_ppl': values[PRIMARY], 'baseline_ppl': values[BASE],
            'maximum_primary_ppl': .99*values[BASE],
            'passed': values[PRIMARY] <= .99*values[BASE],
            'controls_can_trigger_validation': False}


def checked_file(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or native.sha(path) != expected:
        raise ValueError(f'Bound file identity differs:{path}')
    return path


def verify_inputs(args):
    # The existing scorer performs its unchanged parent/small/source/data audit.
    checked_file(ROOT/'scripts/diagnose_vocab_w4.py', SCORER_SHA)
    checked_file(ROOT/'docs/INPUT_AXIS_RESIDUAL_PROTOCOL.md', PROTOCOL_SHA)
    if VERIFIER_SHA.startswith('PENDING') or BUILDER_SHA.startswith('PENDING'):
        raise ValueError('Builder/verifier pins must be frozen before evaluation')
    verifier_path = ROOT/'mamba_e8w5/input_axis_residual.py'
    builder_path = ROOT/'scripts/quantize_input_axis_residual.py'
    checked_file(verifier_path, VERIFIER_SHA)
    checked_file(builder_path, BUILDER_SHA)
    verifier = importlib.import_module('mamba_e8w5.input_axis_residual')
    parent, small, receipt, data, windows, base, history, checked = frozen.verify_inputs()
    manifest = verifier.verify_overlay(args.overlay_dir)
    if (manifest.get('format') != 'MAMBA2_INPUT_AXIS_RESIDUAL_V1'
            or manifest.get('complete') is not True or manifest.get('model_config') != MODEL_CONFIG
            or manifest.get('parent_manifest_sha256') != native.PARENT_SHA
            or manifest.get('small_manifest_sha256') != native.SMALL_SHA):
        raise ValueError('Unexpected input-axis model provenance')
    binding = manifest['binding']
    for key, value in {'protocol_sha256': PROTOCOL_SHA,
        'source_checkpoint_sha256': native.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256': native.TOKENIZER_SHA256,
        'calibration_manifest_sha256': CALIBRATION_SHA,
        'codec_source_sha256': native.sha(ROOT/'mamba_e8w5/codec.py'),
        'runtime_source_sha256': native.sha(ROOT/'mamba_e8w5/runtime.py'),
        'builder_source_sha256': BUILDER_SHA, 'verifier_source_sha256': VERIFIER_SHA}.items():
        if binding.get(key) != value:
            raise ValueError(f'Input-axis binding differs:{key}')
    # Reconstruct the builder's required provenance from authoritative pinned
    # parents. A self-declared checked_input map is not sufficient evidence.
    calibration_path = ROOT/'calibration/v1/manifest.json'
    checked_file(calibration_path, CALIBRATION_SHA)
    calibration = json.loads(calibration_path.read_text())
    if (calibration.get('complete') is not True or calibration['calibration_split'] != 'train'
            or calibration['evaluation_data_used'] is not False or calibration['model_config'] != MODEL_CONFIG
            or calibration['source_checkpoint_sha256'] != native.SOURCE_CHECKPOINT_SHA256
            or parent['binding']['hessian_manifest_sha256'] != CALIBRATION_SHA
            or binding['quip_sha256'] != parent['binding']['quip_sha256']):
        raise ValueError('Original source/calibration/QuIP identity differs')
    required_builder_inputs = {
        ROOT/'docs/INPUT_AXIS_RESIDUAL_PROTOCOL.md': PROTOCOL_SHA,
        ROOT/'reports/vocab_w4_v1.json': W4_REPORT_SHA,
        ROOT/'artifacts/e8w5_v1/manifest.json': native.PARENT_SHA,
        ROOT/'artifacts/small_compensation_v1/manifest.json': native.SMALL_SHA,
        ROOT/'artifacts/small_compensation_v1/other_fp16.pt': native.SMALL_VALUES_SHA,
        calibration_path: CALIBRATION_SHA,
        frozen.checkpoint_path(ROOT/'models/source'): native.SOURCE_CHECKPOINT_SHA256,
        ROOT/'scripts/evaluate_teacher_kl_compensation.py': 'b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071',
        ROOT/'scripts/evaluate_small_compensation.py': '2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be',
        builder_path: BUILDER_SHA, verifier_path: VERIFIER_SHA}
    required_builder_inputs.update({ROOT/name: digest for name, digest in native.FROZEN.items()})
    required_builder_inputs.update({ROOT/'mamba_e8w5'/name: digest
        for name, digest in parent['binding']['code_sha256'].items()})
    required_builder_inputs.update({ROOT/'third_party/quip-sharp'/name: digest
        for name, digest in parent['binding']['quip_sha256'].items()})
    required_builder_inputs.update({ROOT/'artifacts/e8w5_v1'/name: row['sha256']
        for name, row in parent['files'].items()})
    expected_inherited = {name: {**row, 'origin': 'parent'} for name, row in parent['files'].items()
        if name not in {label+'.e8' for label in LABELS}}
    expected_inherited['other_fp16.pt'] = {**small['files']['other_fp16.pt'], 'origin': 'small_compensation_v1'}
    if len(expected_inherited) != 61 or manifest.get('inherited_files') != expected_inherited:
        raise ValueError('Expected61 inherited files, including accepted393 small tensors')
    if set(manifest['matrices']) != set(LABELS):
        raise ValueError('Exactly56 input projection files required')
    axis = {}
    for layer, (label, name) in enumerate(zip(LABELS, IN_KEYS)):
        row = manifest['matrices'][label]
        if (row['source_key'] != name or row['shape'] != [18560, 4096]
                or row['seed'] != 1000+2*layer or row['index_bits'] != 20
                or row['values_per_index'] != 8 or row['source_dtype'] != 'torch.bfloat16'):
            raise ValueError(f'Fixed projection recipe/geometry differs:{label}')
        hessian = calibration['matrices'][label]
        if (hessian['file'] != label+'.pt' or hessian['shape'] != [4096, 4096]
                or hessian['sha256'] != parent['matrices'][label]['hessian_sha256']
                or row['hessian_sha256'] != hessian['sha256']):
            raise ValueError(f'Original calibration Hessian binding differs:{label}')
        required_builder_inputs[ROOT/'calibration/v1'/hessian['file']] = hessian['sha256']
        axis[name] = row['decoded_fp16_sha256']
    if binding['checked_input_sha256'] != {str(path): digest for path, digest in required_builder_inputs.items()}:
        raise ValueError('Builder input ledger differs from independently reconstructed required provenance')
    w4_path = ROOT/'reports/vocab_w4_v1.json'
    checked_file(w4_path, W4_REPORT_SHA)
    w4 = json.loads(w4_path.read_text())
    if (w4.get('complete') is not True or w4.get('baseline_repeat_exact') is not True
            or w4.get('references_restored') is not True or w4['dataset'] != data['dataset']
            or w4['window_plan'] != data['heldout_windows']
            or w4['base507_initial']['unchanged_content_sha256'] != base):
        raise ValueError('Completed W4 artifacts/model/data identity differs')
    for filename, name in zip(frozen.FILES, frozen.VOCAB):
        row = w4['files'][filename]
        path = ROOT/'artifacts/vocab_w4_v1'/filename
        checked_file(path, row['sha256'])
        layout = frozen.uniform_layout(path)
        if (layout['header'] != {'shape': [256000, 4096], 'bits': 4, 'group': 128, 'scale': 'fp16'}
                or layout['actual_file_bytes'] != 540672079 or row['source_key'] != name
                or row['source_dtype'] != 'torch.bfloat16'):
            raise ValueError('Unexpected original-BF16 W4 file')
        checked[str(path)] = row['sha256']
    w4_hashes = w4['stored_w4_decoded_fp16_sha256']
    resolved = {arm: expected_hashes(base, axis, w4_hashes, arm) for arm in ARM_FLAGS}
    paths = {ROOT/'docs/INPUT_AXIS_RESIDUAL_PROTOCOL.md': PROTOCOL_SHA,
        verifier_path: VERIFIER_SHA, builder_path: BUILDER_SHA, w4_path: W4_REPORT_SHA,
        args.overlay_dir/'manifest.json': native.sha(args.overlay_dir/'manifest.json'),
        Path(__file__).resolve(): native.sha(__file__)}
    for filename, row in manifest['files'].items():
        paths[args.overlay_dir/filename] = row['sha256']
    for path, digest in binding['checked_input_sha256'].items():
        paths[Path(path)] = digest
    for path, digest in paths.items():
        if str(path) in checked and checked[str(path)] != digest:
            raise ValueError(f'Conflicting input identity:{path}')
        checked_file(path, digest)
        checked[str(path)] = digest
    prior = json.loads((ROOT/'reports/small_compensation_v1_eval.json').read_text())
    plan = [{'index': i, **row, 'target_tokens': row['targets']}
            for i, row in enumerate(data['heldout_windows'])]
    if len(plan) != 64 or sum(row['target_tokens'] for row in plan) != 131008:
        raise ValueError('Development target coverage differs')
    proof = {'overlay_manifest_sha256': native.sha(args.overlay_dir/'manifest.json'),
        'protocol_sha256': PROTOCOL_SHA, 'w4_report_sha256': W4_REPORT_SHA,
        'builder_source_sha256': BUILDER_SHA, 'verifier_source_sha256': VERIFIER_SHA,
        'scorer_source_sha256': SCORER_SHA, 'checked_input_sha256': checked,
        'resolved507_fp16_sha256_by_arm': {arm: item[0] for arm, item in resolved.items()},
        'changed_parameters_by_arm': {arm: item[1] for arm, item in resolved.items()},
        'base_package': receipt, 'overlay_binding': binding,
        'metadata_bytes': (args.overlay_dir/'manifest.json').stat().st_size,
        'overlay_data_bytes': sum(row['bytes'] for row in manifest['files'].values()),
        'complete_distribution_built': False}
    base_bytes = receipt['logical_candidate_data_bytes']
    extra = sum(manifest['files'][f'{label}.e8']['bytes']-parent['files'][f'{label}.e8']['bytes'] for label in LABELS)
    saving = [parent['files'][f]['bytes']-w4['files'][f]['actual_file_bytes'] for f in frozen.FILES]
    proof['resolved_raw_data_bytes_by_arm'] = {BASE: base_bytes, PRECISION: base_bytes+extra,
        PRIMARY: base_bytes+extra-saving[0], CAPACITY: base_bytes+extra-sum(saving)}
    proof['storage_note'] = 'Actual replacement-data bytes; manifests/tokenizer/software excluded. No Huffman distribution or compressed residency claim.'
    return verifier, parent, small, manifest, data, windows, plan, base, axis, w4_hashes, resolved, prior, w4, proof


def model_identity(model):
    return {name: (id(value), value.data_ptr(), value._version) for name, value in model.named_parameters()}


def set_parameter(model, name, value):
    old = model.get_parameter(name)
    if (value.shape != old.shape or value.dtype != torch.float16 or value.device != old.device
            or not torch.isfinite(value).all()):
        raise ValueError(f'Invalid native replacement:{name}')
    owner, leaf = name.rsplit('.', 1)
    setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(value, requires_grad=False))


@torch.no_grad()
def install_inputs(model, directory, cb, expected, verifier=None):
    """Release each previous matrix after independently decoding its replacement."""
    actual = {}
    for label, name in zip(LABELS, IN_KEYS):
        path = directory/f'{label}.e8'
        payload = verifier.read_axis_e8(path, expected_shape=(18560, 4096))[0] if verifier else codec.read_e8(path)
        decoded = codec.decode_e8(payload, cb, device='cuda')
        digest = native.tensor_sha_fp16(decoded)
        if digest != expected[name]:
            raise ValueError(f'Independent FP16 decode differs:{label}')
        set_parameter(model, name, decoded)
        actual[name] = digest
        del payload, decoded
    return actual


@torch.no_grad()
def install_w4(model, index, expected):
    name, filename = frozen.VOCAB[index], frozen.FILES[index]
    original = model.get_parameter(name)
    output = torch.empty_like(original)
    next_row = 0
    for first, rows in codec.iter_uniform(ROOT/'artifacts/vocab_w4_v1'/filename, device='cuda', chunk_rows=256):
        if first != next_row:
            raise ValueError('Noncontiguous W4 row stream')
        output[first:first+len(rows)].copy_(rows)
        next_row += len(rows)
    if next_row != len(output) or native.tensor_sha_fp16(output) != expected[name]:
        raise ValueError('Stored W4 native decode differs')
    set_parameter(model, name, output)


def check_row(row, tokens, plan):
    count = len(tokens)-1
    expected_chunks = [min(64, count-first) for first in range(0, count, 64)]
    if (row['target_tokens'] != count or count != plan['target_tokens']
            or row['chunk_target_counts'] != expected_chunks or len(row['ce_chunk_sums']) != len(expected_chunks)
            or native.token_digest(tokens.numpy()) != plan['token_sha256_int64le']
            or not all(math.isfinite(v) and v >= 0 for v in row['ce_chunk_sums'])
            or row['nll'] != math.fsum(row['ce_chunk_sums']) or not math.isfinite(row['ppl'])):
        raise ValueError('Native CE window/token/tail identity differs')


def score_arm(model, windows, plan, expected, destination, report_path, report, name, full=False):
    destination.update(coverage=native.audit_parameter_coverage(model),
        actual507_before=native.audit_frozen_values(model, expected), windows=[])
    identity = model_identity(model)
    for index, (tokens, expected_plan) in enumerate(zip(windows, plan)):
        row = frozen.score_window(model, tokens, chunk_tokens=64)
        check_row(row, tokens, expected_plan)
        row.update(expected_plan)
        destination['windows'].append(row)
        if (index+1) % 8 == 0:
            print(f'[axis evaluation] {name} {index+1}/{len(plan)}', flush=True)
            native.write_json(report_path, report)
    if len(destination['windows']) != len(plan) or model_identity(model) != identity:
        raise ValueError('Incomplete scoring or mutated parameter identity')
    summary = frozen.summarize(destination['windows'])
    if summary['target_tokens'] != sum(row['target_tokens'] for row in plan):
        raise ValueError('Aggregate target coverage differs')
    if full:
        native.check_arm({**summary, 'execution': 'prefill', 'logits_chunk_tokens': 64,
            'windows': destination['windows']}, plan)
    destination.update(summary=summary, actual507_after=native.audit_frozen_values(model, expected))
    native.write_json(report_path, report)


def exact_repeat(first, second):
    if len(first['windows']) != len(second['windows']):
        return False
    for a, b in zip(first['windows'], second['windows']):
        if a.keys() != b.keys() or a != b:
            return False
        for x, y in zip(a['ce_chunk_sums']+[a['nll']], b['ce_chunk_sums']+[b['nll']]):
            if struct.pack('<d', x) != struct.pack('<d', y):
                return False
    return first['summary'] == second['summary']


def paired_comparison(arms, names, baseline):
    base = arms[baseline]
    result = {}
    for name in names:
        rows = arms[name]['windows']
        deltas = [(a['nll']-b['nll'])/a['target_tokens'] for a, b in zip(rows, base['windows'])]
        result[name] = {'ppl': arms[name]['summary']['ppl'],
            'ppl_relative_change': arms[name]['summary']['ppl']/base['summary']['ppl']-1,
            'mean_nll_delta': (arms[name]['summary']['nll']-base['summary']['nll'])/base['summary']['target_tokens'],
            'per_window_mean_nll_delta': deltas, 'windows_improve': sum(d < 0 for d in deltas),
            'windows_equal': sum(d == 0 for d in deltas), 'windows_worsen': sum(d > 0 for d in deltas)}
    return result


def run(args, report):
    if args.verify_only:
        report['self_test'] = self_test()
    (verifier, parent, small, manifest, data, windows, plan, base, axis, w4_hashes,
     resolved, prior, w4, proof) = verify_inputs(args)
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
    report['reproducibility_before'] = reproducibility_snapshot()
    model = native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    native.small_overlay.apply_overlay(model, ROOT/'artifacts/small_compensation_v1', small)
    model.eval().requires_grad_(False)
    originals = tuple(model.get_parameter(name) for name in frozen.VOCAB)
    cb, _ = codec.load_reference_primitives()
    if cb.grid_packed_abs.cpu().numpy().astype('<i4').tobytes() != (ROOT/'artifacts/e8w5_v1/e8_codebook.bin').read_bytes():
        raise ValueError('Independent decoder codebook differs from pinned raw package')
    cb = cb.to('cuda').requires_grad_(False)
    modified_inputs = False

    def restore():
        nonlocal modified_inputs
        if modified_inputs:
            report['restored56_input_hashes'] = install_inputs(model, ROOT/'artifacts/e8w5_v1', cb, base)
            modified_inputs = False
        frozen.set_vocab(model, originals)

    try:
        for arm in DEV_ARMS:
            if arm == PRECISION:
                modified_inputs = True  # A partial installation must also be restored on failure.
                report['independent56_axis_hashes'] = install_inputs(model, args.overlay_dir, cb, axis, verifier)
            if arm == PRIMARY:
                install_w4(model, 0, w4_hashes)
            if arm == CAPACITY:
                install_w4(model, 1, w4_hashes)
            entry = report['development'][arm] = {'changed_parameters': resolved[arm][1]}
            score_arm(model, windows, plan, resolved[arm][0], entry, args.report, report, arm)
        restore()
        entry = report['development'][REPEAT] = {'changed_parameters': []}
        score_arm(model, windows, plan, base, entry, args.report, report, REPEAT)
        report['baseline_repeat_exact'] = exact_repeat(report['development'][BASE], entry)
        native.write_json(args.report, report)
        if not report['baseline_repeat_exact']:
            raise ValueError('Same-process complete64-window baseline repeat differs')
        report['development_comparison'] = paired_comparison(report['development'], DEV_ARMS[1:], BASE)
        report['development_gate'] = development_gate({name: report['development'][name] for name in DEV_ARMS})
        report['historical_development_baseline_drift'] = {
            'prior_ppl': w4['arms']['w5_w5']['summary']['ppl'],
            'current_ppl': entry['summary']['ppl'],
            'relative_ppl_change': entry['summary']['ppl']/w4['arms']['w5_w5']['summary']['ppl']-1,
            'scope': 'Descriptive only; all acceptance comparisons are paired in this process.'}
        native.write_json(args.report, report)
        if not report['development_gate']['passed']:
            report['full_validation']['reason'] = 'Fixed primary did not improve development PPL by1%; validation not loaded.'
            report['complete'] = True
            return
        # This is the first validation data access. No controls enter full validation.
        tokenizer = native.SentencePieceTokenizer(ROOT/'models/source')
        ids, dataset = native.load_wikitext_tokens(tokenizer, 'validation', native.WIKITEXT_REVISION)
        full_windows = native.ppl_windows(ids, 2048, None)
        full_plan = native.window_plan(full_windows)
        if dataset != prior['dataset'] or full_plan != prior['window_plan']:
            raise ValueError('Established complete validation identities differ')
        tokens = [value for _, value in full_windows]
        full = report['full_validation']
        full.update(performed=True, reason='Fixed primary development gate passed', dataset=dataset,
                    window_plan=full_plan, arms={})
        source = native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
        full['source_package_receipt'] = source._package_receipt
        source_hashes = {name: native.tensor_sha_fp16(value) for name, value in source.named_parameters()}
        source_entry = full['arms']['source_fp16'] = {}
        score_arm(source, tokens, full_plan, source_hashes, source_entry, args.report, report, 'source_fp16', full=True)
        del source
        gc.collect()
        torch.cuda.empty_cache()
        base_entry = full['arms'][BASE] = {}
        score_arm(model, tokens, full_plan, base, base_entry, args.report, report, BASE, full=True)
        # Reload actual files again; do not reuse generated/restored tensors.
        modified_inputs = True
        full['reloaded56_axis_hashes'] = install_inputs(model, args.overlay_dir, cb, axis, verifier)
        install_w4(model, 0, w4_hashes)
        candidate_entry = full['arms'][PRIMARY] = {}
        score_arm(model, tokens, full_plan, resolved[PRIMARY][0], candidate_entry,
                  args.report, report, PRIMARY, full=True)
        full['comparison'] = paired_comparison(full['arms'], (PRIMARY,), BASE)
        src, baseline, candidate = [full['arms'][name]['summary']['ppl'] for name in ('source_fp16', BASE, PRIMARY)]
        full['gates'] = {'meaningful_improvement_reference': {'required_reduction': .01, 'met': candidate <= .99*baseline},
            'source_plus5_percent_reference': {'maximum_ppl': 1.05*src, 'met': candidate <= 1.05*src},
            'candidate_vs_source_ppl_relative_change': candidate/src-1}
        full['historical_baseline_drift'] = {
            'source_relative_ppl': src/prior['results']['source_fp16']['ppl']-1,
            'accepted_relative_ppl': baseline/prior['results']['small_e8w5']['ppl']-1,
            'scope': 'Descriptive cross-process drift; not a historical bitwise gate.'}
        report['complete'] = True
    finally:
        restore()
        report['final_base507_restored_audit'] = native.audit_frozen_values(model, base)
        report['reproducibility_after'] = reproducibility_snapshot()
        report['gpu_memory'] = native.gpu_memory_receipt()
        for path, digest in proof['checked_input_sha256'].items():
            checked_file(path, digest)


def self_test():
    names = ['backbone.embedding.weight', 'lm_head.weight', 'backbone.norm_f.weight']
    for i in range(56):
        names += [f'backbone.layers.{i}.{suffix}' for suffix in (
            'norm.weight', 'mixer.norm.weight', 'mixer.dt_bias', 'mixer.A_log', 'mixer.D',
            'mixer.conv1d.weight', 'mixer.conv1d.bias', 'mixer.in_proj.weight', 'mixer.out_proj.weight')]
    base = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    axis = {name: hashlib.sha256(('axis'+name).encode()).hexdigest() for name in IN_KEYS}
    vocab = {name: hashlib.sha256(('w4'+name).encode()).hexdigest() for name in frozen.VOCAB}
    assert len(base) == 507
    for arm in ARM_FLAGS:
        actual, changed = expected_hashes(base, axis, vocab, arm)
        assert len(actual) == 507 and len(changed) == CHANGED_COUNTS[arm]
        assert all(actual[name] == digest for name, digest in base.items() if name not in changed)
    try:
        expected_hashes(base, {**axis, 'unexpected.weight': 'invalid'}, vocab, PRIMARY)
    except ValueError:
        pass
    else:
        raise AssertionError('Unlisted projection accepted')
    arms = {name: {'summary': {'ppl': value}} for name, value in zip(DEV_ARMS, (100., 90., 100., 80.))}
    assert not development_gate(arms)['passed']  # Good controls must not select themselves.
    arms[PRIMARY]['summary']['ppl'] = 99.
    assert development_gate(arms)['passed']
    try:
        development_gate({name: value for name, value in arms.items() if name != CAPACITY})
    except ValueError:
        pass
    else:
        raise AssertionError('Gate accepted incomplete fixed arms')
    ids = torch.arange(264765)
    windows = native.ppl_windows(ids, 2048, None)
    plan = native.window_plan(windows)
    assert len(plan) == 130 and plan[-1]['target_tokens'] == 572
    assert sum(row['target_tokens'] for row in plan) == 264764
    assert [min(64, 572-first) for first in range(0, 572, 64)] == [64]*8+[60]
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'cuda_initialized': False, 'checks': [
        'Independent complete507 inventory and exact0/56/57/58 replacement sets',
        'Reject unlisted projection and incomplete fixed-arm gate',
        'Only primary triggers validation; exact1% boundary',
        'Complete130-window/264764-target validation coverage and572-target final window/60-token CE tail'],
        'scorer_source_sha256': SCORER_SHA,
        'native_ce_tail_test_reused_from': 'Frozen diagnose_vocab_w4.self_test; no production scorer changes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay-dir', type=Path, default=ROOT/'artifacts/input_axis_residual_v1')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.report is not None:
        args.report = args.report.resolve()
        if args.report.parent != ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
            raise ValueError('Use a fresh report directly under reports/')
    if not args.self_test and args.report is None:
        parser.error('--report is required')
    if args.self_test and args.verify_only:
        parser.error('Select self-test or verify-only')
    args.overlay_dir = args.overlay_dir.resolve()
    torch.set_num_threads(8)
    report = {'format': 'MAMBA2_INPUT_AXIS_RESIDUAL_EVALUATION_V1', 'complete': False,
        'script_sha256': native.sha(__file__), 'protocol_sha256': PROTOCOL_SHA,
        'development_arm_order': list(DEV_ARMS)+[REPEAT], 'primary_arm': PRIMARY,
        'fitting_performed': False, 'mk_performed': False, 'candidate_advancement': False,
        'publication_performed': False, 'limitations': [
            'Development windows were already observed; complete validation has informed development.',
            'The primary adds raw bytes; no new complete Huffman distribution has been built.',
            'Native dense FP16 execution is not compressed runtime residency.',
            'Development controls cannot trigger validation or replace the fixed primary.']}
    started = time.monotonic()
    try:
        if args.self_test:
            report.update(self_test=self_test(), complete=True, mode='self_test')
        else:
            run(args, report)
    except BaseException as error:
        report.update(complete=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        if args.report is not None:
            native.write_json(args.report, report)
    print(json.dumps({'complete': report['complete'], 'report': str(args.report) if args.report else None,
        'self_test': report.get('self_test'), 'development_gate': report.get('development_gate'),
        'full_validation_performed': report.get('full_validation', {}).get('performed', False)}), flush=True)


if __name__ == '__main__':
    main()
