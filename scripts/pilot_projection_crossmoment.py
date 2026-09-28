#!/usr/bin/env python3
"""Bounded six-projection same-bitrate repair using native teacher cross moments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.codec import load_reference_primitives, vector_quantize, write_e8, read_e8, decode_e8
from mamba_e8w5.evaluation import evaluate_ppl
from mamba_e8w5.projection_repair import anchored_ridge_target, fresh_train_starts
from mamba_e8w5.projection_statistics import (DEFAULT_SIX, collect_projection_statistics,
    score_dense_from_statistics, score_native_fp16)
from mamba_e8w5.runtime import (load_source_model, load_quantized_model, SentencePieceTokenizer,
    token_digest, sha256_file, environment_receipt, gpu_memory_receipt)
from mamba_e8w5.small_overlay import apply_overlay
from audit_decoded import audit_loaded_model, tensor_sha_fp16

PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
CALIBRATION_SHA = '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
SMALL_DATA_SHA = '88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709'
FROZEN = {'runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'calibration.py': 'b5a7356a2ab769538940aed218653a092a7055aa26a9ec6f62d3783b9ac78ee5',
    'evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def key_for(label):
    layer, part = label.split('.')
    return f'backbone.layers.{int(layer[5:])}.mixer.{part}.weight'


def strip_tensors(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'finite': bool(torch.isfinite(value).all()), 'frobenius_norm': float(value.norm())}
    if isinstance(value, dict):
        return {key: strip_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strip_tensors(item) for item in value]
    return value


def summarize_gate(native, ppl):
    reductions = [1-native['crossmoment']['matrices'][name]['relative_output_mse'] /
        native['current']['matrices'][name]['relative_output_mse'] for name in DEFAULT_SIX]
    control = [1-native['crossmoment']['matrices'][name]['relative_output_mse'] /
        native['h_refresh']['matrices'][name]['relative_output_mse'] for name in DEFAULT_SIX]
    conditions = {'median_at_least_1_percent': statistics.median(reductions) >= .01,
        'at_least_four_improve_half_percent': sum(x >= .005 for x in reductions) >= 4,
        'no_regression_above_tenth_percent': min(reductions) >= -.001,
        'median_over_h_refresh_at_least_half_percent': statistics.median(control) >= .005,
        'joint_heldout_train_ppl_not_worse': ppl['crossmoment']['ppl'] <= ppl['current']['ppl']}
    return {'native_heldout_mse_reductions_vs_current': dict(zip(DEFAULT_SIX, reductions)),
        'native_heldout_mse_reductions_vs_h_refresh': dict(zip(DEFAULT_SIX, control)),
        'median_vs_current': statistics.median(reductions), 'median_vs_h_refresh': statistics.median(control),
        'conditions': conditions, 'pass': all(conditions.values()),
        'scope': 'Six projections on TRAIN; not full validation, equal-quality or release acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('models/source'))
    parser.add_argument('--parent-dir', type=Path, default=Path('artifacts/e8w5_v1'))
    parser.add_argument('--small-dir', type=Path, default=Path('artifacts/small_compensation_v1'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('docs/PROJECTION_CROSSMOMENT_PILOT_PROTOCOL.md'))
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.report.exists() or args.out_dir.exists():
        raise FileExistsError('Pilot outputs must be new; no resume or overwrite')
    identities = {str(args.protocol): args.protocol_sha256,
        str(args.parent_dir/'manifest.json'): PARENT_SHA,
        str(args.small_dir/'manifest.json'): SMALL_SHA,
        str(args.small_dir/'other_fp16.pt'): SMALL_VALUES_SHA,
        'calibration/v1/manifest.json': CALIBRATION_SHA,
        'training_data/small_compensation_v1/manifest.json': SMALL_DATA_SHA}
    identities.update({f'mamba_e8w5/{name}': digest for name, digest in FROZEN.items()})
    for filename, digest in identities.items():
        if sha256_file(filename) != digest:
            raise ValueError(f'Frozen input identity differs: {filename}')
    code_files = [Path(__file__), ROOT/'mamba_e8w5/projection_repair.py',
        ROOT/'mamba_e8w5/projection_statistics.py', ROOT/'mamba_e8w5/small_overlay.py',
        ROOT/'scripts/audit_decoded.py']
    code_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files}
    parent = json.loads((args.parent_dir/'manifest.json').read_text())
    small = json.loads((args.small_dir/'manifest.json').read_text())
    calibration = json.loads(Path('calibration/v1/manifest.json').read_text())
    training = json.loads(Path('training_data/small_compensation_v1/manifest.json').read_text())
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'train', WIKITEXT_REVISION)
    if dataset != training['dataset']:
        raise ValueError('Pinned TRAIN tokenstream differs')
    old_starts = calibration.get('starts', calibration.get('window_starts'))
    if old_starts is None or len(old_starts) != 32:
        raise ValueError('Expected32 original calibration intervals')
    fit_starts, heldout_starts, selection = fresh_train_starts(len(ids), old_starts+training['starts'])
    fit_windows = [ids[start:start+2048] for start in fit_starts]
    heldout_windows = [ids[start:start+2048] for start in heldout_starts]
    language_windows = list(zip(heldout_starts, heldout_windows))
    def receipt(starts, windows):
        return [{'start': start, 'stored_tokens': len(window),
                 'token_sha256_int64le': token_digest(window.numpy())} for start, window in zip(starts, windows)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True)
    for arm in ('h_refresh', 'crossmoment'):
        (args.out_dir/arm).mkdir()
    report = {'format': 'MAMBA2_PROJECTION_CROSSMOMENT_PILOT_V1', 'complete': False,
        'scope': 'six-matrix train-only same-raw-bitrate pilot', 'identities': identities,
        'code_sha256': code_hashes, 'environment': environment_receipt(), 'dataset': dataset,
        'selection': selection, 'fit_windows': receipt(fit_starts, fit_windows),
        'heldout_windows': receipt(heldout_starts, heldout_windows),
        'selected_labels': list(DEFAULT_SIX), 'arms': ['current', 'h_refresh', 'crossmoment'],
        'quantizer': {'damping': .01, 'scale_override': .9, 'tune_iters': 2},
        'statistics': {}, 'matrices': {}, 'native_heldout': {}, 'language': {},
        'limitations': ['Original teacher internals may conflict with trained small tensors.',
            'Local projection MSE cannot establish full-model PPL improvement.',
            'Statistics use unchanged candidate prefixes; this is not sequential layer repair.',
            'No validation/test/MK, full112 expansion, entropy-container measurement or publication.']}
    write_json(args.report, report)
    print(json.dumps({'stage': 'bound_before_statistics', 'selection': selection,
        'protocol_sha256': args.protocol_sha256, 'code_sha256': code_hashes}), flush=True)
    teacher = load_source_model(args.source_dir)
    candidate = load_quantized_model(args.parent_dir)
    report['source_receipt'] = teacher._package_receipt
    report['parent_loaded_audit'] = audit_loaded_model(candidate, parent, args.parent_dir)
    candidate = apply_overlay(candidate, args.small_dir, small)
    report['small_overlay_receipt'] = candidate._small_overlay_receipt
    original = dict(candidate.named_parameters())
    if len(original) != 507 or sum(p.numel() for p in original.values()) != 8236999680:
        raise ValueError('Architecture coverage differs')
    initial_hashes = {name: tensor_sha_fp16(value) for name, value in original.items()}
    write_json(args.report, report)
    # A discarded32-token pairing/serialization probe; no fitted weights or
    # pilot statistics from this call are reused in the48-window experiment.
    smoke = collect_projection_statistics(teacher, candidate, [fit_windows[0][:32]],
        split_label='discarded_train_pairing_smoke')
    smoke_native = score_native_fp16(teacher, candidate, [fit_windows[0][:32]],
        {name: original[key_for(name)] for name in DEFAULT_SIX}, split_label='discarded_train_pairing_smoke')
    for name in DEFAULT_SIX:
        collected = smoke['matrices'][name]['candidate_native_mse_per_token']
        measured = smoke_native['matrices'][name]['mse_per_token']
        if abs(collected-measured) > 1e-6*max(abs(collected), 1e-12):
            raise ValueError(f'Native pairing smoke differs: {name}')
    report['discarded_pairing_smoke'] = {'positions': 32, 'statistics_reused': False,
        'statistics': strip_tensors(smoke), 'native': smoke_native,
        'all_six_native_error_checks_pass': True}
    write_json(args.report, report)
    del smoke, smoke_native
    fit = collect_projection_statistics(teacher, candidate, fit_windows, split_label='fresh_train_fit')
    heldout = collect_projection_statistics(teacher, candidate, heldout_windows, split_label='fresh_train_heldout')
    report['statistics'] = {'fit': strip_tensors(fit), 'heldout': strip_tensors(heldout)}
    write_json(args.report, report)
    cb, ldlq = load_reference_primitives()
    cb = cb.cuda()
    decoded_arms = {'current': {name: original[key_for(name)] for name in DEFAULT_SIX},
                    'h_refresh': {}, 'crossmoment': {}}
    for label in DEFAULT_SIX:
        print(json.dumps({'stage': 'quantize', 'label': label}), flush=True)
        matrix_started = time.perf_counter()
        h = fit['matrices'][label]['H']
        k = fit['matrices'][label]['K']
        anchor = original[key_for(label)]
        target, solve = anchored_ridge_target(h, k, anchor)
        if fit['matrices'][label]['tokens'] != 65536 or heldout['matrices'][label]['tokens'] != 32768:
            raise ValueError('Projection token coverage differs')
        entry = {'solve': solve, 'dense_proxy': {}, 'quantized': {}}
        for name, weight in [('current', anchor), ('unquantized_target', target)]:
            entry['dense_proxy'][name] = {split: score_dense_from_statistics(weight, stats['matrices'][label])
                for split, stats in [('fit', fit), ('heldout', heldout)]}
        layer, part = label.split('.')
        seed = 1000+2*int(layer[5:])+(part == 'out_proj')
        for arm, weight in [('h_refresh', teacher.get_parameter(key_for(label))), ('crossmoment', target)]:
            restored, info, payload = vector_quantize(weight, h, cb, ldlq, seed,
                damping=.01, scale_override=.9, tune_iters=2)
            path = args.out_dir/arm/f'{label}.e8'
            write_e8(path, payload, info)
            disk = decode_e8(read_e8(path), cb)
            if not torch.equal(restored, disk) or not torch.isfinite(disk).all():
                raise ValueError(f'Nonexact/nonfinite export: {label}/{arm}')
            if path.stat().st_size != parent['matrices'][label]['bytes']:
                raise ValueError('Raw E8 matrix capacity changed')
            decoded_arms[arm][label] = disk
            entry['quantized'][arm] = {'file': str(path), 'bytes': path.stat().st_size,
                'sha256': sha256_file(path), 'decoded_fp16_sha256': tensor_sha_fp16(disk),
                'decode_exact': True, 'codec_info_about_fitting_target': info,
                'teacher_proxy': {split: score_dense_from_statistics(disk, stats['matrices'][label])
                    for split, stats in [('fit', fit), ('heldout', heldout)]}}
            del restored, payload
        entry['elapsed_seconds'] = time.perf_counter()-matrix_started
        report['matrices'][label] = entry
        write_json(args.report, report)
        del target, h, k
    for arm in report['arms']:
        print(json.dumps({'stage': 'native_heldout', 'arm': arm}), flush=True)
        report['native_heldout'][arm] = score_native_fp16(teacher, candidate, heldout_windows,
            decoded_arms[arm], split_label='fresh_train_heldout')
        write_json(args.report, report)
    try:
        for arm in report['arms']:
            for label, value in decoded_arms[arm].items():
                module, leaf = key_for(label).rsplit('.', 1)
                parameter = original[key_for(label)] if arm == 'current' else torch.nn.Parameter(value, requires_grad=False)
                setattr(candidate.get_submodule(module), leaf, parameter)
            report['language'][arm] = evaluate_ppl(candidate, language_windows, execution='prefill', logits_chunk=64)
            result = report['language'][arm]
            if result['target_tokens'] != 32752 or len(result['windows']) != 16:
                raise ValueError('Heldout TRAIN language coverage differs')
            expected_tokens = [token_digest(window.numpy()) for window in heldout_windows]
            if [row['token_sha256_int64le'] for row in result['windows']] != expected_tokens:
                raise ValueError('Language arms used different tokens')
            write_json(args.report, report)
    finally:
        for label in DEFAULT_SIX:
            module, leaf = key_for(label).rsplit('.', 1)
            setattr(candidate.get_submodule(module), leaf, original[key_for(label)])
    if any(candidate.get_parameter(name) is not value for name, value in original.items()):
        raise ValueError('Original parameter references were not restored')
    for name, value in original.items():
        if tensor_sha_fp16(value) != initial_hashes[name]:
            raise ValueError(f'Original tensor content changed: {name}')
    for path, digest in {**identities, **code_hashes}.items():
        if sha256_file(path) != digest:
            raise ValueError(f'Input changed during pilot: {path}')
    report['integrity'] = {'original_parameter_references_restored': 507,
        'original_tensor_hashes_unchanged': initial_hashes,
        'raw_replacement_bytes': {arm: sum(report['matrices'][label]['quantized'][arm]['bytes']
            for label in DEFAULT_SIX) for arm in ('h_refresh', 'crossmoment')},
        'logical_inference_payload_change_bytes': 0}
    report['gate'] = summarize_gate(report['native_heldout'], report['language'])
    report.update(complete=True, elapsed_seconds=time.perf_counter()-started, gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    print(json.dumps({'complete': True, 'gate': report['gate'],
        'ppl': {arm: result['ppl'] for arm, result in report['language'].items()},
        'report_sha256': sha256_file(args.report)}), flush=True)


if __name__ == '__main__':
    main()
