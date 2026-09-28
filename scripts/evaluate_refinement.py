#!/usr/bin/env python3
"""Evaluate a complete, manifest-bound E8 refinement overlay without copying v1.

The parent supplies both W5 vocabularies, small FP16 tensors, configuration and
codebook. Exactly112 replacement E8 projections are required. GPU evaluation
compares parent and candidate in one process with the frozen public dev suite.
Optional full-validation PPL runs only after the declared development gate.
This is an experiment helper, never publication or release approval.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAT = 'MAMBA2_E8W5_REFINEMENT_OVERLAY_V1'
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
RECIPE = {'tune_iters': 8, 'scale_override': 0.9, 'damping': 0.01, 'buffer_width': 128,
          'residual_bits': 0, 'index_bits': 16, 'seed_rule': '1000 + 2*layer + part'}
SCREEN_LABELS = [f'layer{i}.{part}' for i in (0, 18, 55) for part in ('in_proj', 'out_proj')]
SCREEN_GATE = {'median_relative_mse_reduction_at_least': .01, 'individual_relative_mse_reduction_at_least': .005,
               'minimum_individual_passes': 4, 'maximum_individual_regression': .001}
ROLES = {'z': [0, 8192], 'x': [8192, 16384], 'B': [16384, 17408],
         'C': [17408, 18432], 'dt': [18432, 18560]}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def checked_file(directory, name, receipt):
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError(f'Unsafe package filename: {name!r}')
    path = directory/name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular package file: {name}')
    if path.stat().st_size != receipt['bytes'] or sha(path) != receipt['sha256']:
        raise ValueError(f'File integrity failure: {name}')
    return path


def read_header(path):
    with path.open('rb') as stream:
        prefix = stream.read(12)
        if len(prefix) != 12:
            raise ValueError('Truncated E8 header')
        magic, length = struct.unpack('<8sI', prefix)
        if magic != b'ME8HD001' or not 0 < length < 65536:
            raise ValueError('Unsupported E8 header')
        header = json.loads(stream.read(length))
    if (header.get('format') != 'mamba-e8-dct-hadamard-v1' or
            header.get('rotation') != 'dct2-kron-hadamard; maximal power-of-two factor' or
            header.get('index_order') != 'row-major' or header.get('axis_residual_amplitude', 0) != 0):
        raise ValueError('Refinement must preserve the frozen two-bit E8 format')
    m, n = header['shape']
    expected = 12+length+m*n//4+2*n+(m+7)//8+(n+7)//8+4
    if path.stat().st_size != expected:
        raise ValueError('E8 shape and file length disagree')
    return header


def verify_screen(screen_path, protocol_path, parent, overlay, calibration):
    """Recompute the declared training gate and bind the six retained pilot files."""
    binding = overlay['binding']
    if binding.get('refinement_protocol_sha256') != sha(protocol_path) or binding.get('screen_report_sha256') != sha(screen_path):
        raise ValueError('Overlay protocol or screen identity differs')
    screen = json.loads(screen_path.read_text())
    if (screen.get('format') != 'MAMBA2_E8W5_REFINEMENT_SCREEN_V1' or not screen.get('complete') or
            screen.get('parent_manifest_sha256') != overlay['parent_manifest_sha256'] or
            screen.get('predeclared_labels') != SCREEN_LABELS or screen.get('sweeps') != [2, 8] or
            screen.get('expansion_gate') != SCREEN_GATE or screen.get('calibration_split') != 'train' or
            screen.get('evaluation_data_used') is not False or set(screen.get('matrices', {})) != set(SCREEN_LABELS) or
            set(screen.get('comparisons', {})) != set(SCREEN_LABELS)):
        raise ValueError('Requires the complete predeclared six-matrix training screen')
    for field in ('source_checkpoint_sha256', 'tokenizer_sha256', 'hessian_manifest_sha256', 'codec_source_sha256'):
        if screen['binding'].get(field) != binding[field]:
            raise ValueError(f'Screen input binding differs: {field}')
    if any(screen['binding'].get(k) != v for k, v in {'scale': .9, 'damping': .01, 'tune_iters': 8}.items()):
        raise ValueError('Screen used a different recipe')
    if screen['binding'].get('runtime_source_sha256') != sha(ROOT/'mamba_e8w5/runtime.py'):
        raise ValueError('Screen runtime source differs')
    if screen['binding'].get('quip_sha256') != parent['binding'].get('quip_sha256'):
        raise ValueError('Screen QuIP source differs from parent')
    reductions = []
    for label in SCREEN_LABELS:
        arms = screen['matrices'][label]
        if set(arms) != {'2', '8'}:
            raise ValueError('Screen must retain both sweep counts')
        baseline, candidate = arms['2'], arms['8']
        for arm, sweeps in ((baseline, 2), (candidate, 8)):
            if (arm.get('disk_roundtrip_fp16_equal') is not True or arm.get('tune_iters') != sweeps or
                    arm.get('hessian_sha256') != calibration['matrices'][label]['sha256'] or
                    arm.get('scale_override') != .9 or arm.get('damping') != .01 or
                    arm.get('seed') != overlay['matrices'][label]['seed']):
                raise ValueError(f'Screen arm provenance differs: {label}/{sweeps}')
        if (baseline.get('baseline_decoded_and_raw_sha_match') is not True or
                baseline['decoded_fp16_sha256'] != parent['matrices'][label]['decoded_fp16_sha256'] or
                baseline['sha256'] != parent['files'][label+'.e8']['sha256'] or
                baseline['bytes'] != parent['files'][label+'.e8']['bytes']):
            raise ValueError(f'Two-sweep screen did not reproduce parent: {label}')
        if any(candidate[key] != overlay['matrices'][label][key] for key in ('bytes', 'sha256', 'decoded_fp16_sha256')):
            raise ValueError(f'Full overlay changed a retained pilot matrix: {label}')
        b, c = baseline['original_hessian_weighted_squared_error'], candidate['original_hessian_weighted_squared_error']
        if not all(math.isfinite(v) and v >= 0 for v in (b, c)) or (b == 0 and c != 0):
            raise ValueError('Invalid screen quadratic errors')
        reduction = 1-c/b if b else 0.0
        if not math.isclose(reduction, screen['comparisons'][label]['relative_output_mse_reduction'], abs_tol=1e-12):
            raise ValueError('Screen recorded reduction differs from measured errors')
        reductions.append(reduction)
    median, passes, worst = statistics.median(reductions), sum(v >= .005 for v in reductions), min(reductions)
    passed = median >= .01 and passes >= 4 and worst >= -.001
    if not passed or screen.get('gate_result', {}).get('pass') is not True:
        raise ValueError('The predeclared six-matrix training screen did not pass')
    return {'screen_report_sha256': sha(screen_path), 'protocol_sha256': sha(protocol_path),
            'gate_recomputed': True, 'median_relative_mse_reduction': median,
            'individual_pass_count': passes, 'worst_relative_mse_reduction': worst, 'passed': passed}


def verify_overlay(parent_dir, overlay_dir, calibration_manifest, screen_path, protocol_path):
    """CPU-only integrity/provenance check, before any model or GPU allocation."""
    parent_path, overlay_path = parent_dir/'manifest.json', overlay_dir/'manifest.json'
    parent, overlay = json.loads(parent_path.read_text()), json.loads(overlay_path.read_text())
    calibration = json.loads(calibration_manifest.read_text())
    if not parent.get('complete') or not overlay.get('complete') or not calibration.get('complete'):
        raise ValueError('Parent, overlay and calibration must all be complete')
    if overlay.get('format') != FORMAT:
        raise ValueError('Unsupported overlay manifest format')
    if overlay.get('parent_manifest_sha256') != sha(parent_path):
        raise ValueError('Overlay belongs to another parent package')
    if parent.get('source_checkpoint_sha256') != SOURCE_SHA or parent.get('tokenizer_sha256') != TOKENIZER_SHA:
        raise ValueError('Parent source/tokenizer differs from audited public NVIDIA source')
    if overlay.get('model_config') != parent['model_config'] or calibration['model_config'] != parent['model_config']:
        raise ValueError('Architecture binding differs')
    if overlay.get('recipe') != RECIPE:
        raise ValueError('Overlay must use the fixed eight-sweep refinement recipe')
    binding = overlay['binding']
    expected = {'source_checkpoint_sha256': parent['source_checkpoint_sha256'],
                'tokenizer_sha256': parent['tokenizer_sha256'],
                'hessian_manifest_sha256': parent['binding']['hessian_manifest_sha256'],
                'codec_source_sha256': sha(ROOT/'mamba_e8w5/codec.py'),
                'runtime_source_sha256': sha(ROOT/'mamba_e8w5/runtime.py'),
                'quip_sha256': parent['binding']['quip_sha256']}
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f'Overlay provenance differs: {key}')
    for name in ('runtime.py', 'codec.py', 'quantize.py'):
        if parent['binding']['code_sha256'].get(name) != sha(ROOT/'mamba_e8w5'/name):
            raise ValueError(f'Frozen parent software changed: {name}')
    if set(parent['binding']['quip_sha256']) != {'lib/codebook/latticee8_padded12.py', 'lib/algo/quip.py'}:
        raise ValueError('Unexpected QuIP source inventory')
    for name, digest in parent['binding']['quip_sha256'].items():
        if digest != sha(ROOT/'third_party/quip-sharp'/name):
            raise ValueError(f'Frozen QuIP software changed: {name}')
    if (sha(calibration_manifest) != expected['hessian_manifest_sha256'] or
            calibration['source_checkpoint_sha256'] != expected['source_checkpoint_sha256'] or
            calibration['dataset']['tokenizer_sha256'] != expected['tokenizer_sha256'] or
            calibration.get('calibration_split') != 'train' or calibration.get('evaluation_data_used') is not False):
        raise ValueError('Calibration must match the original train-only Hessian receipt')
    labels = {f'layer{i}.{family}' for i in range(56) for family in ('in_proj', 'out_proj')}
    if set(overlay['matrices']) != labels:
        raise ValueError('Overlay requires exactly112 projection replacements')
    expected_files = {label+'.e8' for label in labels}
    if set(overlay['files']) != expected_files:
        raise ValueError('Overlay file ledger must contain exactly112 E8 files')
    if {path.name for path in overlay_dir.iterdir()} != expected_files | {'manifest.json'}:
        raise ValueError('Overlay directory has missing or unlisted entries; exact disk accounting requires a clean inventory')
    inherited = {name: value for name, value in parent['files'].items() if name not in expected_files}
    if set(inherited) != {'embedding.uniform', 'lm_head.uniform', 'other_fp16.pt', 'e8_codebook.bin', 'config.json'}:
        raise ValueError('Unexpected inherited parent inventory')
    if overlay.get('inherited_files') != inherited:
        raise ValueError('Inherited file ledger must exactly match the five unchanged parent files')
    if {path.name for path in parent_dir.iterdir()} != set(parent['files']) | {'manifest.json'}:
        raise ValueError('Parent directory has unlisted entries; exact disk accounting requires a clean inventory')
    for name, entry in parent['files'].items():
        checked_file(parent_dir, name, entry)
    for label in sorted(labels):
        entry = overlay['matrices'][label]
        layer, family = label.split('.')
        filename = label+'.e8'
        source_key = f'backbone.layers.{int(layer[5:])}.mixer.{family}.weight'
        shape = [18560, 4096] if family == 'in_proj' else [4096, 8192]
        if entry.get('file') != filename or entry.get('source_key') != source_key or entry.get('shape') != shape:
            raise ValueError(f'Projection identity/shape mismatch: {label}')
        seed = 1000+2*int(layer[5:])+(family == 'out_proj')
        if entry.get('seed') != seed:
            raise ValueError(f'Fixed per-matrix seed differs: {label}')
        fixed = {'tune_iters': 8, 'scale_override': .9, 'damping': .01, 'index_bits': 16,
                 'values_per_index': 8, 'axis_residual_amplitude': 0.0, 'feedback': True}
        if any(entry.get(key) != value for key, value in fixed.items()):
            raise ValueError(f'Per-matrix refinement recipe differs: {label}')
        if entry.get('hessian_sha256') != calibration['matrices'][label]['sha256']:
            raise ValueError(f'Per-matrix calibration identity mismatch: {label}')
        digest = entry.get('decoded_fp16_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Missing decoded FP16 hash: {label}')
        ledger = overlay['files'][filename]
        if any(entry.get(k) != ledger[k] for k in ('bytes', 'sha256')):
            raise ValueError(f'Duplicate file ledger disagrees: {label}')
        path = checked_file(overlay_dir, filename, ledger)
        if read_header(path)['shape'] != shape:
            raise ValueError(f'E8 header shape mismatch: {label}')
        if ledger['bytes'] != parent['files'][filename]['bytes']:
            raise ValueError(f'Same-bitrate refinement changed the serialized byte count: {label}')
    screen_receipt = verify_screen(screen_path, protocol_path, parent, overlay, calibration)
    parent_bytes = sum(x['bytes'] for x in parent['files'].values())
    replacement_bytes = sum(x['bytes'] for x in overlay['files'].values())
    removed_bytes = sum(parent['files'][name]['bytes'] for name in expected_files)
    receipt = {'parent_manifest_sha256': sha(parent_path), 'overlay_manifest_sha256': sha(overlay_path),
               'calibration_manifest_sha256': sha(calibration_manifest), 'binding': binding,
               'recipe': RECIPE, 'inherited_files': inherited,
               'screen_receipt': screen_receipt,
               'replacement_matrices': 112, 'native_in_proj_row_roles': ROLES,
               'parent_verified_file_bytes': parent_bytes, 'overlay_verified_file_bytes': replacement_bytes,
               'logical_candidate_data_bytes': parent_bytes-removed_bytes+replacement_bytes,
               'logical_reused_parent_data_bytes': parent_bytes-removed_bytes,
               'original_projection_bytes_replaced_logically': removed_bytes,
               'additive_overlay_disk_bytes': replacement_bytes+overlay_path.stat().st_size,
               'combined_parent_plus_overlay_disk_bytes': parent_bytes+parent_path.stat().st_size+replacement_bytes+overlay_path.stat().st_size,
               'manifest_bytes': {'parent': parent_path.stat().st_size, 'overlay': overlay_path.stat().st_size},
               'storage_note': 'Logical candidate reuses parent W5/FP16/config/codebook and replaces its E8 matrices. Actual experimental disk keeps the entire parent plus overlay. Tokenizer is counted separately below; executable code/licenses are not included, so these are not release archive byte totals.'}
    resolved = {name: {'origin': 'parent', **item} for name, item in inherited.items()}
    resolved.update({name: {'origin': 'overlay', **item} for name, item in overlay['files'].items()})
    receipt['resolved_files'] = resolved
    receipt['resolved_file_ledger_sha256'] = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return parent, overlay, receipt


def apply_overlay(model, overlay_dir, overlay):
    import torch
    from mamba_e8w5.codec import load_reference_primitives, read_e8, decode_e8
    from audit_decoded import tensor_sha_fp16
    device = next(model.parameters()).device
    identities = {name: id(p) for name, p in model.named_parameters()}
    if len(identities) != 507 or sum(p.numel() for p in model.parameters()) != 8236999680:
        raise ValueError('Loaded parent parameter coverage differs from audited8B architecture')
    changed = {entry['source_key'] for entry in overlay['matrices'].values()}
    book, _ = load_reference_primitives()
    book = book.to(device)
    hashes = {}
    for label, entry in sorted(overlay['matrices'].items()):
        decoded = decode_e8(read_e8(overlay_dir/entry['file']), book, device=device)
        actual = tensor_sha_fp16(decoded)
        if actual != entry['decoded_fp16_sha256'] or not torch.isfinite(decoded).all():
            raise ValueError(f'Decoded overlay FP16 hash/finite check failed: {label}')
        name = entry['source_key']
        parent, leaf = name.rsplit('.', 1)
        module = model.get_submodule(parent)
        if tuple(decoded.shape) != tuple(getattr(module, leaf).shape):
            raise ValueError(f'Runtime shape mismatch: {name}')
        setattr(module, leaf, torch.nn.Parameter(decoded, requires_grad=False))
        hashes[label] = actual
        del decoded
    if any(id(model.get_parameter(name)) != old for name, old in identities.items() if name not in changed):
        raise RuntimeError('Overlay modified an unrelated parameter')
    return {'decoded_fp16_hashes': hashes, 'unchanged_parameter_objects': len(identities)-len(changed),
            'resolved_parameter_tensors': len(identities), 'resolved_parameter_count': sum(p.numel() for p in model.parameters()),
            'coverage': 'Exactly112 E8 projections replaced; all W5 and other FP16 parameter objects retained'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', type=Path, required=True, help='Pinned tokenizer directory')
    ap.add_argument('--parent-dir', type=Path, required=True)
    ap.add_argument('--overlay-dir', type=Path, required=True)
    ap.add_argument('--calibration-manifest', type=Path, default=Path('calibration/v1/manifest.json'))
    ap.add_argument('--screen-report', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--verify-only', action='store_true', help='CPU manifest/file checks only; no torch/GPU/model load')
    ap.add_argument('--full-validation-if-improved', action='store_true')
    ap.add_argument('--protocol', type=Path, default=Path('docs/REFINEMENT_PROTOCOL.md'))
    args = ap.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    parent, overlay, receipt = verify_overlay(args.parent_dir, args.overlay_dir, args.calibration_manifest, args.screen_report, args.protocol)
    report = {'complete': False, 'scope': 'Refinement development comparison; no publication approval',
              'script_sha256': sha(__file__), 'refinement_protocol_sha256': sha(args.protocol),
              'overlay_receipt': receipt, 'verify_only': args.verify_only}
    if args.verify_only:
        report.update(complete=True, quality_evaluated=False)
        write_json(args.report, report)
        return
    import torch
    from mamba_e8w5.runtime import (MODEL_CONFIG, SentencePieceTokenizer, load_quantized_model,
                                    environment_receipt, gpu_memory_receipt)
    from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
    from mamba_e8w5.evaluation import evaluate_ppl, evaluate_mk, ppl_windows
    if parent['model_config'] != MODEL_CONFIG:
        raise ValueError('Parent does not match the audited8B configuration')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    if tokenizer.sha256 != parent['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from parent binding')
    receipt['tokenizer_bytes'] = tokenizer.path.stat().st_size
    receipt['logical_candidate_with_manifests_tokenizer_bytes'] = (receipt['logical_candidate_data_bytes']+
        sum(receipt['manifest_bytes'].values())+receipt['tokenizer_bytes'])
    report.update(runtime_source_sha256=sha(ROOT/'mamba_e8w5/runtime.py'),
                  evaluation_source_sha256=sha(ROOT/'mamba_e8w5/evaluation.py'),
                  decoded_hash_helper_sha256=sha(ROOT/'scripts/audit_decoded.py'),
                  codec_source_sha256=sha(ROOT/'mamba_e8w5/codec.py'), environment=environment_receipt(),
                  execution='prefill', cache_dtype='FP16 for MK generation; SSD scan for prose prefill')
    report['declared_dev_gate'] = {'candidate_PPL_at_most_parent_times': 0.99,
        'normal_MK': 'correct count must not decrease', 'target_removed_MK': 'correct count must not increase',
        'limitation': '24 public dev cases are a screen; near-zero baseline recall cannot establish preservation'}
    write_json(args.report, report)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', WIKITEXT_REVISION)
    reference_path = ROOT/'reports/e8w5_dev_prefill.json'
    reference = json.loads(reference_path.read_text())
    if dataset != reference['dataset'] or any(report[name+'_source_sha256'] != reference[name+'_source_sha256']
                                               for name in ('runtime', 'evaluation')):
        raise ValueError('Frozen development dataset or evaluation/runtime code changed')
    report['historical_dev_protocol_reference_sha256'] = sha(reference_path)
    report['dataset'] = dataset
    dev_windows = ppl_windows(ids, 1024, 4)
    from mamba_e8w5.runtime import token_digest
    if len(dev_windows) != len(reference['ppl']['windows']):
        raise ValueError('Development window count differs')
    for (first, tokens), prior in zip(dev_windows, reference['ppl']['windows']):
        if (first, len(tokens)-1, token_digest(tokens.numpy())) != (prior['start'], prior['target_tokens'], prior['token_sha256_int64le']):
            raise ValueError('Development window identity differs from frozen protocol')
    model = load_quantized_model(args.parent_dir)
    if model._package_receipt['manifest_sha256'] != receipt['parent_manifest_sha256']:
        raise ValueError('Loaded parent identity differs')
    report['parent_package_receipt'] = model._package_receipt
    report['parent_dev'] = {'ppl': evaluate_ppl(model, dev_windows),
                            'mk': evaluate_mk(model, tokenizer, 'validation', 2, 'prefill')}
    write_json(args.report, report)
    parent_projection_refs = {entry['source_key']: model.get_parameter(entry['source_key'])
                              for entry in overlay['matrices'].values()} if args.full_validation_if_improved else {}
    report['overlay_application'] = apply_overlay(model, args.overlay_dir, overlay)
    report['candidate_package_receipt'] = {key: receipt[key] for key in
        ('parent_manifest_sha256', 'overlay_manifest_sha256', 'resolved_file_ledger_sha256',
         'calibration_manifest_sha256', 'logical_candidate_data_bytes')}
    report['candidate_package_receipt']['format'] = 'Resolved E8 refinement overlay plus unchanged parent files, decoded to FP16'
    report['candidate_dev'] = {'ppl': evaluate_ppl(model, dev_windows),
                               'mk': evaluate_mk(model, tokenizer, 'validation', 2, 'prefill')}
    base, cand = report['parent_dev'], report['candidate_dev']
    gain = (base['ppl']['nll']-cand['ppl']['nll'])/base['ppl']['target_tokens']
    base_mk, cand_mk = base['mk']['summary'], cand['mk']['summary']
    paired = []
    for a, b in zip(base['mk']['rows'], cand['mk']['rows']):
        if (a['id'], a['prompt_token_sha256_int64le']) != (b['id'], b['prompt_token_sha256_int64le']):
            raise ValueError('MK paired prompts differ')
        paired.append({'id': a['id'], 'condition': a['condition'], 'parent_correct': a['correct'], 'candidate_correct': b['correct']})
    passed = (cand['ppl']['ppl'] <= 0.99*base['ppl']['ppl'] and
              cand_mk['normal']['correct'] >= base_mk['normal']['correct'] and
              cand_mk['target_removed']['correct'] <= base_mk['target_removed']['correct'])
    report['dev_comparison'] = {'mean_nll_improvement': gain, 'ppl_relative_change': cand['ppl']['ppl']/base['ppl']['ppl']-1,
                               'gate_passed': passed, 'paired_MK': paired}
    write_json(args.report, report)
    if args.full_validation_if_improved and passed:
        validation_windows = ppl_windows(ids, 2048, None)
        result = evaluate_ppl(model, validation_windows)
        if result['target_tokens'] != len(ids)-1:
            raise ValueError('Full validation target coverage differs')
        report['candidate_full_validation'] = result
        write_json(args.report, report)
        candidate_projection_refs = {name: model.get_parameter(name) for name in parent_projection_refs}
        try:
            for name, value in parent_projection_refs.items():
                owner, leaf = name.rsplit('.', 1)
                setattr(model.get_submodule(owner), leaf, value)
            report['parent_full_validation_same_process'] = evaluate_ppl(model, validation_windows)
        finally:
            for name, value in candidate_projection_refs.items():
                owner, leaf = name.rsplit('.', 1)
                setattr(model.get_submodule(owner), leaf, value)
        paired_parent = report['parent_full_validation_same_process']
        report['full_validation_comparison'] = {
            'mean_nll_improvement': (paired_parent['nll']-result['nll'])/result['target_tokens'],
            'ppl_relative_change': result['ppl']/paired_parent['ppl']-1,
            'note': 'Paired parent and candidate in the same process, after development advancement only'}
    else:
        report['full_validation_skipped'] = 'Not requested' if not args.full_validation_if_improved else 'Predeclared development gate did not pass'
    report.update(complete=True, quality_evaluated=True, elapsed_seconds=time.perf_counter()-started,
                  gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    print(json.dumps({'complete': True, 'dev_PPL': cand['ppl']['ppl'], 'dev_gate_passed': passed,
                      'full_validation_evaluated': 'candidate_full_validation' in report}), flush=True)


if __name__ == '__main__':
    main()
