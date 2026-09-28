#!/usr/bin/env python3
"""CPU-only independent audit of the fixed six-matrix joint-axis screen.

Actual files, immutable metadata, provenance and reported objective arithmetic
are checked. This does not repeat GPU FP16 decoding or dense original-H GEMMs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import tempfile
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from mamba_e8w5 import codec
from mamba_e8w5.input_axis_residual import read_axis_e8, sha

PROTOCOL_SHA = '73c175d7a81886cc7102640966d0b2b8ff47734234d586f7c584f222e0a03bcd'
INPUT_MANIFEST_SHA = '0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85'
OUTPUT_MANIFEST_SHA = '2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f'
CURRENT_EVALUATION_SHA = '23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
CALIBRATION_SHA = '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
CODEC_SHA = 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f'
READER_SHA = 'baff8c56c875258830ac00076f880105b7933a173f57ba0bf38e27cbf1436c06'
SELECTOR_SHA = 'b84525d4ba2da0cc711b5e140bfc55ff631055fcfa0dd583d92237dbac6adbba'
SCREEN_SHA = '014822bf77d928e3a1cd8163574c5616bcba23342f607fe4ceb66f479945f34c'
LABELS = tuple(f'layer{layer}.{part}' for layer in (0, 18, 55) for part in ('in_proj', 'out_proj'))
PROJECTION_KEYS = {f'backbone.layers.{layer}.mixer.{part}.weight' for layer in range(56)
                   for part in ('in_proj', 'out_proj')}


def checked_file(path, digest):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != digest:
        raise ValueError(f'Bound regular-file identity differs:{path}')
    return path


def finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid finite nonnegative metric:{name}')
    return float(value)


def reduction(baseline, candidate):
    baseline = finite_nonnegative(baseline, 'greedy J')
    candidate = finite_nonnegative(candidate, 'joint J')
    if baseline == 0:
        if candidate != 0:
            raise ValueError('Zero-error baseline worsened; relative-error screen must reject')
        return 0.0
    return 1.0-candidate/baseline


def recompute_gate(greedy, joint):
    if set(greedy) != set(LABELS) or set(joint) != set(LABELS):
        raise ValueError('Exactly the six predeclared matrices are required')
    values = {label: reduction(greedy[label], joint[label]) for label in LABELS}
    median = statistics.median(values.values())
    count = sum(value >= .02 for value in values.values())
    worst = min(values.values())
    return values, {'reductions': values, 'median_reduction': median, 'improve_at_least_2pct_count': count,
        'min_reduction': worst, 'thresholds': {'median': .05, 'count_at_least_2pct': 4, 'minimum': -.001},
        'passed': median >= .05 and count >= 4 and worst >= -.001}


def file_parts(path, shape):
    """Read exact header/metadata bytes separately from the two code planes."""
    payload, layout = read_axis_e8(path, expected_shape=shape)
    raw = Path(path).read_bytes()
    first = layout['header_bytes']
    end = first+layout['base_uint16_bytes']+layout['axis_nibble_bytes']
    return raw, raw[:first], raw[end:], payload, layout


def compare_files(original, candidate, shape, require_full_equal=False):
    baseline, bh, bt, bp, bl = file_parts(original, shape)
    actual, ah, at, ap, al = file_parts(candidate, shape)
    if len(actual) != len(baseline) or ah != bh or at != bt or al != bl:
        raise ValueError('Fixed header/shape/amplitude/balance/signs/scale/raw length changed')
    if require_full_equal and actual != baseline:
        raise ValueError('Greedy replay bytes differ from the current stored baseline')
    if require_full_equal and not torch.equal(ap['indices'], bp['indices']):
        raise ValueError('Greedy replay indices differ')
    metadata = {'raw_bytes':len(actual), 'header_bytes':len(ah),
        'header_sha256':hashlib.sha256(ah).hexdigest(), 'tail_sha256':hashlib.sha256(at).hexdigest(),
        'tail_bytes':len(at), 'amplitude_float64_hex':struct.pack('<d',al['axis_residual_amplitude']).hex()}
    offset = 0
    for key in ('balance_fp16_bytes','input_sign_bytes','output_sign_bytes','scale_fp32_bytes'):
        length = al[key]
        metadata[key+'_sha256'] = hashlib.sha256(at[offset:offset+length]).hexdigest()
        offset += length
    if offset != len(at):
        raise ValueError('Metadata segment coverage differs')
    return {'bytes': len(actual), 'sha256': hashlib.sha256(actual).hexdigest(),
        'original_file_sha256': hashlib.sha256(baseline).hexdigest(),
        'header_sha256': hashlib.sha256(ah).hexdigest(),
        'metadata_tail_sha256': hashlib.sha256(at).hexdigest(),
        'metadata_bitwise_equal': True, 'full_file_equal': actual == baseline,
        'changed_codewords': int((ap['indices'] != bp['indices']).sum()), 'layout': al,
        'metadata_identity':metadata}


def self_test():
    greedy = {label: 100. for label in LABELS}
    joint = {label: 94. for label in LABELS}
    _, gate = recompute_gate(greedy, joint)
    assert gate['passed'] and gate['improve_at_least_2pct_count'] == 6
    joint[LABELS[0]] = 101.
    assert not recompute_gate(greedy, joint)[1]['passed']
    # High median alone is insufficient: only three matrices clear2% here.
    joint = dict(zip(LABELS, (100.05, 100., 99.9, 88., 87., 86.)))
    assert not recompute_gate(greedy, joint)[1]['passed']
    # Four individual gains are insufficient when the median misses5%.
    joint = dict(zip(LABELS, (100., 100., 97., 97., 97., 97.)))
    assert not recompute_gate(greedy, joint)[1]['passed']
    assert reduction(0, 0) == 0
    for bad in (-1., float('nan'), float('inf'), True):
        try:
            reduction(bad, 0)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid baseline metric accepted')
    try:
        reduction(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero baseline regression accepted')
    try:
        recompute_gate({k: v for k, v in greedy.items() if k != LABELS[0]}, joint)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete six-matrix gate accepted')
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        count = 16*128//8
        payload = {'indices': torch.arange(count, dtype=torch.int32).reshape(16,16),
            'balance': torch.ones(128, dtype=torch.float16),
            'input_sign': torch.ones(128, dtype=torch.int8),
            'output_sign': torch.ones(16, dtype=torch.int8),
            'scale': torch.tensor(.5, dtype=torch.float32), 'axis_residual_amplitude': .25}
        a, b = directory/'original.e8', directory/'candidate.e8'
        codec.write_e8(a, payload, {'shape': [16,128]})
        codec.write_e8(b, payload, {'shape': [16,128]})
        assert compare_files(a,b,[16,128],True)['full_file_equal']
        replacement = {**payload, 'indices': payload['indices'].clone()}
        replacement['indices'][0,0] = (65535<<4)|15
        codec.write_e8(b,replacement,{'shape':[16,128]})
        proof = compare_files(a,b,[16,128])
        assert proof['changed_codewords'] == 1 and not proof['full_file_equal']
        try:
            compare_files(a,b,[16,128],True)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed greedy replay codeword accepted')
        replacement['balance'] = payload['balance']*2
        codec.write_e8(b,replacement,{'shape':[16,128]})
        try:
            compare_files(a,b,[16,128])
        except ValueError:
            pass
        else:
            raise AssertionError('Changed non-code metadata accepted')
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'cuda_initialized': False, 'checks': [
        'Six-matrix squared-error gate, incomplete coverage and malformed/zero error handling',
        'Actual20bit boundary code roundtrip, full greedy-byte equality and one-codeword replacement',
        'Bytewise fixed header/amplitude/FP16balance/sign/FP32scale and metadata-change rejection']}


def audit(args):
    if SELECTOR_SHA.startswith('PENDING') or SCREEN_SHA.startswith('PENDING'):
        raise ValueError('Selector/screen source hashes must be frozen first')
    fixed = {
        ROOT/'docs/JOINT_AXIS_SELECTION_PROTOCOL.md': PROTOCOL_SHA,
        ROOT/'artifacts/input_axis_residual_v1/manifest.json': INPUT_MANIFEST_SHA,
        ROOT/'artifacts/output_axis_residual_v1/manifest.json': OUTPUT_MANIFEST_SHA,
        ROOT/'reports/output_axis_residual_v1_eval.json': CURRENT_EVALUATION_SHA,
        ROOT/'artifacts/e8w5_v1/manifest.json': PARENT_SHA,
        ROOT/'artifacts/small_compensation_v1/manifest.json': SMALL_SHA,
        ROOT/'calibration/v1/manifest.json': CALIBRATION_SHA,
        ROOT/'models/source/model_optim_rng.pt': SOURCE_SHA,
        ROOT/'mamba_e8w5/codec.py': CODEC_SHA,
        ROOT/'mamba_e8w5/input_axis_residual.py': READER_SHA,
        ROOT/'mamba_e8w5/joint_axis.py': SELECTOR_SHA,
        ROOT/'scripts/screen_joint_axis.py': SCREEN_SHA}
    for path, digest in fixed.items():
        checked_file(path, digest)
    inputs = json.loads((ROOT/'artifacts/input_axis_residual_v1/manifest.json').read_text())
    outputs = json.loads((ROOT/'artifacts/output_axis_residual_v1/manifest.json').read_text())
    current = json.loads((ROOT/'reports/output_axis_residual_v1_eval.json').read_text())
    parent = json.loads((ROOT/'artifacts/e8w5_v1/manifest.json').read_text())
    calibration = json.loads((ROOT/'calibration/v1/manifest.json').read_text())
    manifest_path = args.screen_dir/'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('A regular completed screen manifest is required')
    report = json.loads(manifest_path.read_text())
    expected_files = {f'{method}/{label}.e8' for method in ('greedy','joint') for label in LABELS}
    if (report.get('format') != 'MAMBA2_JOINT_AXIS_SCREEN_V1' or report.get('complete') is not True
            or report.get('stage') != 'complete' or report.get('script_sha256') != SCREEN_SHA
            or report.get('ppl_evaluated') is not False or report.get('candidate_advancement') is not False
            or report.get('labels') != list(LABELS)
            or set(report.get('greedy_replay',{})) != set(LABELS)
            or set(report.get('matrices',{})) != set(LABELS)
            or set(report.get('files',{})) != expected_files):
        raise ValueError('Wrong or incomplete fixed six-matrix screen inventory')
    recipe = {'residual_bits':4, 'index_bits':20, 'values_per_index':8, 'tune_iters':2,
        'damping':.01, 'scale_override':.9, 'feedback':True, 'buffer_width':128,
        'seed_rule':'1000 + 2*layer + part', 'row_batch':4096, 'shift_order':list(range(16))}
    if report.get('recipe') != recipe or report.get('model_config') != parent['model_config']:
        raise ValueError('Fixed six-screen recipe/model configuration differs')
    if {p.name for p in args.screen_dir.iterdir()} != {'greedy','joint','manifest.json'}:
        raise ValueError('Unlisted screen root files/directories')
    for method in ('greedy','joint'):
        directory = args.screen_dir/method
        if directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != {label+'.e8' for label in LABELS}:
            raise ValueError('Unexpected screen files or non-regular subdirectory')
    if (current.get('complete') is not True or not current['baseline_repeat_exact']
            or not current['full_validation']['performed'] or calibration['calibration_split'] != 'train'
            or calibration['evaluation_data_used'] is not False
            or calibration['source_checkpoint_sha256'] != SOURCE_SHA
            or parent['binding']['hessian_manifest_sha256'] != CALIBRATION_SHA):
        raise ValueError('Pinned baseline/calibration provenance differs')
    accepted = current['integrity']['resolved507_fp16_sha256_by_arm']['axis_in_out_w4_w5']
    if len(accepted) != 507 or not PROJECTION_KEYS.issubset(accepted):
        raise ValueError('Current complete507 ledger differs')
    inherited395 = {name: value for name,value in accepted.items() if name not in PROJECTION_KEYS}
    inherited_files = {name:row for name,row in outputs['inherited_files'].items() if not name.endswith('.e8')}
    if (len(inherited395) != 395 or report.get('inherited395_fp16_sha256') != inherited395
            or len(inherited_files) != 5 or report.get('inherited_files') != inherited_files):
        raise ValueError('Fixed395 tensor/five inherited-file identity differs')
    origins = {'parent': ROOT/'artifacts/e8w5_v1', 'vocab_w4_v1': ROOT/'artifacts/vocab_w4_v1',
               'small_compensation_v1': ROOT/'artifacts/small_compensation_v1'}
    for name,row in inherited_files.items():
        path = origins[row['origin']]/name
        checked_file(path,row['sha256'])
        if path.stat().st_size != row['bytes']:
            raise ValueError('Inherited file size differs')
        fixed[path] = row['sha256']
    binding = report['binding']
    tokenizer = next(row for row in parent['binding']['source']['files']
        if row['path'] == 'mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model')
    expected_binding = {'protocol_sha256':PROTOCOL_SHA, 'parent_manifest_sha256':PARENT_SHA,
        'small_manifest_sha256':SMALL_SHA, 'input_manifest_sha256':INPUT_MANIFEST_SHA,
        'output_manifest_sha256':OUTPUT_MANIFEST_SHA, 'current_evaluation_sha256':CURRENT_EVALUATION_SHA,
        'source_checkpoint_sha256':SOURCE_SHA, 'calibration_manifest_sha256':CALIBRATION_SHA,
        'tokenizer_sha256':tokenizer['sha256'], 'quip_sha256':parent['binding']['quip_sha256'],
        'builder_source_sha256':SCREEN_SHA, 'selector_source_sha256':SELECTOR_SHA}
    if {key:value for key,value in binding.items() if key!='checked_input_sha256'} != expected_binding:
        raise ValueError('Screen protocol/source/code/calibration binding differs')
    # Reconstruct exactly the narrow screen inputs from pinned authoritative
    # parents. Do not accept an arbitrary self-declared input map.
    required = dict(fixed)
    required.update({ROOT/'mamba_e8w5'/name:digest for name,digest in parent['binding']['code_sha256'].items()})
    required.update({ROOT/'third_party/quip-sharp'/name:digest for name,digest in parent['binding']['quip_sha256'].items()})
    required[ROOT/'models/source'/tokenizer['path']] = tokenizer['sha256']
    for label in LABELS:
        matrix_manifest, directory = ((inputs,ROOT/'artifacts/input_axis_residual_v1') if label.endswith('in_proj')
            else (outputs,ROOT/'artifacts/output_axis_residual_v1'))
        required[directory/(label+'.e8')] = matrix_manifest['files'][label+'.e8']['sha256']
        required[ROOT/'calibration/v1'/(label+'.pt')] = calibration['matrices'][label]['sha256']
    recorded = binding['checked_input_sha256']
    if recorded != {str(path):digest for path,digest in required.items()}:
        raise ValueError('Screen input ledger differs from independently required exact provenance')
    for path,digest in recorded.items():
        path = Path(path)
        if not path.is_absolute() or not path.is_relative_to(ROOT):
            raise ValueError('Out-of-repository screen input path')
        if path in fixed and fixed[path] != digest:
            raise ValueError('Contradictory screen provenance')
        checked_file(path,digest)
    greedy_errors, joint_errors, proofs = {}, {}, {}
    for label in LABELS:
        layer_text, part = label.split('.')
        layer = int(layer_text.removeprefix('layer'))
        shape = [18560,4096] if part == 'in_proj' else [4096,8192]
        seed = 1000+2*layer+(part == 'out_proj')
        key = f'backbone.layers.{layer}.mixer.{part}.weight'
        origin = inputs if part == 'in_proj' else outputs
        origin_dir = ROOT/'artifacts'/('input_axis_residual_v1' if part == 'in_proj' else 'output_axis_residual_v1')
        original = origin_dir/(label+'.e8')
        original_row = origin['matrices'][label]
        checked_file(original,original_row['sha256'])
        if original_row['decoded_fp16_sha256'] != accepted[key]:
            raise ValueError('Current native decoded matrix identity differs')
        hessian = calibration['matrices'][label]
        hpath = ROOT/'calibration/v1'/hessian['file']
        if (hessian['file'] != label+'.pt' or hessian['shape'] != [shape[1],shape[1]]
                or hessian['sha256'] != parent['matrices'][label]['hessian_sha256']
                or original_row['hessian_sha256'] != hessian['sha256']):
            raise ValueError('Original TRAIN Hessian does not match both parents')
        checked_file(hpath,hessian['sha256'])
        proofs[label] = {}
        for method, entries in (('greedy',report['greedy_replay']),('joint',report['matrices'])):
            row = entries[label]
            filename = f'{method}/{label}.e8'
            file_row = report['files'][filename]
            if (row['file'] != filename or row['shape'] != shape or row['seed'] != seed
                    or row['source_key'] != key or row['source_dtype'] != 'torch.bfloat16'
                    or row['source_tensor_sha256'] != original_row['source_tensor_sha256']
                    or row['hessian_sha256'] != hessian['sha256']
                    or row['original_file_sha256'] != original_row['sha256']
                    or row['index_bits'] != 20 or row['values_per_index'] != 8
                    or row['tune_iters'] != 2 or row['damping'] != .01 or row['scale_override'] != .9
                    or row['feedback'] is not True or row['disk_roundtrip_fp16_equal'] is not True
                    or row['metadata_equal'] is not True or row['raw_length_equal'] is not True
                    or row['packed_indices_equal'] is not True):
                raise ValueError(f'Fixed matrix recipe/source/readback receipt differs:{filename}')
            proof = compare_files(original,args.screen_dir/filename,shape,method=='greedy')
            if (row['bytes'] != proof['bytes'] or row['sha256'] != proof['sha256']
                    or row['layout'] != proof['layout'] or row['metadata_identity'] != proof['metadata_identity']
                    or file_row != {k:row[k] for k in ('bytes','sha256','decoded_fp16_sha256')}):
                raise ValueError('Actual file does not match complete receipt/ledger')
            digest = row['decoded_fp16_sha256']
            if not isinstance(digest,str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Malformed GPU native decoded identity')
            if (row['original_decoded_fp16_sha256'] != original_row['decoded_fp16_sha256']
                    or (method == 'greedy' and (digest != original_row['decoded_fp16_sha256']
                        or any(row[k] is not True for k in ('replay_indices_equal','replay_file_equal','replay_fp16_equal'))))):
                raise ValueError('Greedy replay/current native decoded identity differs')
            numerator = finite_nonnegative(row['original_h_squared_error'], 'original-H J')
            denominator = finite_nonnegative(row['original_h_denominator'], 'original-H denominator')
            if denominator <= 0:
                raise ValueError('Nonpositive original-H normalization')
            (greedy_errors if method=='greedy' else joint_errors)[label] = numerator
            proofs[label][method] = proof
        if report['greedy_replay'][label]['original_h_denominator'] != report['matrices'][label]['original_h_denominator']:
            raise ValueError('Original-H denominator changed between methods')
    values, gate = recompute_gate(greedy_errors,joint_errors)
    if report['gate'] != gate:
        raise ValueError('Declared screen gate differs from independent squared-error recomputation')
    for label,value in values.items():
        if report['matrices'][label]['relative_squared_error_reduction'] != value:
            raise ValueError('Per-matrix squared-error reduction differs')
        row = report['matrices'][label]
        stats = row['selector_stats']
        m,n = row['shape']
        expected_calls = 3*(n//8)
        if (type(stats['calls']) is not int or stats['calls'] != expected_calls
                or type(stats['query_rows']) is not int or stats['query_rows'] != expected_calls*m
                or type(stats['strictly_improved_rows']) is not int
                or not 0 <= stats['strictly_improved_rows'] <= stats['query_rows']
                or stats['local_nonregression_checked'] is not True or stats['row_batch'] != 4096
                or stats['shift_order'] != list(range(16)) or stats['incumbent_scope'] != 'full original query'
                or stats['tie_rule'] != 'greedy incumbent first, then first nibble; strict-less only'):
            raise ValueError('Fixed selector scope/LDLQ invocation receipt differs')
        old_local = finite_nonnegative(stats['greedy_local_squared_error_sum'],'greedy query score')
        new_local = finite_nonnegative(stats['joint_local_squared_error_sum'],'joint query score')
        if new_local > old_local:
            raise ValueError('Recorded local incumbent score increased')
    replay_bytes = sum(proofs[label]['greedy']['bytes'] for label in LABELS)
    joint_bytes = sum(proofs[label]['joint']['bytes'] for label in LABELS)
    expected_storage = {'greedy_replay_bytes':replay_bytes, 'joint_pilot_bytes':joint_bytes,
        'physical_data_bytes':replay_bytes+joint_bytes, 'raw_replacement_delta_bytes':0,
        'full_candidate_built':False, 'current_resolved_raw_data_bytes':3138928792}
    if joint_bytes != replay_bytes or report['storage'] != expected_storage:
        raise ValueError('Actual fixed-capacity/physical screen accounting differs')
    return {'manifest_sha256':sha(manifest_path), 'fixed_labels':list(LABELS),
        'file_count':12, 'greedy_replay_files':6, 'joint_files':6,
        'candidate_data_bytes':sum(proofs[label]['joint']['bytes'] for label in LABELS),
        'greedy_replay_data_bytes':sum(proofs[label]['greedy']['bytes'] for label in LABELS),
        'per_matrix':proofs, 'relative_squared_error_reduction':values, 'recomputed_gate':gate,
        'inherited395_verified':True, 'inherited_file_count':5, 'bound_input_count':len(recorded),
        'gpu_metric_scope':'Squared errors and native FP16 hashes are frozen-screen GPU receipts; CPU audit independently checks provenance, files and gate arithmetic.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-dir', type=Path, default=ROOT/'artifacts/joint_axis_screen_v1')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/joint_axis_screen_v1_audit.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    args.screen_dir, args.report = args.screen_dir.resolve(), args.report.resolve()
    if args.report.parent != ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('Use a fresh audit report under reports/')
    torch.set_num_threads(8)
    report = {'format': 'MAMBA2_JOINT_AXIS_SCREEN_CPU_AUDIT_V1', 'complete': False,
        'script_sha256': sha(__file__), 'protocol_sha256': PROTOCOL_SHA,
        'scope': 'Actual-file/provenance audit and reported-GPU-objective gate recomputation; no repeated GPU decode or dense-H computation.',
        'gpu_work_performed': False}
    started = time.monotonic()
    try:
        report['self_test'] = self_test()
        if not args.self_test:
            report['audit'] = audit(args)
        if torch.cuda.is_initialized():
            raise ValueError('Auditor must remain CPU only')
        report.update(complete=True, mode='self_test' if args.self_test else 'audit', cuda_initialized=False)
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        pending = args.report.with_suffix('.json.tmp')
        pending.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        pending.replace(args.report)
    print(json.dumps({'complete': report['complete'], 'report': str(args.report),
        'gate': report.get('audit', {}).get('recomputed_gate')}), flush=True)


if __name__ == '__main__':
    main()
