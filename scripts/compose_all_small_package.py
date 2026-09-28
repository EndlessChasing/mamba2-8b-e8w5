#!/usr/bin/env python3
"""Compose the evaluated all-small model into a standalone flat raw package.

CPU only. This creates a new manifest, never edits either input artifact, and
does not pack, score, train, publish, or support learned prototype tables.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba_e8w5.runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256
from mamba_e8w5.small_training import selected_inventory

FORMAT = 'MAMBA2_E8W5_ALL_SMALL_RESOLVED_V1'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
QUALITY_SHA = '957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
LABELS = tuple(f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj'))
REQUIRED_FILES = {label+'.e8' for label in LABELS} | {
    'embedding.uniform', 'lm_head.uniform', 'other_fp16.pt', 'config.json', 'e8_codebook.bin'}
FROZEN_CODE = {
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/small_training.py': '6e98858460534c4047f7bf75bd62eaf7769d7e46919cd9de6868fd6177a80086',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().astype('<f2', copy=False).tobytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def plain_file(directory, name):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f'Expected real directory: {directory}')
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Expected flat member name')
    path = directory/name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular member: {path}')
    return path


def check_file(path, entry):
    if (type(entry.get('bytes')) is not int or entry['bytes'] < 0
            or not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64
            or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']):
        raise ValueError(f'File differs from evaluated ledger: {path}')


def check_inventory(directory, names):
    if {p.name for p in directory.iterdir()} != set(names):
        raise ValueError(f'Unlisted or missing file: {directory}')
    for name in names:
        plain_file(directory, name)


def audit_small(values, inventory, expected_hashes):
    """Check actual CPU FP16 bytes, including signed zeros; no model creation."""
    if not isinstance(values, dict) or set(values) != set(inventory) or set(expected_hashes) != set(inventory):
        raise ValueError('Small tensor inventory differs')
    records = {}
    for name, entry in inventory.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float16 or list(value.shape) != entry['shape']
                or value.numel() != entry['numel'] or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid CPU small tensor: {name}')
        digest = tensor_sha(value)
        if digest != expected_hashes[name]:
            raise ValueError(f'Small tensor differs from evaluated export: {name}')
        records[name] = {'shape': list(value.shape), 'numel': value.numel(),
                         'dtype': 'float16', 'decoded_fp16_sha256': digest}
    return records


def verify_inputs(parent_dir, small_dir, quality_path):
    """Verify the fixed measured candidate, including every inherited file."""
    pp, sp = plain_file(parent_dir, 'manifest.json'), plain_file(small_dir, 'manifest.json')
    if quality_path.is_symlink() or not quality_path.is_file():
        raise ValueError('Expected regular quality report')
    for path, digest in ((pp, PARENT_SHA), (sp, SMALL_SHA), (quality_path, QUALITY_SHA)):
        if sha(path) != digest:
            raise ValueError(f'Pinned artifact identity differs: {path}')
    for name, digest in FROZEN_CODE.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Frozen code differs: {name}')
    parent, small, quality = (json.loads(p.read_text()) for p in (pp, sp, quality_path))
    if (parent.get('complete') is not True or small.get('complete') is not True
            or parent.get('model_config') != MODEL_CONFIG or small.get('model_config') != MODEL_CONFIG
            or parent.get('source_checkpoint_sha256') != SOURCE_CHECKPOINT_SHA256
            or parent.get('tokenizer_sha256') != TOKENIZER_SHA256
            or small.get('parent_manifest_sha256') != PARENT_SHA
            or set(parent['files']) != REQUIRED_FILES or set(small['files']) != {'other_fp16.pt'}):
        raise ValueError('Wrong complete all-small model/parent')
    check_inventory(parent_dir, REQUIRED_FILES | {'manifest.json'})
    check_inventory(small_dir, {'manifest.json', 'other_fp16.pt'})
    inherited = {name: entry for name, entry in parent['files'].items() if name != 'other_fp16.pt'}
    if small.get('inherited_files') != inherited or len(inherited) != 116:
        raise ValueError('Inherited file ledger differs')
    if (quality.get('complete') is not True or quality.get('same_process') is not True
            or quality['inputs']['parent_manifest']['sha256'] != PARENT_SHA
            or quality['inputs']['small_manifest']['sha256'] != SMALL_SHA
            or quality['inputs']['small_export']['sha256'] != SMALL_VALUES_SHA
            or quality['parent_package_receipt']['verified_files'] != parent['files']
            or quality['small_overlay_receipt']['overlay_manifest_sha256'] != SMALL_SHA
            or quality['small_overlay_receipt']['training_receipt'] != small['training_receipt']):
        raise ValueError('Full validation did not evaluate this exact parent/export')
    protocol = quality['protocol']
    result = quality['results']['small_e8w5']
    if (protocol['split'] != 'validation' or protocol['target_tokens'] != 264764
            or protocol['number_of_windows'] != 130 or protocol['maximum_targets_per_window'] != 2048
            or result['target_tokens'] != 264764 or len(result['windows']) != 130
            or not math.isclose(math.exp(result['nll']/264764), result['ppl'], rel_tol=1e-12)):
        raise ValueError('Expected the completed fixed full-validation result')
    inventory = selected_inventory()
    mapping = parent['parameter_mapping']
    if (small['selected_small_tensors'] != inventory or small['parameter_mapping'] != mapping
            or len(mapping) != 507 or parent['total_parameter_count'] != 8236999680
            or len(inventory) != 393 or sum(x['numel'] for x in inventory.values()) != 3580928
            or {name for name, file in mapping.items() if file == 'other_fp16.pt'} != set(inventory)):
        raise ValueError('Expected exactly393 small +114 frozen parameters')
    loaded = quality['small_loaded_tensor_audit']
    frozen = loaded['frozen']['unchanged_content_sha256']
    if loaded['coverage'] != {'tensors':507, 'parameters':8236999680} or len(frozen) != 114:
        raise ValueError('Full evaluated model coverage differs')
    if set(frozen) != set(mapping)-set(inventory):
        raise ValueError('Frozen114 hash coverage differs')
    paths = {}
    files = {**inherited, 'other_fp16.pt': small['files']['other_fp16.pt']}
    if files['other_fp16.pt']['sha256'] != SMALL_VALUES_SHA:
        raise ValueError('Replacement small archive identity differs')
    for name, entry in files.items():
        path = plain_file(small_dir if name == 'other_fp16.pt' else parent_dir, name)
        check_file(path, entry)
        paths[name] = path
    if json.loads(paths['config.json'].read_text()) != MODEL_CONFIG:
        raise ValueError('Stored model configuration differs')
    records = audit_small(torch.load(paths['other_fp16.pt'], map_location='cpu', weights_only=True),
        inventory, loaded['small']['loaded_and_export_fp16_sha256'])
    for label in LABELS:
        row = parent['matrices'][label]
        if (row['source_key'] not in frozen or row['decoded_fp16_sha256'] != frozen[row['source_key']]
                or mapping[row['source_key']] != label+'.e8'
                or any(row[k] != files[label+'.e8'][k] for k in ('bytes', 'sha256'))):
            raise ValueError('E8 projection differs from evaluated frozen tensor')
    if sum(entry['bytes'] for entry in files.values()) != quality['storage']['logical_candidate_data_bytes']:
        raise ValueError('Resolved file bytes differ from measured candidate')
    return parent, small, quality, files, paths, records, frozen


def make_manifest(parent, small, quality, files, records, frozen):
    """Explicit whitelist: never clone stale original-small metrics or hashes."""
    matrices = {label: {key: parent['matrices'][label][key]
        for key in ('source_key', 'shape', 'bytes', 'sha256', 'decoded_fp16_sha256',
                    'index_bits', 'values_per_index')} for label in LABELS}
    vocab = {label: {key: parent['vocabularies'][label][key]
        for key in ('source_key', 'shape', 'bytes', 'sha256')} for label in ('embedding', 'lm_head')}
    for row in vocab.values():
        row.update(bits=5, group_size=128, decoded_fp16_sha256=frozen[row['source_key']])
    hashes = {**frozen, **{name: row['decoded_fp16_sha256'] for name, row in records.items()}}
    return {'format': FORMAT, 'complete': True, 'model_config': MODEL_CONFIG,
        'source_checkpoint_sha256': SOURCE_CHECKPOINT_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
        'total_parameter_count': 8236999680, 'total_parameter_tensors': 507,
        'binding': {'parent_manifest_sha256':PARENT_SHA, 'small_manifest_sha256':SMALL_SHA,
            'small_values_sha256':SMALL_VALUES_SHA, 'quality_report_sha256':QUALITY_SHA,
            'composition_script_sha256':sha(__file__), 'frozen_code_sha256':FROZEN_CODE,
            # Archived restore smoke resolves these names under mamba_e8w5/.
            'code_sha256':{name:FROZEN_CODE['mamba_e8w5/'+name] for name in ('runtime.py','codec.py')},
            'source':parent['binding']['source']},
        'files': files, 'parameter_mapping': parent['parameter_mapping'],
        'matrices': matrices, 'vocabularies': vocab, 'small_tensors': records,
        'decoded_fp16_sha256': hashes,
        'composition': {'inherited_files':116, 'inherited_large_tensors':114,
            'replacement_file':'other_fp16.pt', 'replacement_tensors':393,
            'replacement_parameters':3580928, 'prototype_tables':0,
            'runtime':'mamba_e8w5.runtime.load_quantized_model',
            'source_checkpoint_required_for_inference':False,
            'training_checkpoints_required_for_inference':False},
        'quality': {'report_sha256':QUALITY_SHA, 'arm':'small_e8w5',
            'ppl':quality['results']['small_e8w5']['ppl'],
            'nll':quality['results']['small_e8w5']['nll'], 'protocol':quality['protocol'],
            'dataset':quality['dataset'], 'window_plan_sha256':quality['window_plan_sha256'],
            'scope':'Previously measured full validation on the exact resolved tensor bytes; not a new score, MK/test result, or release approval.'},
        'data_file_bytes':sum(entry['bytes'] for entry in files.values()),
        'manifest_bytes':0, 'raw_package_bytes':0,
        'storage_scope':'117 resolved raw data files plus this manifest; tokenizer, release source/licenses/reports and container overhead excluded.',
        'note':'All-small final1024 FP16 values replace original small tensors. Original two-sweep E8 and independent W5 vocabularies remain byte-identical. No prototype tables.'}


def materialize(source, target, entry, mode):
    """Never write/chmod the source or overwrite a target, including hardlinks."""
    if mode not in ('auto', 'hardlink', 'copy'):
        raise ValueError('Unknown materialization mode')
    method = 'copy'
    if mode != 'copy':
        try:
            os.link(source, target)
            method = 'hardlink'
        except OSError as error:
            if mode == 'hardlink' or error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS):
                raise
    if method == 'copy':
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=8 << 20)
    check_file(target, entry)
    return method


def finalize_manifest(manifest):
    for _ in range(16):
        raw = encoded(manifest)
        if manifest['manifest_bytes'] == len(raw):
            return raw
        manifest['manifest_bytes'] = len(raw)
        manifest['raw_package_bytes'] = manifest['data_file_bytes']+len(raw)
    raise RuntimeError('Manifest byte accounting did not converge')


def compose(args):
    for path in (args.parent_dir, args.small_dir, args.quality_report, args.output):
        if path.is_symlink():
            raise ValueError('Input aliases via symlinks are not accepted')
    parent_dir, small_dir, quality_path, output = (
        p.resolve() for p in (args.parent_dir, args.small_dir, args.quality_report, args.output))
    staging = output.with_name(output.name+'.building')
    if output.exists() or staging.exists():
        raise FileExistsError('Output/staging exists; composition never resumes or overwrites')
    for directory in (output, staging):
        if any(directory == source or source in directory.parents for source in (parent_dir, small_dir)):
            raise ValueError('Output must be outside both immutable input directories')
    bound = {parent_dir/'manifest.json':PARENT_SHA, small_dir/'manifest.json':SMALL_SHA,
             quality_path:QUALITY_SHA}
    parent, small, quality, files, paths, records, frozen = verify_inputs(parent_dir, small_dir, quality_path)
    manifest = make_manifest(parent, small, quality, files, records, frozen)
    staging.mkdir(parents=True)
    methods = {}
    for name in sorted(files):
        methods[name] = materialize(paths[name], staging/name, files[name], args.mode)
    with (staging/'manifest.json').open('xb') as stream:
        stream.write(finalize_manifest(manifest))
    check_inventory(staging, REQUIRED_FILES | {'manifest.json'})
    # Validate input and output identities after composition; never promote an
    # input race, even when --mode copy created independent output inodes.
    for path, digest in bound.items():
        if sha(path) != digest:
            raise ValueError('Bound manifest/report changed during composition')
    for name, entry in files.items():
        check_file(paths[name], entry)
        check_file(staging/name, entry)
    if sum(p.stat().st_size for p in staging.iterdir()) != manifest['raw_package_bytes']:
        raise ValueError('Actual output bytes differ from complete ledger')
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    return {'complete':True, 'output':str(output), 'manifest_sha256':sha(output/'manifest.json'),
        'data_files':117, 'actual_small_tensors_verified':393, 'inherited_files_verified':116,
        'frozen_large_tensor_hashes_bound_to_evaluation':114, 'parameter_tensors':507,
        'parameters':8236999680, 'quality_report_sha256':QUALITY_SHA,
        'data_file_bytes':manifest['data_file_bytes'], 'manifest_bytes':manifest['manifest_bytes'],
        'raw_package_bytes':manifest['raw_package_bytes'],
        'materialization_counts':{name:sum(v == name for v in methods.values()) for name in ('hardlink','copy')},
        'gpu_used':False, 'full_pack_performed':False, 'quality_remeasured':False,
        'note':'No source checkpoint loaded. Output is immutable by workflow, not an OS write lock; hardlinks share input inodes and must never be edited.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', type=Path, default=ROOT/'artifacts/e8w5_v1')
    parser.add_argument('--small-dir', type=Path, default=ROOT/'artifacts/small_compensation_v1')
    parser.add_argument('--quality-report', type=Path, default=ROOT/'reports/small_compensation_v1_eval.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('auto','hardlink','copy'), default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(compose(args), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
