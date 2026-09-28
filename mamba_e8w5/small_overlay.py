"""Strict final1024 all-small overlay; original E8/W5 files remain inherited."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import torch

from . import norm_overlay as norm
from .runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256, token_digest
from .small_training import HYPERPARAMETERS, selected_inventory, training_schedule, eligible_window_starts

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'MAMBA2_E8W5_SMALL_OVERLAY_V1'
INITIAL_MANIFEST_SHA256 = '1fa9d35ff56a08e2f7ddf258bfa1fb61bc2e83ae2af404ab2f68a781821ac4d8'
INITIAL_OTHER_SHA256 = '8a98319ad513fceaf48673ff4199a8a22b5c1046c64b91bbf3bbbb09f5d84530'
PROTOCOL_SHA256 = 'f57229f979f8bc94def0780f87aa48ae0b83f7d30dc29fef527b46aaa2be0c31'
CALIBRATION_MANIFEST_SHA256 = '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
DATA_MANIFEST_SHA256 = '88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709'
TRAINING_TOKENS_SHA256 = 'f9d2321f580832b19d62ec498841b552b5f66b90be16a195c004b30be1f6d0b8'
ORIGINAL_CALIBRATION_STARTS = [0, 81665, 163330, 244996, 326661, 408327, 489992, 571658,
    653323, 734989, 816654, 898320, 979985, 1061651, 1143316, 1224982, 1306647, 1388313,
    1469978, 1551644, 1633309, 1714975, 1796640, 1878306, 1959971, 2041637, 2123302,
    2204968, 2286633, 2368299, 2449964, 2531630]
INITIALIZATION = {'norm_manifest_sha256': INITIAL_MANIFEST_SHA256, 'other_fp16_sha256': INITIAL_OTHER_SHA256}
sha = norm.sha


def digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid SHA-256: {label}')
    return value


def plain_file(directory, name):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f'Unsafe directory: {directory}')
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError(f'Unsafe filename: {name}')
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular file: {path}')
    return path


def checked_file(directory, name, entry):
    if not isinstance(entry, dict) or type(entry.get('bytes')) is not int or entry['bytes'] < 0:
        raise ValueError(f'Invalid file byte count: {name}')
    digest(entry.get('sha256'), name)
    path = plain_file(directory, name)
    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError(f'File integrity mismatch: {name}')
    return path


def compare_small(before, after):
    inventory = selected_inventory()
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(inventory) or set(after) != set(inventory):
        raise ValueError('Expected exactly393 small tensor keys')
    changed, hashes = [], {}
    for name, entry in inventory.items():
        a, b = before[name], after[name]
        for value in (a, b):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float16
                    or list(value.shape) != entry['shape'] or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid small FP16 tensor: {name}')
        if not norm.bitwise_equal(a, b): changed.append(name)
        hashes[name] = norm.tensor_sha(b)
    return {'tensor_count': len(inventory), 'parameter_count': sum(x['numel'] for x in inventory.values()),
            'actual_changed_keys': changed, 'actual_changed_tensor_count': len(changed), 'fp16_tensor_sha256': hashes}


def verify_training_data(directory):
    directory = Path(directory)
    mp = plain_file(directory, 'manifest.json')
    if sha(mp) != DATA_MANIFEST_SHA256: raise ValueError('Prepared training manifest differs')
    manifest = json.loads(mp.read_text())
    if {p.name for p in directory.iterdir()} != {'manifest.json', 'training_tokens.pt'}:
        raise ValueError('Unlisted training data files')
    if (manifest.get('format') != 'MAMBA2_SMALL_TRAIN_WINDOWS_V1' or manifest.get('complete') is not True
            or manifest.get('split') != 'train' or manifest.get('evaluation_data_used') is not False
            or type(manifest.get('nwin')) is not int or manifest['nwin'] != 256
            or type(manifest.get('seqlen')) is not int or manifest['seqlen'] != 2048):
        raise ValueError('Wrong training data protocol')
    dataset = manifest.get('dataset', {})
    required = {'dataset': 'Salesforce/wikitext', 'configuration': 'wikitext-2-raw-v1', 'split': 'train',
        'revision_argument': 'b08601e04326c79dfdd32d625aee71d232d685c3', 'total_tokens': 2533678,
        'tokenizer_sha256': TOKENIZER_SHA256, 'automatic_special_tokens': False,
        'document_join': 'two newline characters',
        'text_sha256': 'aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5',
        'token_stream_sha256_int64le': '7205aba52233c95b0a2b84cbf6cf6ae5e8b58463e6ecf51f6a694f1339e5f172'}
    if any(dataset.get(k) != v for k, v in required.items()): raise ValueError('Pinned train dataset differs')
    if (manifest.get('original_calibration_manifest_sha256') != CALIBRATION_MANIFEST_SHA256
            or manifest.get('excluded_original_calibration_starts') != ORIGINAL_CALIBRATION_STARTS):
        raise ValueError('Original32 calibration exclusion identity differs')
    for key, value in [('unique_stored_window_tokens', 524288), ('unique_input_positions_per_pass', 524032),
                       ('targets_per_pass', 524032), ('four_pass_target_exposures', 2096128)]:
        if type(manifest.get(key)) is not int or manifest[key] != value:
            raise ValueError(f'Training position accounting differs: {key}')
    for key, path in [('preparation_source_sha256', ROOT/'scripts/prepare_small_training_data.py'),
                      ('selection_source_sha256', ROOT/'mamba_e8w5/small_training.py'),
                      ('calibration_loader_source_sha256', ROOT/'mamba_e8w5/calibration.py'),
                      ('runtime_source_sha256', ROOT/'mamba_e8w5/runtime.py')]:
        if manifest.get(key) != sha(path): raise ValueError(f'Data preparation implementation differs: {key}')
    if manifest.get('token_windows_file') != 'training_tokens.pt' or manifest.get('token_windows_file_sha256') != TRAINING_TOKENS_SHA256:
        raise ValueError('Wrong training token filename/identity')
    path = checked_file(directory, 'training_tokens.pt', {'bytes': manifest.get('token_windows_file_bytes'), 'sha256': TRAINING_TOKENS_SHA256})
    windows = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(windows, torch.Tensor) or windows.dtype != torch.int64 or list(windows.shape) != [256, 2048]:
        raise ValueError('Training token geometry differs')
    if int(windows.min()) < 0 or int(windows.max()) >= 256000: raise ValueError('Invalid training token IDs')
    if token_digest(windows.flatten().numpy()) != manifest['selected_tokens_sha256_int64le']:
        raise ValueError('Selected token digest differs')
    starts, audit = eligible_window_starts(2533678, ORIGINAL_CALIBRATION_STARTS)
    if (manifest.get('starts') != starts or any(type(x) is not int for x in manifest['starts'])
            or manifest.get('overlap') != audit or (audit['grid_blocks'], audit['excluded_grid_blocks'], audit['eligible_blocks']) != (1237, 62, 1175)):
        raise ValueError('Data starts/overlap audit differs')
    return manifest, windows


def verify_accounting(record, *, exposures=False):
    """Committed optimizer history; discarded replay attempts must be separate."""
    count, overflow, updates = (record.get(k) for k in ('attempts', 'overflow_retries', 'successful_updates'))
    if (any(type(x) is not int for x in (count, overflow, updates)) or updates != 1024
            or not 0 <= overflow <= 8 or count != 1024 + overflow):
        raise ValueError('Final training attempt/update accounting differs')
    schedule = record.get('schedule')
    if not isinstance(schedule, list) or any(type(x) is not int for x in schedule) or schedule != training_schedule():
        raise ValueError('Final training schedule differs')
    history = record.get('history')
    if not isinstance(history, list) or len(history) != count:
        raise ValueError('Committed attempt history differs')
    completed = skipped = 0
    for index, row in enumerate(history):
        if (not isinstance(row, dict) or type(row.get('attempt')) is not int or row['attempt'] != index+1
                or type(row.get('overflow')) is not bool or type(row.get('window')) is not int
                or completed >= 1024 or row['window'] != schedule[completed]
                or type(row.get('targets')) is not int or row['targets'] != 2047):
            raise ValueError('Committed attempt identity differs')
        skipped += int(row['overflow']); completed += int(not row['overflow'])
        if type(row.get('successful_updates_after')) is not int or row['successful_updates_after'] != completed:
            raise ValueError('Committed update counter differs')
    if completed != updates or skipped != overflow: raise ValueError('History totals differ')
    if exposures:
        for key, expected in [('final_step', 1024), ('successful_target_exposures', 2096128), ('attempted_target_exposures', count*2047)]:
            if type(record.get(key)) is not int or record[key] != expected:
                raise ValueError(f'Final exposure accounting differs: {key}')
    return {'final_step': updates, 'attempts': count, 'overflow_retries': overflow, 'committed_history_rows_verified': len(history)}


def verify_final_checkpoint(state, after, binding, training):
    if (not isinstance(state, dict) or state.get('format') != 'MAMBA2_SMALL_TRAINING_CHECKPOINT_V1'
            or state.get('resumable') is not True
            or state.get('binding') != binding or state.get('parent_manifest_sha256') != norm.PARENT_MANIFEST_SHA256
            or state.get('initialization') != INITIALIZATION
            or state.get('smoke_report_sha256') != training['smoke_report_sha256']):
        raise ValueError('Checkpoint provenance differs')
    accounting = verify_accounting(state)
    masters = state.get('masters')
    if not isinstance(masters, dict) or set(masters) != set(selected_inventory()) or set(after) != set(masters):
        raise ValueError('Checkpoint master coverage differs')
    for name, master in masters.items():
        if (not isinstance(master, torch.Tensor) or master.dtype != torch.float32
                or master.shape != after[name].shape or not torch.isfinite(master).all()
                or not norm.bitwise_equal(master.half(), after[name])):
            raise ValueError(f'FP16 export differs from saved master: {name}')
    return {**accounting, 'rounded_master_tensors_verified': len(masters),
            'rounded_master_parameters_verified': sum(x.numel() for x in masters.values())}


def verify_overlay(parent_dir, directory, *, initialization_dir=None, data_dir=None, protocol=None,
                   training_report=None, training_checkpoint=None, smoke_report=None):
    parent_dir, directory = Path(parent_dir), Path(directory)
    pp, cp = plain_file(parent_dir, 'manifest.json'), plain_file(directory, 'manifest.json')
    if sha(pp) != norm.PARENT_MANIFEST_SHA256: raise ValueError('Wrong E8/W5 parent')
    parent, candidate = json.loads(pp.read_text()), json.loads(cp.read_text())
    if parent.get('complete') is not True or candidate.get('complete') is not True or candidate.get('format') != FORMAT:
        raise ValueError('Incomplete or wrong small overlay')
    if candidate.get('parent_manifest_sha256') != norm.PARENT_MANIFEST_SHA256 or candidate.get('model_config') != MODEL_CONFIG:
        raise ValueError('Architecture/parent differs')
    if candidate.get('initialization') != INITIALIZATION: raise ValueError('Wrong training initialization')
    if candidate.get('selected_small_tensors') != selected_inventory() or candidate.get('parameter_mapping') != parent['parameter_mapping']:
        raise ValueError('Parameter mapping differs')
    if len(parent['parameter_mapping']) != 507 or parent['total_parameter_count'] != 8236999680: raise ValueError('Wrong total coverage')
    binding = candidate['binding']
    expected = {'source_checkpoint_sha256': SOURCE_CHECKPOINT_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
        'protocol_sha256': PROTOCOL_SHA256, 'small_overlay_source_sha256': sha(__file__),
        'training_data_manifest_sha256': DATA_MANIFEST_SHA256, 'training_tokens_sha256': TRAINING_TOKENS_SHA256,
        'trainer_source_sha256': sha(ROOT/'scripts/train_small_compensation.py'),
        'training_helper_source_sha256': sha(ROOT/'mamba_e8w5/small_training.py'),
        'norm_training_source_sha256': sha(ROOT/'mamba_e8w5/norm_training.py'),
        'norm_trainer_utility_source_sha256': sha(ROOT/'scripts/train_norm_compensation.py'),
        'norm_overlay_source_sha256': sha(norm.__file__), 'hyperparameters': HYPERPARAMETERS}
    for key, value in expected.items():
        if binding.get(key) != value: raise ValueError(f'Binding differs: {key}')
    for name, value in parent['binding']['code_sha256'].items():
        if sha(ROOT/'mamba_e8w5'/name) != value: raise ValueError(f'Frozen pipeline changed: {name}')
    for name, value in parent['binding']['quip_sha256'].items():
        if sha(ROOT/'third_party/quip-sharp'/name) != value: raise ValueError(f'Frozen QuIP changed: {name}')
    inherited = {name: entry for name, entry in parent['files'].items() if name != 'other_fp16.pt'}
    if candidate.get('inherited_files') != inherited or len(inherited) != 116: raise ValueError('Inherited E8/W5 ledger differs')
    if set(candidate.get('files', {})) != {'other_fp16.pt'}: raise ValueError('Only small FP16 file may be replaced')
    if {p.name for p in directory.iterdir()} != {'manifest.json', 'other_fp16.pt'}: raise ValueError('Unlisted overlay files')
    if {p.name for p in parent_dir.iterdir()} != set(parent['files']) | {'manifest.json'}: raise ValueError('Unlisted parent files')
    for name, entry in parent['files'].items(): checked_file(parent_dir, name, entry)
    path = checked_file(directory, 'other_fp16.pt', candidate['files']['other_fp16.pt'])
    base = torch.load(parent_dir/'other_fp16.pt', map_location='cpu', weights_only=True)
    after = torch.load(path, map_location='cpu', weights_only=True)
    tensors = compare_small(base, after)
    audit = {}
    if initialization_dir is not None:
        ip, iw = plain_file(initialization_dir, 'manifest.json'), plain_file(initialization_dir, 'other_fp16.pt')
        if sha(ip) != INITIAL_MANIFEST_SHA256 or sha(iw) != INITIAL_OTHER_SHA256:
            raise ValueError('Initialization file identity differs')
        initial = torch.load(iw, map_location='cpu', weights_only=True)
        audit['change_relative_to_initialization'] = compare_small(initial, after)
    if data_dir is not None:
        verify_training_data(data_dir); audit['training_data_verified'] = True
    if protocol is not None:
        if sha(protocol) != PROTOCOL_SHA256: raise ValueError('Protocol differs')
        audit['protocol_verified'] = True
    training = candidate['training_receipt']
    if type(training.get('final_step')) is not int or training['final_step'] != 1024: raise ValueError('Only final1024 is eligible')
    for key in ('checkpoint_sha256', 'report_sha256', 'smoke_report_sha256'): digest(training.get(key), key)
    final_report = None
    for file, kind in [(training_report, 'report'), (smoke_report, 'smoke_report')]:
        if file is None: continue
        if sha(file) != training[kind+'_sha256']: raise ValueError(f'{kind} digest differs')
        report = json.loads(Path(file).read_text())
        if (report.get('complete') is not True or report.get('binding') != binding
                or report.get('parent_manifest_sha256') != norm.PARENT_MANIFEST_SHA256
                or report.get('initialization') != INITIALIZATION
                or report.get('selected_small_tensors') != selected_inventory()):
            raise ValueError(f'{kind} input/code provenance differs')
        if kind == 'smoke_report':
            if report.get('passed') is not True: raise ValueError('Smoke did not pass')
        else:
            verify_accounting(report, exposures=True)
            if (report.get('final_checkpoint', {}).get('sha256') != training['checkpoint_sha256']
                    or report.get('smoke_report_sha256') != training['smoke_report_sha256']):
                raise ValueError('Final training receipt binding differs')
            final_report = report
        audit[kind+'_verified'] = True
    if training_checkpoint is not None:
        if sha(training_checkpoint) != training['checkpoint_sha256']: raise ValueError('Checkpoint digest differs')
        state = torch.load(training_checkpoint, map_location='cpu', weights_only=True)
        audit['checkpoint_export_receipt'] = verify_final_checkpoint(state, after, binding, training)
        if final_report is not None and any(state[key] != final_report[key] for key in ('attempts', 'overflow_retries', 'successful_updates', 'schedule', 'history')):
            raise ValueError('Checkpoint and training report histories differ')
        audit['checkpoint_rounded_masters_verified'] = 393
    resolved = {name: {'origin': 'parent', **entry} for name, entry in inherited.items()}
    resolved['other_fp16.pt'] = {'origin': 'overlay', **candidate['files']['other_fp16.pt']}
    logical = sum(entry['bytes'] for entry in resolved.values())
    return parent, candidate, {'parent_manifest_sha256': norm.PARENT_MANIFEST_SHA256,
        'overlay_manifest_sha256': sha(cp), 'binding': binding, 'training_receipt': training, 'provenance_audit': audit,
        'tensor_receipt_relative_to_raw_parent': tensors, 'resolved_parameter_tensors': 507, 'resolved_parameter_count': 8236999680,
        'resolved_files': resolved, 'resolved_file_ledger_sha256': hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'logical_candidate_data_bytes': logical, 'parent_raw_data_bytes': parent['data_file_bytes'],
        'replacement_file_bytes': path.stat().st_size, 'other_fp16_tensor_payload_bytes': sum(x.numel()*x.element_size() for x in after.values()),
        'archive_byte_delta': path.stat().st_size-parent['files']['other_fp16.pt']['bytes'],
        'manifest_bytes': {'parent': pp.stat().st_size, 'overlay': cp.stat().st_size},
        'physical_parent_plus_overlay_bytes': parent['data_file_bytes']+pp.stat().st_size+path.stat().st_size+cp.stat().st_size,
        'storage_scope': '116 original inherited files plus the full393-tensor replacement. Norm-v1 initialization, training data/checkpoints, tokenizer, code/licenses and reports are not included in these raw byte counts.'}


@torch.no_grad()
def apply_overlay(model, directory, manifest):
    if (manifest.get('format') != FORMAT or manifest.get('complete') is not True
            or manifest.get('parent_manifest_sha256') != norm.PARENT_MANIFEST_SHA256
            or manifest.get('model_config') != MODEL_CONFIG or manifest.get('selected_small_tensors') != selected_inventory()):
        raise ValueError('Incomplete or wrong small overlay')
    mp = plain_file(directory, 'manifest.json')
    if json.loads(mp.read_text()) != manifest: raise ValueError('Supplied manifest differs from disk')
    if model._package_receipt.get('manifest_sha256') != norm.PARENT_MANIFEST_SHA256: raise ValueError('Wrong loaded raw parent')
    parameters = dict(model.named_parameters())
    if len(parameters) != 507 or sum(p.numel() for p in parameters.values()) != 8236999680:
        raise ValueError('Loaded model coverage differs')
    path = checked_file(directory, 'other_fp16.pt', manifest['files']['other_fp16.pt'])
    values = torch.load(path, map_location='cpu', weights_only=True)
    before = {name: parameters[name].detach().cpu() for name in selected_inventory()}
    comparison = compare_small(before, values)
    fixed = {name: id(value) for name, value in parameters.items() if name not in selected_inventory()}
    if len(fixed) != 114: raise ValueError('Wrong frozen projection/vocabulary coverage')
    for name, value in values.items():
        owner, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(value.to(parameters[name].device), requires_grad=False))
    loaded = {}
    for name, expected in values.items():
        actual = model.get_parameter(name)
        if not norm.bitwise_equal(actual, expected): raise ValueError(f'Reloaded small tensor differs: {name}')
        loaded[name] = norm.tensor_sha(actual)
    if any(id(model.get_parameter(name)) != identity for name, identity in fixed.items()):
        raise ValueError('Unrelated parameter object changed')
    if any(p.dtype != torch.float16 or p.requires_grad for p in model.parameters()):
        raise ValueError('Candidate is not a frozen FP16 model')
    model._small_overlay_receipt = {'manifest_sha256': sha(mp), 'overlay_manifest_sha256': sha(mp),
        'parent_manifest_sha256': norm.PARENT_MANIFEST_SHA256, 'unchanged_parameter_objects': 114,
        'resolved_parameter_tensors': 507, 'resolved_parameter_count': 8236999680,
        'tensor_comparison_relative_to_preapplication_model': comparison, 'reloaded_fp16_tensor_sha256': loaded}
    return model
