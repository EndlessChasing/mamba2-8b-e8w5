#!/usr/bin/env python3
"""Independent final1024 rank-4 factor verification and native three-arm PPL.

No training bank or trainer is imported. CPU verification binds actual exported
FP16 pairs to all224 final FP32 masters, fixed data/schedule, and frozen base.
Quality explicitly merges factor files into native weights with highest FP32
precision; it never executes a two-linear branch or the training forward path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5 import small_overlay
from mamba_e8w5.low_rank_residual import read_factors, factor_layout, validate_factors, FORMAT
from mamba_e8w5.runtime import (MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256,
    SentencePieceTokenizer, load_source_model, load_quantized_model, token_digest,
    environment_receipt, gpu_memory_receipt)
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from evaluate_small_compensation import (sha, write_json, window_plan, check_arm,
    audit_parameter_coverage, audit_selected_values, audit_frozen_values)
from audit_decoded import audit_loaded_model, tensor_sha_fp16

PROTOCOL_SHA = 'b732c24692ae5dc91c5c67c1a0e0a0c435be3402959bfe5db874c67246836cf2'
DATA_SHA = 'e168e1d90ea7cd9c14c8210f13b5e6b569612c82dcaf832100c250cb72d24bef'
TOKEN_FILE_SHA = '539684ba450894a2e56d4f958b69942bacd7e6260ef349684c2c965d273012b1'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
BASELINE_SHA = '957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
HELPER_SHA = '2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be'
FOUNDATION_SHA = '050bc995c14477189b5b919d8d75c31e6c18dd4815d0de54cb72da39f6e083d6'
LABELS = tuple(f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj'))
ARMS = ('source_fp16', 'best_small_e8w5', 'low_rank_e8w5')
TARGETS = 264764
FACTOR_PAYLOAD_BYTES = 15654912
FACTOR_FILE_BYTES = 15658496
TRAIN_CODE_PATHS = {'scripts/train_low_rank_compensation.py', 'mamba_e8w5/low_rank_training.py',
    'mamba_e8w5/low_rank_residual.py', 'scripts/prepare_low_rank_training_data.py'}
FROZEN = {
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/small_overlay.py': '3512c1b232c4039c16a26ffc0fe8c2eac1be6bb9c704ccbcb1d4e8e7788a1286',
    'mamba_e8w5/norm_training.py': '25ac3ec6ba167902014018f99fbf8dbbc659e0e4903f80baf28e28a4a3e74087',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
}
DATA_UPSTREAM = {
    'calibration/v1/manifest.json': '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab',
    'training_data/small_compensation_v1/manifest.json': '88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709',
    'reports/projection_crossmoment_pilot_v1.json': '034a161df40fa4dac5be99b25545f9087da6695accc414a852fafe0e47228f87',
    'training_data/prototype_compensation_v1/manifest.json': 'a485424d1b7894eb1d5a6361ab4c25fdb4f6e5bedc56d7a15a7e4add2c454bcd',
}
CALIBRATION_SHA = 'b5a7356a2ab769538940aed218653a092a7055aa26a9ec6f62d3783b9ac78ee5'


def expected_inventory():
    return {label: {'file': label + '.lrf',
        **factor_layout(*( (18560, 4096) if label.endswith('in_proj') else (4096, 8192) ), 4),
        'source_key': 'backbone.layers.' + label[5:].replace('.', '.mixer.', 1) + '.weight'}
        for label in LABELS}


def expected_schedule():
    generator = torch.Generator(device='cpu').manual_seed(20260928)
    return [index for _ in range(4) for index in torch.randperm(256, generator=generator).tolist()]


def verify_accounting(state):
    schedule = expected_schedule()
    actual = state.get('schedule')
    if (not isinstance(actual, list) or actual != schedule or any(type(v) is not int for v in actual)
            or type(state.get('successful_updates')) is not int or state['successful_updates'] != 1024
            or type(state.get('attempts')) is not int or type(state.get('overflow_retries')) is not int):
        raise ValueError('Only the declared final1024 schedule is eligible')
    successes = overflows = 0
    history = state.get('history')
    if not isinstance(history, list):
        raise ValueError('Attempt history must be a list')
    for number, row in enumerate(history, 1):
        if (not isinstance(row, dict) or successes >= 1024 or type(row.get('attempt')) is not int
                or row['attempt'] != number or type(row.get('overflow')) is not bool
                or type(row.get('schedule_entry')) is not int or row['schedule_entry'] != schedule[successes]
                or type(row.get('targets')) is not int or row['targets'] != 2047
                or type(row.get('successful_updates_after')) is not int):
            raise ValueError('Attempt history differs from the declared schedule')
        overflows += int(row['overflow'])
        successes += int(not row['overflow'])
        if row['successful_updates_after'] != successes:
            raise ValueError('Incorrect successful-update history')
    if (successes != 1024 or overflows > 8 or len(history) != 1024 + overflows
            or state['attempts'] != len(history) or state['overflow_retries'] != overflows):
        raise ValueError('Final update/overflow accounting differs')
    return {'successful_updates': 1024, 'attempts': len(history), 'overflow_retries': overflows,
        'successful_target_exposures': 2096128, 'attempted_target_exposures': len(history) * 2047}


def verify_checkpoint_factors(state, factors, binding, frozen_hashes, smoke_sha):
    if (state.get('format') != 'MAMBA2_LOW_RANK_TRAINING_CHECKPOINT_V1'
            or state.get('binding') != binding or state.get('parent_manifest_sha256') != PARENT_SHA
            or state.get('small_manifest_sha256') != SMALL_SHA or state.get('smoke_report_sha256') != smoke_sha
            or state.get('frozen_model_sha256') != frozen_hashes or state.get('resumable') is not False
            or state.get('reason') != 'final'):
        raise ValueError('Final checkpoint provenance differs')
    counts = verify_accounting(state)
    master_names = {label + '.' + part for label in LABELS for part in ('B', 'A')}
    if set(state.get('masters', {})) != master_names or set(factors) != set(LABELS):
        raise ValueError('Exactly112 factor pairs and224 final masters are required')
    hashes = {}
    inventory = expected_inventory()
    parameters = 0
    for label in LABELS:
        B, A = factors[label]
        layout = validate_factors(B, A)
        if layout['shape'] != inventory[label]['shape'] or layout['rank'] != 4:
            raise ValueError(f'Invalid final factor shape/rank: {label}')
        for part, value in (('B', B), ('A', A)):
            name = label + '.' + part
            master = state['masters'][name]
            if (not isinstance(master, torch.Tensor) or master.dtype != torch.float32
                    or master.layout != torch.strided or list(master.shape) != inventory[label][part + '_shape']
                    or not torch.isfinite(master).all()):
                raise ValueError(f'Invalid final master: {name}')
            rounded = master.half()
            if not torch.isfinite(rounded).all() or tensor_sha_fp16(rounded) != tensor_sha_fp16(value):
                raise ValueError(f'Final master.half() differs from serialized factor: {name}')
            hashes[name] = tensor_sha_fp16(value)
            parameters += value.numel()
    if parameters != 7827456:
        raise ValueError('Final factor parameter capacity differs')
    return {**counts, 'verified_factor_files': 112, 'verified_masters': 224,
        'verified_parameters': parameters, 'rounded_master_equals_serialized_fp16_sha256': hashes}


def checked_file(directory, name, receipt):
    if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected safe flat file name')
    path = Path(directory) / name
    if (type(receipt.get('bytes')) is not int or path.is_symlink() or not path.is_file()
            or path.stat().st_size != receipt['bytes'] or sha(path) != receipt['sha256']):
        raise ValueError(f'File identity differs: {name}')
    return path


def verify_training_data(directory):
    path = Path(directory) / 'manifest.json'
    if sha(path) != DATA_SHA:
        raise ValueError('Fresh training-data manifest differs')
    data = json.loads(path.read_text())
    if (data.get('format') != 'MAMBA2_LOW_RANK_TRAIN_WINDOWS_V1' or data.get('complete') is not True
            or data.get('split') != 'train' or data.get('evaluation_data_used') is not False
            or data.get('protocol_sha256') != PROTOCOL_SHA or data.get('nwin') != 256
            or data.get('seqlen') != 2048 or data.get('source_checkpoint_sha256') != SOURCE_CHECKPOINT_SHA256
            or data.get('tokenizer_sha256') != TOKENIZER_SHA256 or data.get('excluded_manifests_sha256') != DATA_UPSTREAM
            or data.get('schedule') != expected_schedule() or data.get('schedule_seed') != 20260928
            or data.get('preparation_source_sha256') != sha(ROOT/'scripts/prepare_low_rank_training_data.py')
            or data.get('calibration_loader_source_sha256') != CALIBRATION_SHA
            or data.get('runtime_source_sha256') != FROZEN['mamba_e8w5/runtime.py']):
        raise ValueError('Training-data scope or provenance differs')
    documents = []
    for name, digest in DATA_UPSTREAM.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Exclusion provenance differs: {name}')
        document = json.loads((ROOT/name).read_text())
        if document.get('complete') is not True or document['dataset'] != data['dataset']:
            raise ValueError('TRAIN dataset provenance differs')
        documents.append(document)
    cal, small, pilot, prototype = documents
    excluded = {'original_calibration': cal['starts'], 'all_small_training': small['starts'],
        'crossmoment_fit': [r['start'] for r in pilot['fit_windows']],
        'crossmoment_heldout': [r['start'] for r in pilot['heldout_windows']],
        'prototype_training': prototype['starts']}
    if excluded != data['excluded_starts'] or [len(v) for v in excluded.values()] != [32,256,32,16,32]:
        raise ValueError('Prior stored TRAIN token intervals differ')
    prior = [s for starts in excluded.values() for s in starts]
    eligible = [s for s in range(0, 2533678-2048+1, 2048)
                if all(s+2048 <= old or old+2048 <= s for old in prior)]
    if len(eligible) != 839:
        raise ValueError('Eligible disjoint TRAIN blocks differ')
    starts = [eligible[i*838//255] for i in range(256)]
    if starts != data['starts'] or len(data['windows']) != 256:
        raise ValueError('Declared256-window selection differs')
    token_path = checked_file(directory, 'training_tokens.pt',
        {'bytes': data['token_windows_file_bytes'], 'sha256': TOKEN_FILE_SHA})
    if data['token_windows_file_sha256'] != TOKEN_FILE_SHA or data['token_windows_file'] != token_path.name:
        raise ValueError('Training-token file declaration differs')
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tuple(tokens.shape) != (256,2048)
            or tokens.min() < 0 or tokens.max() >= 256000
            or token_digest(tokens.flatten().numpy()) != data['selected_tokens_sha256_int64le']):
        raise ValueError('Training-token tensor identity differs')
    for row, start, window in zip(data['windows'], starts, tokens):
        if row != {'start':start, 'stored_tokens':2048, 'targets':2047,
                   'token_sha256_int64le':token_digest(window.numpy())}:
            raise ValueError('Training-window content differs')
    expected_overlap = {'grid_blocks':1237,'excluded_grid_blocks':398,'eligible_blocks':839,
        'selected_blocks':256,'internal_overlap_tokens':0,'all_previous_fitting_overlap_tokens':0,
        'previous_family_overlap_tokens':{name:0 for name in excluded},'minimum_start_gap':6144}
    if data['overlap'] != expected_overlap:
        raise ValueError('TRAIN overlap accounting differs')
    return data


def verify_export(args):
    """CPU-only independent ledger/master verification; no training imports."""
    pins = {args.protocol:PROTOCOL_SHA, args.baseline_report:BASELINE_SHA,
        args.parent_dir/'manifest.json':PARENT_SHA, args.small_dir/'manifest.json':SMALL_SHA,
        args.small_dir/'other_fp16.pt':SMALL_VALUES_SHA,
        ROOT/'scripts/evaluate_small_compensation.py':HELPER_SHA,
        ROOT/'mamba_e8w5/low_rank_residual.py':FOUNDATION_SHA,
        ROOT/'mamba_e8w5/calibration.py':CALIBRATION_SHA}
    pins.update({ROOT/name:digest for name,digest in {**FROZEN,**DATA_UPSTREAM}.items()})
    for path,digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f'Pinned input/code differs: {path}')
    baseline = json.loads(args.baseline_report.read_text())
    if baseline.get('complete') is not True or sha(ROOT/'scripts/audit_decoded.py') != baseline['loaded_tensor_audit_source_sha256']:
        raise ValueError('Baseline/audit helper differs')
    pins[ROOT/'scripts/audit_decoded.py'] = baseline['loaded_tensor_audit_source_sha256']
    parent, small, base_receipt = small_overlay.verify_overlay(args.parent_dir, args.small_dir,
        initialization_dir=args.initialization_dir)
    verify_training_data(args.data_dir)
    manifest_path = args.overlay_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'MAMBA2_LOW_RANK_COMPENSATION_V1' or manifest.get('complete') is not True
            or manifest.get('parent_manifest_sha256') != PARENT_SHA or manifest.get('small_manifest_sha256') != SMALL_SHA
            or manifest.get('small_values_sha256') != SMALL_VALUES_SHA or manifest.get('model_config') != MODEL_CONFIG):
        raise ValueError('Invalid low-rank overlay or baseline identity')
    binding = manifest['binding']
    expected = {'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'small_values_sha256':SMALL_VALUES_SHA,'source_checkpoint_sha256':SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256':TOKENIZER_SHA256,'baseline_report_sha256':BASELINE_SHA,
        'protocol_sha256':PROTOCOL_SHA,'training_data_manifest_sha256':DATA_SHA,
        'training_tokens_sha256':TOKEN_FILE_SHA,'frozen_code_sha256':FROZEN}
    for key,value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f'Low-rank binding differs: {key}')
    code = binding['code_sha256']
    if set(code) != TRAIN_CODE_PATHS:
        raise ValueError('Training code binding coverage differs')
    for name,digest in code.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Training source differs: {name}')
    hp = binding['hyperparameters']
    required_hp = {'successful_updates':1024,'maximum_overflow_retries':8,'maximum_attempts':1032,
        'epochs':4,'windows_per_epoch':256,'stored_window_tokens':2048,'targets_per_update':2047,
        'seed':20260928,'initialization_seed':20260928,'rank':4,'optimizer':'AdamW',
        'learning_rate':.0003,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,'gradient_clip_norm':1.,
        'ce_coefficient':.5,'teacher_to_student_kl_coefficient':.5,'temperature':1.,'logits_chunk_tokens':64,
        'resume':False,'checkpoint_every_successful_updates':32,
        'loss_scaler':{'init_scale':1024.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000}}
    if any(hp.get(key) != value for key,value in required_hp.items()):
        raise ValueError('Fixed optimization recipe differs')
    inventory = expected_inventory()
    expected_names = {label+'.lrf' for label in LABELS}
    if manifest.get('factors_inventory') != inventory or set(manifest.get('files',{})) != expected_names:
        raise ValueError('Expected exactly112 declared rank4 factor files')
    if {p.name for p in args.overlay_dir.iterdir()} != expected_names|{'manifest.json'}:
        raise ValueError('Unlisted low-rank overlay file')
    factors = {}
    for label, geometry in inventory.items():
        name = label+'.lrf'
        entry = manifest['files'][name]
        path = checked_file(args.overlay_dir, name, entry)
        for key,value in factor_layout(*geometry['shape'],4).items():
            if entry.get(key) != value:
                raise ValueError(f'Factor file capacity/geometry differs: {name}/{key}')
        if entry.get('format') != FORMAT or entry.get('disk_roundtrip_bitwise_equal') is not True:
            raise ValueError('Factor file format/readback receipt differs')
        B,A = read_factors(path,expected_shape=geometry['shape'],expected_rank=4)
        raw_B = B.numpy().astype('<f2',copy=False).tobytes()
        raw_A = A.numpy().astype('<f2',copy=False).tobytes()
        if (entry['B_sha256'] != hashlib.sha256(raw_B).hexdigest()
                or entry['A_sha256'] != hashlib.sha256(raw_A).hexdigest()
                or entry['payload_sha256'] != hashlib.sha256(raw_B+raw_A).hexdigest()):
            raise ValueError(f'Factor payload digest differs: {label}')
        factors[label] = (B,A)
    frozen_hashes = {entry['name']:entry['decoded_fp16_sha256']
        for entry in baseline['parent_loaded_tensor_audit']['tensors']}
    frozen_hashes.update(baseline['small_loaded_tensor_audit']['small']['loaded_and_export_fp16_sha256'])
    if len(frozen_hashes) != 507 or manifest.get('frozen_base_hash_ledger') != frozen_hashes:
        raise ValueError('Frozen baseline507 hash ledger differs')
    receipt = manifest['training_receipt']
    for path,key in ((args.training_report,'report_sha256'),(args.smoke_report,'smoke_report_sha256'),
                     (args.training_checkpoint,'checkpoint_sha256')):
        if sha(path) != receipt[key]:
            raise ValueError(f'Final training receipt differs: {key}')
    train = json.loads(args.training_report.read_text())
    smoke = json.loads(args.smoke_report.read_text())
    for report,mode in ((train,'train'),(smoke,'smoke')):
        if (report.get('format') != 'MAMBA2_LOW_RANK_COMPENSATION_TRAIN_V1' or report.get('complete') is not True
                or report.get('mode') != mode or report.get('binding') != binding
                or report.get('split') != 'train' or report.get('evaluation_data_used') is not False):
            raise ValueError(f'{mode} provenance/scope differs')
        if (report.get('frozen_parameter_audit',{}).get('before_sha256') != frozen_hashes
                or report['frozen_parameter_audit'].get('after_sha256') != frozen_hashes
                or report['frozen_parameter_audit'].get('bitwise_unchanged') is not True
                or report.get('export_parity',{}).get('bitwise_equal') is not True):
            raise ValueError(f'{mode} frozen content/export parity receipt differs')
    if smoke.get('passed') is not True or smoke.get('trial_state_discarded') is not True:
        raise ValueError('Required smoke did not pass/discard trial state')
    if (type(train.get('final_step')) is not int or train['final_step'] != 1024
            or train.get('final_export',{}).get('files') != manifest['files']
            or train.get('smoke_report_sha256') != receipt['smoke_report_sha256']
            or train['final_export'].get('checkpoint1024_master_to_fp16_bitwise_equal') is not True
            or train['final_export'].get('checkpoint_sha256') != receipt['checkpoint_sha256']
            or train.get('final_checkpoint',{}).get('sha256') != receipt['checkpoint_sha256']
            or train['final_checkpoint']['bytes'] != args.training_checkpoint.stat().st_size):
        raise ValueError('Final export/training receipt differs')
    state = torch.load(args.training_checkpoint,map_location='cpu',weights_only=True)
    rounding = verify_checkpoint_factors(state,factors,binding,frozen_hashes,receipt['smoke_report_sha256'])
    if train.get('history') != state['history'] or train.get('schedule') != state['schedule']:
        raise ValueError('Training report/checkpoint schedule differs')
    for key in ('successful_updates','attempts','overflow_retries'):
        if (type(train.get(key)) is not int or type(receipt.get(key)) is not int
                or train[key] != rounding[key] or receipt[key] != rounding[key]):
            raise ValueError('Final training accounting differs')
    for key in ('successful_target_exposures','attempted_target_exposures'):
        if type(train.get(key)) is not int or train[key] != rounding[key]:
            raise ValueError('Training target exposure count differs')
    factor_bytes = sum(entry['bytes'] for entry in manifest['files'].values())
    logical = base_receipt['logical_candidate_data_bytes'] + factor_bytes
    storage = {'base_logical_data_bytes':base_receipt['logical_candidate_data_bytes'],
        'factor_payload_bytes':FACTOR_PAYLOAD_BYTES,'factor_file_bytes':factor_bytes,
        'logical_inference_data_bytes':logical,'low_rank_manifest_bytes':manifest_path.stat().st_size,
        'physical_parent_small_low_rank_directory_bytes':sum(p.stat().st_size
            for directory in (args.parent_dir,args.small_dir,args.overlay_dir) for p in directory.iterdir()),
        'scope':'Raw resolved model/overlay files and measured manifests; tokenizer/software/reports/training/source excluded.'}
    for key in ('base_logical_data_bytes','factor_payload_bytes','factor_file_bytes','logical_inference_data_bytes'):
        if manifest['storage'].get(key) != storage[key]:
            raise ValueError(f'Manifest storage accounting differs: {key}')
    if factor_bytes != FACTOR_FILE_BYTES or manifest['storage'].get('manifest_bytes') != manifest_path.stat().st_size:
        raise ValueError('Actual factor/manifest capacity differs')
    checked_paths = {str(path.resolve()):digest for path,digest in pins.items()}
    checked_paths.update({str((ROOT/name).resolve()):digest for name,digest in code.items()})
    checked_paths.update({str((args.overlay_dir/name).resolve()):entry['sha256'] for name,entry in manifest['files'].items()})
    checked_paths.update({str((args.parent_dir/name).resolve()):entry['sha256'] for name,entry in parent['files'].items()})
    for path in (manifest_path,args.training_report,args.smoke_report,args.training_checkpoint,
                 args.data_dir/'manifest.json',args.data_dir/'training_tokens.pt',Path(__file__)):
        checked_paths[str(path.resolve())] = sha(path)
    result = {'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'low_rank_manifest_sha256':sha(manifest_path),'binding':binding,'training_receipt':receipt,
        'independent_final1024_factor_audit':rounding,'verified_file_count':112,
        'frozen_baseline507_hashes':frozen_hashes,'data_manifest_sha256':DATA_SHA,
        'storage':storage,'checked_input_sha256':checked_paths}
    return parent,small,manifest,baseline,result


@torch.no_grad()
def merge_native_projection(base, B, A):
    """Independent deployment arithmetic; intentionally no bank/trainer import."""
    layout = validate_factors(B,A)
    if (not isinstance(base,torch.Tensor) or base.dtype != torch.float16
            or list(base.shape) != layout['shape'] or layout['rank'] != 4):
        raise ValueError('Native base/factor geometry or rank differs')
    if base.device.type not in ('cpu','cuda'):
        raise ValueError('Unsupported native merge device')
    if base.device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32
            or torch.get_float32_matmul_precision() != 'highest'):
        raise ValueError('Native CUDA merge requires highest FP32 matmul precision and TF32 off')
    with torch.autocast(device_type=base.device.type,enabled=False):
        value = (base.detach().float() + B.to(base.device).float() @ A.to(base.device).float()).half()
    if not torch.isfinite(value).all():
        raise ValueError('Independent native merge overflows FP16')
    return value


def compare_results(results, plan):
    if set(results) != set(ARMS):
        raise ValueError('Exactly three native full-validation arms are required')
    for result in results.values():
        check_arm(result,plan)
    source,base,candidate = (results[name] for name in ARMS)
    paired = []
    for i,row in enumerate(plan):
        values = {name:results[name]['windows'][i] for name in ARMS}
        paired.append({**row,'nll':{name:value['nll']for name,value in values.items()},
            'ppl':{name:value['ppl']for name,value in values.items()},
            'candidate_minus_baseline_mean_nll':(values[ARMS[2]]['nll']-values[ARMS[1]]['nll'])/row['target_tokens'],
            'candidate_minus_source_mean_nll':(values[ARMS[2]]['nll']-values[ARMS[0]]['nll'])/row['target_tokens']})
    comparison = {'ppl':{name:results[name]['ppl']for name in ARMS},
        'candidate_vs_baseline_ppl_relative_change':candidate['ppl']/base['ppl']-1,
        'candidate_vs_source_ppl_relative_change':candidate['ppl']/source['ppl']-1,
        'baseline_vs_source_ppl_relative_change':base['ppl']/source['ppl']-1,
        'candidate_minus_baseline_mean_nll':(candidate['nll']-base['nll'])/TARGETS,
        'meaningful_improvement_reference':{'required_reduction':.01,'met':candidate['ppl']<=.99*base['ppl']},
        'source_plus5_percent_reference':{'maximum_ppl':1.05*source['ppl'],'met':candidate['ppl']<=1.05*source['ppl']},
        'window_counts':{'improved':sum(row['candidate_minus_baseline_mean_nll']<0 for row in paired),
            'unchanged':sum(row['candidate_minus_baseline_mean_nll']==0 for row in paired),
            'worsened':sum(row['candidate_minus_baseline_mean_nll']>0 for row in paired)}}
    return comparison,paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('overlay-dir','training-report','training-checkpoint','smoke-report','report'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    for key,default in {'source-dir':'models/source','parent-dir':'artifacts/e8w5_v1',
        'small-dir':'artifacts/small_compensation_v1','initialization-dir':'artifacts/norm_compensation_v1',
        'data-dir':'training_data/low_rank_compensation_v1','protocol':'docs/LOW_RANK_COMPENSATION_PROTOCOL.md',
        'baseline-report':'reports/small_compensation_v1_eval.json'}.items():
        parser.add_argument('--'+key,type=Path,default=ROOT/default)
    args = parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve())
    if args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():
        raise FileExistsError(args.report)
    for directory in (args.source_dir,args.parent_dir,args.small_dir,args.initialization_dir,args.data_dir,args.overlay_dir):
        if args.report == directory or directory in args.report.parents:
            raise ValueError('Evaluation report must not be inside an immutable input directory')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    report = {'format':'MAMBA2_LOW_RANK_COMPENSATION_EVALUATION_V1','complete':False,
        'mode':'cpu_integrity_only' if args.verify_only else 'three_arm_full_validation',
        'script_sha256':sha(__file__),'imported_window_audit_helper_sha256':HELPER_SHA,
        'arm_order':list(ARMS),'results':{},'same_process':True,
        'quality_forward':'native model only; independent FP32 factor merge followed by FP16 cast; no training bank forward',
        'limitations':['Validation has informed development; this is not untouched test evidence.',
            'No MK/test/training/checkpoint selection/publication is performed here.',
            'Rank4 factor files add15658496bytes plus manifest; this is not same-byte compression.',
            'Evaluation expands weights to FP16; timings are diagnostics, not compressed-runtime throughput.']}
    write_json(args.report,report)
    try:
        parent,small,manifest,prior,receipt = verify_export(args)
        report['integrity'] = receipt
        write_json(args.report,report)
        if args.verify_only:
            report.update(complete=True,elapsed_seconds=time.perf_counter()-started,
                          cuda_initialized=torch.cuda.is_initialized())
            write_json(args.report,report)
            print(json.dumps({'complete':True,'mode':report['mode'],'report_sha256':sha(args.report)}),flush=True)
            return
        report['environment'] = environment_receipt()
        tokenizer = SentencePieceTokenizer(args.source_dir)
        ids,dataset = load_wikitext_tokens(tokenizer,'validation',WIKITEXT_REVISION)
        windows = ppl_windows(ids,2048,None)
        plan = window_plan(windows)
        if dataset != prior['dataset'] or plan != prior['window_plan']:
            raise ValueError('Frozen validation dataset/window hashes differ')
        report.update(dataset=dataset,window_plan=plan,
            protocol={'target_tokens':TARGETS,'windows':130,'maximum_targets_per_window':2048,
                'execution':'prefill','logits_chunk_tokens':64,'fresh_state_per_window':True})
        write_json(args.report,report)
        torch.cuda.reset_peak_memory_stats()

        def score(name,model):
            print(f'[low_rank evaluation] Scoring {name}',flush=True)
            result = evaluate_ppl(model,windows,execution='prefill',logits_chunk=64)
            check_arm(result,plan)
            report['results'][name] = result
            write_json(args.report,report)

        source = load_source_model(args.source_dir)
        report['source_package_receipt'] = source._package_receipt
        report['source_coverage'] = audit_parameter_coverage(source)
        score('source_fp16',source)
        del source
        gc.collect()
        torch.cuda.empty_cache()
        model = load_quantized_model(args.parent_dir)
        report['parent_package_receipt'] = model._package_receipt
        report['parent507_loaded_audit'] = audit_loaded_model(model,parent,args.parent_dir)
        small_overlay.apply_overlay(model,args.small_dir,small)
        report['small_overlay_application'] = model._small_overlay_receipt
        small_values = torch.load(args.small_dir/'other_fp16.pt',map_location='cpu',weights_only=True)
        report['baseline393_export_audit'] = audit_selected_values(model,small_values)
        report['baseline507_actual_content_audit'] = audit_frozen_values(model,receipt['frozen_baseline507_hashes'])
        report['baseline_coverage'] = audit_parameter_coverage(model)
        score('best_small_e8w5',model)
        projection_names = {entry['source_key'] for entry in expected_inventory().values()}
        unchanged = {name:value for name,value in model.named_parameters() if name not in projection_names}
        if len(unchanged) != 395:
            raise ValueError('Expected393 small plus2 W5 unchanged tensors')
        expected_unchanged = {name:receipt['frozen_baseline507_hashes'][name] for name in unchanged}
        merged_hashes = {}
        for position,label in enumerate(LABELS,1):
            entry = expected_inventory()[label]
            path = checked_file(args.overlay_dir,entry['file'],manifest['files'][entry['file']])
            B,A = read_factors(path,expected_shape=entry['shape'],expected_rank=4)
            name = entry['source_key']
            owner,leaf = name.rsplit('.',1)
            base = model.get_parameter(name)
            if tensor_sha_fp16(base) != receipt['frozen_baseline507_hashes'][name]:
                raise ValueError(f'Base projection changed before independent merge: {label}')
            value = merge_native_projection(base,B,A)
            digest = tensor_sha_fp16(value)
            setattr(model.get_submodule(owner),leaf,torch.nn.Parameter(value,requires_grad=False))
            if tensor_sha_fp16(model.get_parameter(name)) != digest:
                raise ValueError(f'Installed native projection differs: {label}')
            merged_hashes[label] = {'source_key':name,'shape':list(value.shape),'merged_fp16_sha256':digest,
                'factor_file_sha256':sha(path),'base_fp16_sha256':receipt['frozen_baseline507_hashes'][name]}
            if position % 16 == 0 or position == 112:
                print(f'[low_rank evaluation] Independent native merge {position}/112',flush=True)
        del base,value,B,A
        gc.collect()
        torch.cuda.empty_cache()
        if any(model.get_parameter(name) is not value for name,value in unchanged.items()):
            raise ValueError('Candidate replaced a frozen small/vocabulary tensor object')
        report['candidate112_merged_fp16'] = merged_hashes
        report['candidate395_unchanged_content_audit'] = audit_frozen_values(model,expected_unchanged)
        report['candidate_coverage'] = audit_parameter_coverage(model)
        score('low_rank_e8w5',model)
        expected_candidate = dict(expected_unchanged)
        expected_candidate.update({entry['source_key']:entry['merged_fp16_sha256'] for entry in merged_hashes.values()})
        report['candidate507_post_evaluation_actual_content_audit'] = audit_frozen_values(model,expected_candidate)
        if any(model.get_parameter(name) is not value for name,value in unchanged.items()):
            raise ValueError('Candidate evaluation replaced a frozen small/vocabulary object')
        report['comparison'],report['paired_windows'] = compare_results(report['results'],plan)
        for path,digest in receipt['checked_input_sha256'].items():
            if sha(path) != digest:
                raise ValueError(f'Bound input changed during evaluation: {path}')
        report.update(complete=True,elapsed_seconds=time.perf_counter()-started,gpu_memory=gpu_memory_receipt())
        write_json(args.report,report)
        print(json.dumps({'complete':True,'comparison':report['comparison'],'report_sha256':sha(args.report)}),flush=True)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        if torch.cuda.is_initialized():
            report['gpu_memory'] = gpu_memory_receipt()
        write_json(args.report,report)
        raise


if __name__ == '__main__':
    main()
