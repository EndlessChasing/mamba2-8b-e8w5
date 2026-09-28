#!/usr/bin/env python3
"""Independent final448 teacher-KL export, reserved gate and conditional PPL.

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

PROTOCOL_SHA = 'c80fe02fe1dafb24b7a983bf347205c709a7af297cc325ed8761a29ecbff5c6a'
DATA_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
TOKEN_FILE_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
HELDOUT_FILE_SHA = 'a3fc0803052890d973b74f91581a6bfc1d2f93ed28db5725f23c16eb8ec20546'
DATA_PREP_SHA = '4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13'
DIAGNOSTIC_SHA = 'e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
BASELINE_SHA = '957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
HELPER_SHA = '2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be'
FOUNDATION_SHA = '050bc995c14477189b5b919d8d75c31e6c18dd4815d0de54cb72da39f6e083d6'
LABELS = tuple(f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj'))
ARMS = ('source_fp16', 'best_small_e8w5', 'teacher_kl_e8w5')
TARGETS = 264764
FACTOR_PAYLOAD_BYTES = 15654912
FACTOR_FILE_BYTES = 15658496
TRAIN_CODE_PATHS = {'scripts/train_teacher_kl_compensation.py','mamba_e8w5/teacher_kl_training.py',
    'scripts/prepare_teacher_kl_data.py','mamba_e8w5/low_rank_training.py','mamba_e8w5/low_rank_residual.py'}
FROZEN = {
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/small_overlay.py': '3512c1b232c4039c16a26ffc0fe8c2eac1be6bb9c704ccbcb1d4e8e7788a1286',
    'mamba_e8w5/norm_training.py': '25ac3ec6ba167902014018f99fbf8dbbc659e0e4903f80baf28e28a4a3e74087',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
}
FROZEN.update({
    'scripts/train_low_rank_compensation.py':'fdcb974e7abcd9379a6da900cd5801839e03973275010185f59235d67d938273',
    'mamba_e8w5/low_rank_training.py':'1948c89100750477e2bacda8013c97ff3e2ccbde0e444b961092f28c96c7f503',
    'mamba_e8w5/low_rank_residual.py':FOUNDATION_SHA,
})

DATA_UPSTREAM = {
    'calibration/v1/manifest.json': '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab',
    'training_data/small_compensation_v1/manifest.json': '88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709',
    'reports/projection_crossmoment_pilot_v1.json': '034a161df40fa4dac5be99b25545f9087da6695accc414a852fafe0e47228f87',
    'training_data/prototype_compensation_v1/manifest.json': 'a485424d1b7894eb1d5a6361ab4c25fdb4f6e5bedc56d7a15a7e4add2c454bcd',
}
DATA_UPSTREAM.update({
    'training_data/low_rank_compensation_v1/manifest.json':'e168e1d90ea7cd9c14c8210f13b5e6b569612c82dcaf832100c250cb72d24bef',
    'reports/low_rank_generalization_v1.json':'2f75a73ad675f563b035f02ee3fab3522f011785eff3459b5c0caf098347da25',
})

CALIBRATION_SHA = 'b5a7356a2ab769538940aed218653a092a7055aa26a9ec6f62d3783b9ac78ee5'


def expected_inventory():
    return {label: {'file': label + '.lrf',
        **factor_layout(*( (18560, 4096) if label.endswith('in_proj') else (4096, 8192) ), 4),
        'source_key': 'backbone.layers.' + label[5:].replace('.', '.mixer.', 1) + '.weight'}
        for label in LABELS}


def expected_schedule():
    generator = torch.Generator(device='cpu').manual_seed(20260928)
    return torch.randperm(448, generator=generator).tolist()


def verify_accounting(state):
    schedule = expected_schedule()
    actual = state.get('schedule')
    if (not isinstance(actual, list) or actual != schedule or any(type(v) is not int for v in actual)
            or type(state.get('successful_updates')) is not int or state['successful_updates'] != 448
            or type(state.get('attempts')) is not int or type(state.get('overflow_retries')) is not int):
        raise ValueError('Only the declared final448 schedule is eligible')
    successes = overflows = 0
    history = state.get('history')
    if not isinstance(history, list):
        raise ValueError('Attempt history must be a list')
    for number, row in enumerate(history, 1):
        if (not isinstance(row, dict) or successes >= 448 or type(row.get('attempt')) is not int
                or row['attempt'] != number or type(row.get('overflow')) is not bool
                or type(row.get('schedule_entry')) is not int or row['schedule_entry'] != schedule[successes]
                or type(row.get('targets')) is not int or row['targets'] != 2047
                or type(row.get('successful_updates_after')) is not int):
            raise ValueError('Attempt history differs from the declared schedule')
        overflows += int(row['overflow'])
        successes += int(not row['overflow'])
        if row['successful_updates_after'] != successes:
            raise ValueError('Incorrect successful-update history')
    if (successes != 448 or overflows > 8 or len(history) != 448 + overflows
            or state['attempts'] != len(history) or state['overflow_retries'] != overflows):
        raise ValueError('Final update/overflow accounting differs')
    return {'successful_updates': 448, 'attempts': len(history), 'overflow_retries': overflows,
        'successful_target_exposures': 917056, 'attempted_target_exposures': len(history) * 2047}


def verify_checkpoint_factors(state, factors, binding, frozen_hashes, smoke_sha):
    if (state.get('format') != 'MAMBA2_TEACHER_KL_TRAINING_CHECKPOINT_V1'
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
    """Pinned CPU-only preparation verifier; no trainer/bank forward import."""
    import prepare_teacher_kl_data as prep
    if any(value.startswith('PENDING') for value in (DATA_SHA,TOKEN_FILE_SHA,HELDOUT_FILE_SHA,DATA_PREP_SHA)):
        raise ValueError('Teacher-KL data pins are not frozen')
    if sha(ROOT/'scripts/prepare_teacher_kl_data.py') != DATA_PREP_SHA:
        raise ValueError('Pinned data verifier differs')
    data,train,heldout = prep.verify_data(directory,expected_manifest_sha256=DATA_SHA)
    if (data['protocol_sha256'] != PROTOCOL_SHA or data['schedule'] != expected_schedule()
            or data['training_tokens_file_sha256'] != TOKEN_FILE_SHA
            or data['heldout_tokens_file_sha256'] != HELDOUT_FILE_SHA
            or data['excluded_manifests_sha256'] != DATA_UPSTREAM
            or tuple(train.shape) != (448,2048) or tuple(heldout.shape) != (64,2048)):
        raise ValueError('Frozen teacher-KL train/reserved identities differ')
    return data,heldout


def verify_export(args):
    """CPU-only independent ledger/master verification; no training imports."""
    pins = {args.protocol:PROTOCOL_SHA, args.baseline_report:BASELINE_SHA,
        args.parent_dir/'manifest.json':PARENT_SHA, args.small_dir/'manifest.json':SMALL_SHA,
        args.small_dir/'other_fp16.pt':SMALL_VALUES_SHA,
        ROOT/'scripts/evaluate_small_compensation.py':HELPER_SHA,
        ROOT/'mamba_e8w5/low_rank_residual.py':FOUNDATION_SHA,
        ROOT/'mamba_e8w5/calibration.py':CALIBRATION_SHA,
        ROOT/'scripts/prepare_teacher_kl_data.py':DATA_PREP_SHA,
        ROOT/'scripts/diagnose_low_rank_generalization.py':DIAGNOSTIC_SHA}
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
    data,heldout = verify_training_data(args.data_dir)
    manifest_path = args.overlay_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'MAMBA2_TEACHER_KL_COMPENSATION_V1' or manifest.get('complete') is not True
            or manifest.get('parent_manifest_sha256') != PARENT_SHA or manifest.get('small_manifest_sha256') != SMALL_SHA
            or manifest.get('small_values_sha256') != SMALL_VALUES_SHA or manifest.get('model_config') != MODEL_CONFIG):
        raise ValueError('Invalid low-rank overlay or baseline identity')
    binding = manifest['binding']
    expected = {'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'small_values_sha256':SMALL_VALUES_SHA,'source_checkpoint_sha256':SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256':TOKENIZER_SHA256,'baseline_report_sha256':BASELINE_SHA,
        'protocol_sha256':PROTOCOL_SHA,'training_data_manifest_sha256':DATA_SHA,
        'training_tokens_sha256':TOKEN_FILE_SHA,'heldout_tokens_sha256':HELDOUT_FILE_SHA,
        'frozen_code_sha256':FROZEN}
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
    required_hp = {'successful_updates':448,'maximum_overflow_retries':8,'maximum_attempts':456,
        'epochs':1,'windows_per_epoch':448,'stored_window_tokens':2048,'targets_per_update':2047,
        'seed':20260928,'initialization_seed':20260928,'rank':4,'optimizer':'AdamW',
        'learning_rate':.0001,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,'gradient_clip_norm':1.,
        'ce_coefficient':0.,'teacher_to_student_kl_coefficient':1.,'temperature':1.,'logits_chunk_tokens':64,
        'resume':False,'checkpoint_every_successful_updates':64,
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
        if (report.get('format') != 'MAMBA2_TEACHER_KL_COMPENSATION_TRAIN_V1' or report.get('complete') is not True
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
    if (type(train.get('final_step')) is not int or train['final_step'] != 448
            or train.get('final_export',{}).get('files') != manifest['files']
            or train.get('smoke_report_sha256') != receipt['smoke_report_sha256']
            or train['final_export'].get('checkpoint448_master_to_fp16_bitwise_equal') is not True
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
        'logical_inference_data_bytes':logical,'teacher_kl_manifest_bytes':manifest_path.stat().st_size,
        'physical_parent_small_teacher_kl_directory_bytes':sum(p.stat().st_size
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
                 args.data_dir/'manifest.json',args.data_dir/'training_tokens.pt',args.data_dir/'heldout_tokens.pt',Path(__file__)):
        checked_paths[str(path.resolve())] = sha(path)
    result = {'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'teacher_kl_manifest_sha256':sha(manifest_path),'binding':binding,'training_receipt':receipt,
        'independent_final448_factor_audit':rounding,'verified_file_count':112,
        'frozen_baseline507_hashes':frozen_hashes,'data_manifest_sha256':DATA_SHA,
        'storage':storage,'checked_input_sha256':checked_paths}
    return parent,small,manifest,baseline,result,data,heldout


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


def check_reserved_rows(rows, plans):
    import math
    if len(rows) != 64 or len(plans) != 64:
        raise ValueError('Reserved evaluation requires exactly64 windows')
    for row,plan in zip(rows,plans):
        if any(row.get(key) != value for key,value in plan.items()) or row['target_tokens'] != 2047:
            raise ValueError('Reserved token identity/target count differs')
        for chunks,scalar in (('teacher_ce_chunk_sums','source_nll'),
                              ('student_ce_chunk_sums','student_nll'),
                              ('teacher_kl_chunk_sums','teacher_to_student_kl_sum')):
            values = row[chunks]
            if (len(values) != 32 or not all(math.isfinite(x) for x in values)
                    or not math.isclose(math.fsum(values),row[scalar],rel_tol=1e-12,abs_tol=1e-9)):
                raise ValueError('Reserved full-vocabulary chunk accounting differs')
        for prefix in ('source','student'):
            if (not math.isclose(row[prefix+'_mean_nll'],row[prefix+'_nll']/2047,rel_tol=1e-12)
                    or not math.isclose(row[prefix+'_ppl'],math.exp(row[prefix+'_nll']/2047),rel_tol=1e-12)):
                raise ValueError('Reserved per-window PPL/NLL differs')
        if not math.isclose(row['teacher_to_student_kl_mean'],row['teacher_to_student_kl_sum']/2047,
                            rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('Reserved per-window teacher KL differs')


def reserved_comparison(base_rows,candidate_rows,plans):
    import math
    import struct
    check_reserved_rows(base_rows,plans)
    check_reserved_rows(candidate_rows,plans)
    paired = []
    for base,candidate,plan in zip(base_rows,candidate_rows,plans):
        a = base['teacher_ce_chunk_sums']+[base['source_nll']]
        b = candidate['teacher_ce_chunk_sums']+[candidate['source_nll']]
        if any(struct.pack('<d',x) != struct.pack('<d',y) for x,y in zip(a,b)):
            raise ValueError('Source CE must exactly repeat across reserved passes')
        paired.append({**plan,'source_nll':base['source_nll'],
            'base_nll':base['student_nll'],'candidate_nll':candidate['student_nll'],
            'base_teacher_kl_sum':base['teacher_to_student_kl_sum'],
            'candidate_teacher_kl_sum':candidate['teacher_to_student_kl_sum'],
            'candidate_minus_baseline_mean_nll':(candidate['student_nll']-base['student_nll'])/2047,
            'candidate_minus_baseline_mean_teacher_kl':(candidate['teacher_to_student_kl_sum']-base['teacher_to_student_kl_sum'])/2047})
    targets = 64*2047
    nll = {'source_fp16':math.fsum(row['source_nll'] for row in base_rows),
        'best_small_e8w5':math.fsum(row['student_nll'] for row in base_rows),
        'teacher_kl_e8w5':math.fsum(row['student_nll'] for row in candidate_rows)}
    kl = {'source_fp16':0.,
        'best_small_e8w5':math.fsum(row['teacher_to_student_kl_sum'] for row in base_rows)/targets,
        'teacher_kl_e8w5':math.fsum(row['teacher_to_student_kl_sum'] for row in candidate_rows)/targets}
    ppl = {name:math.exp(value/targets) for name,value in nll.items()}
    ppl_met = ppl[ARMS[2]] <= .99*ppl[ARMS[1]]
    kl_met = kl[ARMS[2]] < kl[ARMS[1]]
    gate = {'ppl_met':ppl_met,'kl_met':kl_met,'passed':ppl_met and kl_met,
        'required_ppl_reduction':.01,'maximum_candidate_ppl':.99*ppl[ARMS[1]],
        'candidate_ppl':ppl[ARMS[2]],'baseline_ppl':ppl[ARMS[1]],
        'candidate_mean_teacher_kl':kl[ARMS[2]],'baseline_mean_teacher_kl':kl[ARMS[1]],
        'kl_rule':'candidate strictly lower than baseline; equality fails'}
    summary = {'windows':64,'target_tokens':targets,'nll':nll,
        'mean_nll':{name:value/targets for name,value in nll.items()},'ppl':ppl,
        'teacher_to_model_kl_mean':kl,'source_ce_exact_repeat_all64_windows':True,
        'candidate_vs_baseline_ppl_relative_change':ppl[ARMS[2]]/ppl[ARMS[1]]-1,
        'candidate_minus_baseline_mean_teacher_kl':kl[ARMS[2]]-kl[ARMS[1]],
        'window_counts':{'nll_improved':sum(row['candidate_minus_baseline_mean_nll']<0 for row in paired),
            'nll_worsened':sum(row['candidate_minus_baseline_mean_nll']>0 for row in paired),
            'kl_improved':sum(row['candidate_minus_baseline_mean_teacher_kl']<0 for row in paired),
            'kl_worsened':sum(row['candidate_minus_baseline_mean_teacher_kl']>0 for row in paired)}}
    return summary,paired,gate


def conditional_full_validation(gate, action):
    """The only entry to validation loading/forward; failed reserved gate skips it."""
    if (any(type(gate.get(key)) is not bool for key in ('ppl_met','kl_met','passed'))
            or gate['passed'] != (gate['ppl_met'] and gate['kl_met'])):
        raise ValueError('Invalid reserved gate decision')
    if not gate['passed']:
        return {'performed':False,'reason':'Reserved PPL/KL advancement gate failed; no validation data loaded.'}
    action()
    return {'performed':True,'reason':'Both predeclared reserved-set conditions passed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('overlay-dir','training-report','training-checkpoint','smoke-report','report'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    for key,default in {'source-dir':'models/source','parent-dir':'artifacts/e8w5_v1',
        'small-dir':'artifacts/small_compensation_v1','initialization-dir':'artifacts/norm_compensation_v1',
        'data-dir':'training_data/teacher_kl_compensation_v1','protocol':'docs/TEACHER_KL_COMPENSATION_PROTOCOL.md',
        'baseline-report':'reports/small_compensation_v1_eval.json'}.items():
        parser.add_argument('--'+key,type=Path,default=ROOT/default)
    args = parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.resolve())
    if args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():
        raise FileExistsError(args.report)
    for directory in (args.source_dir,args.parent_dir,args.small_dir,args.initialization_dir,args.data_dir,args.overlay_dir):
        if args.report == directory or directory in args.report.parents:
            raise ValueError('Report must not be inside an immutable input directory')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    report = {'format':'MAMBA2_TEACHER_KL_COMPENSATION_EVALUATION_V1','complete':False,
        'mode':'cpu_integrity_only' if args.verify_only else 'reserved_gate_then_conditional_full_validation',
        'script_sha256':sha(__file__),'imported_window_audit_helper_sha256':HELPER_SHA,
        'reserved_loss_helper_sha256':DIAGNOSTIC_SHA,'arm_order':list(ARMS),'same_process':True,
        'quality_forward':'independent native FP32 factor merge then FP16 cast; no training bank forward',
        'results':{},'reserved_results':{},
        'limitations':['Reserved TRAIN windows are observed once for the predeclared gate; not fitted.',
            'Full validation has informed development; it is not untouched test evidence.',
            'A failed reserved gate is a completed negative experiment, not an execution failure.',
            'No MK/test/training/checkpoint selection/publication is performed here.',
            'Factor files add15658496bytes plus manifest; no compressed-residency claim.']}
    write_json(args.report,report)
    try:
        parent,small,manifest,prior,receipt,data,heldout = verify_export(args)
        report['integrity'] = receipt
        write_json(args.report,report)
        if args.verify_only:
            if torch.cuda.is_initialized(): raise RuntimeError('CPU-only verification initialized CUDA')
            report.update(complete=True,elapsed_seconds=time.perf_counter()-started,cuda_initialized=False)
            write_json(args.report,report)
            return
        from diagnose_low_rank_generalization import score_window
        report['environment'] = environment_receipt()
        report['reserved_dataset'] = data['dataset']
        report['reserved_plan'] = plans = [{'index':i,**row,'target_tokens':row['targets']}
            for i,row in enumerate(data['heldout_windows'])]
        report['reserved_protocol'] = {'windows':64,'target_tokens':131008,'fresh_state_per_window':True,
            'stored_window_tokens':2048,'targets_per_window':2047,'logits_chunk_tokens':64,
            'vocabulary':256000,'temperature':1.,'teacher_kl_direction':'source || student'}
        torch.cuda.reset_peak_memory_stats()
        source = load_source_model(args.source_dir)
        source.eval().requires_grad_(False)
        report['source_package_receipt'] = source._package_receipt
        report['source_coverage'] = audit_parameter_coverage(source)
        model = load_quantized_model(args.parent_dir)
        report['parent_package_receipt'] = model._package_receipt
        report['parent507_loaded_audit'] = audit_loaded_model(model,parent,args.parent_dir)
        small_overlay.apply_overlay(model,args.small_dir,small)
        model.eval().requires_grad_(False)
        small_values = torch.load(args.small_dir/'other_fp16.pt',map_location='cpu',weights_only=True)
        report['baseline393_export_audit'] = audit_selected_values(model,small_values)
        report['baseline507_actual_content_audit'] = audit_frozen_values(model,receipt['frozen_baseline507_hashes'])
        report['baseline_coverage'] = audit_parameter_coverage(model)
        if source.lm_head.weight.shape[0] != 256000 or model.lm_head.weight.shape[0] != 256000:
            raise ValueError('Reserved KL requires the complete256000-token vocabulary')

        def score_reserved(name,current):
            rows = report['reserved_results'][name] = []
            for index,(window,plan) in enumerate(zip(heldout,plans)):
                rows.append({**plan,**score_window(source,current,window,chunk_tokens=64)})
                if (index+1)%8 == 0:
                    print(f'[teacher KL reserved] {name} {index+1}/64',flush=True)
                    write_json(args.report,report)
            check_reserved_rows(rows,plans)

        score_reserved('best_small_e8w5',model)
        projections = {entry['source_key'] for entry in expected_inventory().values()}
        unchanged = {name:value for name,value in model.named_parameters() if name not in projections}
        if len(unchanged) != 395: raise ValueError('Expected395 inherited tensors')
        inherited = {name:receipt['frozen_baseline507_hashes'][name] for name in unchanged}
        merged = {}
        for label,entry in expected_inventory().items():
            path = checked_file(args.overlay_dir,entry['file'],manifest['files'][entry['file']])
            B,A = read_factors(path,expected_shape=entry['shape'],expected_rank=4)
            name = entry['source_key']
            base = model.get_parameter(name)
            if tensor_sha_fp16(base) != receipt['frozen_baseline507_hashes'][name]:
                raise ValueError(f'Frozen base projection changed: {label}')
            value = merge_native_projection(base,B,A)
            digest = tensor_sha_fp16(value)
            owner,leaf = name.rsplit('.',1)
            setattr(model.get_submodule(owner),leaf,torch.nn.Parameter(value,requires_grad=False))
            merged[label] = {'source_key':name,'shape':list(value.shape),'merged_fp16_sha256':digest,
                'factor_file_sha256':sha(path),'base_fp16_sha256':receipt['frozen_baseline507_hashes'][name]}
        del base,value,B,A
        gc.collect();torch.cuda.empty_cache()
        expected_candidate = {**inherited,**{row['source_key']:row['merged_fp16_sha256'] for row in merged.values()}}
        if any(model.get_parameter(name) is not value for name,value in unchanged.items()):
            raise ValueError('Merged model replaced an inherited tensor')
        report['candidate112_merged_fp16'] = merged
        report['candidate395_unchanged_content_audit'] = audit_frozen_values(model,inherited)
        report['candidate507_loaded_content_audit'] = audit_frozen_values(model,expected_candidate)
        report['candidate_coverage'] = audit_parameter_coverage(model)
        score_reserved('teacher_kl_e8w5',model)
        report['candidate507_post_reserved_content_audit'] = audit_frozen_values(model,expected_candidate)
        summary,paired,gate = reserved_comparison(report['reserved_results']['best_small_e8w5'],
            report['reserved_results']['teacher_kl_e8w5'],plans)
        report.update(reserved_summary=summary,reserved_paired_windows=paired,reserved_gate=gate)
        write_json(args.report,report)

        def full_action():
            nonlocal source
            # This is intentionally the first validation loader invocation.
            tokenizer = SentencePieceTokenizer(args.source_dir)
            ids,dataset = load_wikitext_tokens(tokenizer,'validation',WIKITEXT_REVISION)
            windows = ppl_windows(ids,2048,None)
            plan = window_plan(windows)
            if dataset != prior['dataset'] or plan != prior['window_plan']:
                raise ValueError('Frozen full-validation identities differ')
            report.update(dataset=dataset,window_plan=plan,
                protocol={'target_tokens':TARGETS,'windows':130,'maximum_targets_per_window':2048,
                    'execution':'prefill','logits_chunk_tokens':64,'fresh_state_per_window':True})
            def score(name,current):
                result = evaluate_ppl(current,windows,execution='prefill',logits_chunk=64)
                check_arm(result,plan)
                report['results'][name] = result
                write_json(args.report,report)
            score('source_fp16',source)
            source = None
            gc.collect();torch.cuda.empty_cache()
            fresh_base = load_quantized_model(args.parent_dir)
            small_overlay.apply_overlay(fresh_base,args.small_dir,small)
            report['full_validation_reloaded_baseline507'] = audit_frozen_values(fresh_base,receipt['frozen_baseline507_hashes'])
            score('best_small_e8w5',fresh_base)
            del fresh_base
            gc.collect();torch.cuda.empty_cache()
            score('teacher_kl_e8w5',model)
            report['comparison'],report['paired_windows'] = compare_results(report['results'],plan)

        report['full_validation'] = conditional_full_validation(gate,full_action)
        report['candidate507_final_content_audit'] = audit_frozen_values(model,expected_candidate)
        if any(model.get_parameter(name) is not value for name,value in unchanged.items()):
            raise ValueError('Evaluation replaced an inherited tensor')
        for path,digest in receipt['checked_input_sha256'].items():
            if sha(path) != digest: raise ValueError(f'Bound input changed during evaluation: {path}')
        report.update(complete=True,elapsed_seconds=time.perf_counter()-started,gpu_memory=gpu_memory_receipt())
        write_json(args.report,report)
        print(json.dumps({'complete':True,'reserved_gate':gate,'full_validation':report['full_validation'],
            'comparison':report.get('comparison'),'report_sha256':sha(args.report)}),flush=True)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        if torch.cuda.is_initialized(): report['gpu_memory'] = gpu_memory_receipt()
        write_json(args.report,report)
        raise


if __name__ == '__main__':
    main()
