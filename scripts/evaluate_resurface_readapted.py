#!/usr/bin/env python3
"""Independent, manually staged evaluation of one final1536 soft adapter.

This runner never imports the training driver/bank for forward execution.
Native FP16 adapter installation is identical for MK and PPL. Confirm opens
only its own raw/token split; training manifests/checkpoints are provenance.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
FORMAT = 'MAMBA2_RESURFACE_READAPTED_EVALUATION_V1'
PROTOCOL_SHA = '469063147d892282a2bdad8558afca8466d6f365f1efe14ecbba7417e9857dd6'
CURRENT_PPL_SHA = '3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4'
BASELINE_MK_SHA = 'a88f8fe75e82b992b8fc9cebeb6f681847e87aa36ea8bb1771edf979c47d551d'
CURRENT_MANIFEST_SHA = '3ede80b2883d8b8987cb3e2198fd8badc2dbbb5af03c8fe6cac7f5f9f5b62047'
BASE, ACTIVE, RESTORED = 'current_readapted', 'active_resurface', 'restored_current'
CODE_PINS = {
    'mamba_e8w5/axis_small_base.py': '4e7026475fc97b005105b27043a15bae83c576934612b1e346576438eef4da70',
    'scripts/evaluate_axis_small.py': 'f2c2309f1dee45d32b8c4fcb9114a8d0f594cb56277b3a194a039070c361ac4c',
    'scripts/evaluate_readapted_mk.py': '3ea23e3587611346801c3a479a92a6606225cd2e6848870f3a01abf1744b3a6c',
    'scripts/evaluate_input_axis_residual.py': 'c8dece7ec209b74ef65244086c7c245f08a59ea344124559f964e44c8a2540f0',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/resurface_native.py': '2dd08c7ee8958c832f0ae9da7cf5f261dceeff3e528e493c905c7c52df7166d7',
    'mamba_e8w5/resurface_data.py': 'e813a382c25e99529eaddedb78aecd122b3c875903831f485c1cdf054ef7760c',
    'scripts/train_resurface_readapted.py': '9b8ea5911c47a4f333db994154cd6913da81beaef05bb4de2751455f113fef41',
}
CODE_PINS.update({
    'scripts/smoke_resurface_native.py': '5dc34965f0993d470c5699b2d2a1a5f85df77960fd00e1674658e9aa7b03441f',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
    'mamba_e8w5/resurface_loss.py': 'ec045368583b6e49f484341b2d718658bb59e8d11050710377742752c1c67294',
    'scripts/diagnose_vocab_w4.py': 'c1dd8343a204923d6799f5b4eed31a6f0ffee1bc0774b2deaa15f34e0d54091d',
})
TRAIN_MANIFEST_SHA = 'dfe076ef18d5b0610e016d41d00a38948f3878b7c67d9cd8f6f46237e09d7e11'
TRAINING_CODE_NAMES = ('mamba_e8w5/axis_small_base.py', 'scripts/evaluate_axis_small.py',
    'scripts/train_norm_compensation.py', 'mamba_e8w5/resurface_data.py',
    'mamba_e8w5/resurface_native.py', 'mamba_e8w5/resurface_loss.py', 'scripts/train_resurface_readapted.py')
STEPS = 1536
PARAMETERS = 1154104
HYPERPARAMETERS={
 'steps':1536,'maximum_attempts':1544,'maximum_overflow_retries':8,'seed':2026092803,
 'gate_mode':'soft','layers':56,'tensors':224,'parameters':PARAMETERS,'master_dtype':'float32','forward_dtype':'float16',
 'initialization':{'V_read':0.,'g_read':1.,'router_w':0.,'router_b':-4.},
 'optimizer':'AdamW','mix_lr':1e-4,'router_lr':3e-4,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,
 'gradient_clip_norm':1.,'lr_multiplier':'0.1+0.9*(1+cos(pi*j/1535))/2; j=successful_updates_before',
 'mk_ce_coefficient':1.,'prose_ce_coefficient':.5,'prose_teacher_kl_coefficient':.5,'temperature':1.,
 'prose_closure_coefficient':3.,'closure_budget':.006,'closure_budget_coefficient':10.,'opening_coefficient':0.,
 'prose_stored_tokens':512,'prose_targets':511,'full_vocabulary':256000,'logits_chunk_tokens':64,
 'loss_scaler':{'init_scale':1024.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000},
 'checkpoint_interval':384,'checkpoint_each_block':True,'selection':'final1536 only','resume':False,
 'teacher':'separately loaded identical compressed axis plus final448 small-tensor model; no adapter',
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            result.update(data)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def bind(path, digest, checked, size=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Symlink input forbidden: {path}')
    path = path.resolve()
    if (not isinstance(digest, str) or len(digest) != 64 or path.is_symlink() or not path.is_file()
            or sha(path) != digest or (size is not None and (type(size) is not int or path.stat().st_size != size))):
        raise ValueError(f'Bound regular file differs: {path}')
    key = str(path)
    if key in checked and checked[key] != digest:
        raise ValueError('Contradictory bound input identities')
    checked[key] = digest
    return path


def read_bound(path, digest, checked, size=None):
    return json.loads(bind(path, digest, checked, size).read_text())


def binomial_tail(k, n, p, upper):
    """Stable finite-n binomial tail for exact Clopper-Pearson inversion."""
    if p == 0.:
        return float(k == 0) if upper else 1.
    if p == 1.:
        return 1. if upper else float(k == n)
    indexes = range(k, n + 1) if upper else range(k + 1)
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
             + j * math.log(p) + (n - j) * math.log1p(-p) for j in indexes]
    maximum = max(terms)
    return math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in terms)


def clopper_pearson(k, n, lower, alpha=.025):
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n < 1 or alpha != .025:
        raise ValueError('Expected bounded integer counts and fixed one-sided alpha=.025')
    if lower and k == 0:
        return 0.
    if not lower and k == n:
        return 1.
    left, right = 0., 1.
    for _ in range(80):
        mid = (left + right) / 2
        tail = binomial_tail(k, n, mid, upper=lower)
        if (tail < alpha) == lower:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def recall_gate(comparison):
    normal, removed = comparison['normal'], comparison['target_removed']
    gained, lost = (normal['paired'][key] for key in ('gained', 'lost'))
    n = normal['count']
    lower = clopper_pearson(gained, n, True)
    upper = clopper_pearson(lost, n, False)
    bound = lower - upper
    positive = gained > lost
    significance = normal['mcnemar_exact_two_sided_p'] < .05
    control = removed['current_correct'] <= removed['source_correct']
    return {'normal_gain_positive': positive, 'mcnemar_p_below_0_05': significance,
            'gained_lower_one_sided97_5': lower, 'lost_upper_one_sided97_5': upper,
            'paired_accuracy_improvement_conservative95_lower': bound,
            'paired_lower_bound_positive': bound > 0., 'control_no_increase': control,
            'passed': positive and significance and bound > 0. and control,
            'interval_method': 'Clopper-Pearson gained/n lower(alpha=.025) minus lost/n upper(1-alpha=.975); Bonferroni coverage at least95%'}


def stage_gate(stage, mk_comparison=None, current_ppl=None, active_ppl=None):
    if stage == 'confirm':
        return recall_gate(mk_comparison)
    if not all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in (current_ppl, active_ppl)):
        raise ValueError('Paired finite PPL required')
    result = {'current_ppl': current_ppl, 'active_ppl': active_ppl,
              'ppl_no_regression': active_ppl <= current_ppl, 'ppl_epsilon': 0.}
    if stage == 'dev':
        normal, control = mk_comparison['normal'], mk_comparison['target_removed']
        result.update(normal_gain_positive=normal['current_correct'] > normal['source_correct'],
                      control_no_increase=control['current_correct'] <= control['source_correct'])
        result['passed'] = result['ppl_no_regression'] and result['normal_gain_positive'] and result['control_no_increase']
    elif stage == 'full':
        result['passed'] = result['ppl_no_regression']
    else:
        raise ValueError('Unknown stage')
    return result


def expected_schedule():
    return torch.randperm(STEPS, generator=torch.Generator(device='cpu').manual_seed(2026092803)).tolist()


def expected_inventory():
    return {f'layer{layer}.{name}': shape for layer in range(56) for name, shape in
            (('V_read', [128, 128]), ('g_read', [128]), ('router_w', [4096]), ('router_b', []))}


def require_fields(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label}: object required')
    for key, wanted in expected.items():
        actual = value.get(key)
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError(f'{label}: field differs: {key}')


def independent_plan(train_manifest):
    # Prepared metadata proves every answer has exactly7 tokens. Reconstruct
    # IDs/order and prose segmentation without reading any TRAIN text/tokens.
    require_fields(train_manifest, {'answer_target_tokens': 10752,
        'answer_token_count_histogram': {'7': 1536}}, 'TRAIN target metadata')
    prose = torch.randperm(448, generator=torch.Generator(device='cpu').manual_seed(20260928)).tolist()
    result = []
    for step, index in enumerate(expected_schedule()):
        cell, sample = divmod(index, 256)
        size, template = (16, 64)[cell // 3], cell % 3
        result.append({'schedule_entry': index, 'mk_case_id': f'resurface-train-n{size}-t{template}-s{sample}',
            'mk_targets': 7, 'prose_window_index': prose[step % 448],
            'prose_start': 512*((step//448) % 4), 'prose_targets': 511})
    return result


def verify_history(history, plan):
    if not isinstance(history, list) or not STEPS <= len(history) <= STEPS + 8:
        raise ValueError('Final history must contain1536 successes and at most8 retries')
    success = skipped = 0
    for number, row in enumerate(history, 1):
        if success >= STEPS or type(row.get('overflow')) is not bool:
            raise ValueError('History exceeds fixed final step or malformed overflow flag')
        require_fields(row, {'attempt': number, **plan[success]}, 'Fixed-pair attempt')
        multiplier = .1 + .9*(1 + math.cos(math.pi*success/1535))/2
        require_fields(row, {'mix_lr': 1e-4*multiplier, 'router_lr': 3e-4*multiplier,
                            'lr_multiplier': multiplier}, 'Fixed learning rate')
        if row['overflow']:
            skipped += 1
            if row.get('overflow_optimizer_and_masters_unchanged') is not True:
                raise ValueError('Overflow mutated optimizer/masters')
        else:
            success += 1
            if row.get('gradients', {}).get('finite') is not True:
                raise ValueError('Successful update lacked finite gradients')
        require_fields(row, {'successful_updates_after': success}, 'Cumulative success')
    if success != STEPS or skipped > 8:
        raise ValueError('Final1536 successful updates required')
    return {'successful_updates': success, 'attempts': len(history), 'overflow_retries': skipped,
            'successful_mk_answer_targets': success*7, 'attempted_mk_answer_targets': len(history)*7,
            'successful_prose_targets': success*511, 'attempted_prose_targets': len(history)*511}


def expected_training_binding(args, train_manifest, previous_ppl, tokenizer_sha256):
    training_code = {name: CODE_PINS[name] for name in TRAINING_CODE_NAMES}
    expected_checked = {str((ROOT/name).resolve()): digest for name, digest in training_code.items()}
    expected_checked.update({str(ROOT/'docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md'): PROTOCOL_SHA,
        str(ROOT/'reports/axis_small_readaptation_v1_eval.json'): CURRENT_PPL_SHA,
        str(ROOT/'artifacts/axis_small_readaptation_v1/manifest.json'): CURRENT_MANIFEST_SHA,
        str(args.dataset_root/'train/manifest.json'): TRAIN_MANIFEST_SHA})
    expected_checked.update({str(args.dataset_root/'train'/name): row['sha256'] for name, row in train_manifest['files'].items()})
    for name, digest in previous_ppl['integrity']['checked_input_sha256'].items():
        if name in expected_checked and expected_checked[name] != digest:
            raise ValueError('Contradictory original-current training proof')
        expected_checked[name] = digest
    binding = {
        'protocol_sha256': PROTOCOL_SHA, 'base_manifest_sha256': CURRENT_MANIFEST_SHA,
        'base_evaluation_sha256': CURRENT_PPL_SHA, 'base_integrity': previous_ppl['integrity'],
        'base_shared_binding': previous_ppl['current_package_receipt']['binding'],
        'training_data_manifest_sha256': TRAIN_MANIFEST_SHA, 'training_data_files': train_manifest['files'],
        'prose_data_manifest_sha256': 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d',
        'prose_training_tokens_file_sha256': 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233',
        'tokenizer_sha256': tokenizer_sha256, 'hyperparameters': HYPERPARAMETERS,
        'training_code_sha256': training_code, 'checked_input_sha256': expected_checked}
    return binding


def verify_training_export(args, shared, current_expected, train_manifest, checked, previous_ppl):
    native = importlib.import_module('mamba_e8w5.resurface_native')
    train = read_bound(args.training_report, args.training_report_sha, checked)
    require_fields(train, {'format': 'MAMBA2_RESURFACE_READAPTED_TRAINING_REPORT_V1',
        'complete': True, 'mode': 'train', 'final_step': STEPS, 'protocol_sha256': PROTOCOL_SHA,
        'script_sha256': CODE_PINS['scripts/train_resurface_readapted.py'], 'gate_mode': 'soft',
        'mk_development_loaded': False, 'confirmation_loaded': False, 'validation_scored': False,
        'prose_heldout_verified_for_provenance_only': True, 'selection': 'final1536 only',
        'resume': False, 'training_data_manifest': train_manifest}, 'Final training report')
    inventory = {name: {'shape': shape, 'numel': math.prod(shape)} for name, shape in expected_inventory().items()}
    plan, schedule = independent_plan(train_manifest), expected_schedule()
    require_fields(train, {'schedule': schedule, 'pair_plan': plan, 'adapter_inventory': inventory}, 'Training plan')
    counts = verify_history(train['history'], plan)
    require_fields(train, counts, 'Final training accounting')
    binding = expected_training_binding(args, train_manifest, previous_ppl, shared.runtime.TOKENIZER_SHA256)
    if train.get('binding') != binding:
        raise ValueError('Training recipe/base/data/code binding differs')
    # Binding map is independently reconstructed metadata, deliberately NOT
    # blindly opened: CONFIRM must not open TRAIN/DEV text or token payloads.
    checkpoint = train['final_checkpoint']
    cp_path = bind(checkpoint['file'], checkpoint['sha256'], checked, checkpoint['bytes'])
    state = torch.load(cp_path, map_location='cpu', weights_only=True)
    smoke = read_bound(args.smoke_report, train['smoke_report_sha256'], checked)
    smoke_sha = CODE_PINS['scripts/smoke_resurface_native.py']
    require_fields(smoke, {'format': 'MAMBA2_RESURFACE_NATIVE_SMOKE_V1', 'mode': 'gpu_smoke',
        'complete': True, 'passed': True, 'script_sha256': smoke_sha, 'protocol_sha256': PROTOCOL_SHA,
        'binding': binding, 'trial_state_discarded': True, 'formal_training_started': False}, 'Discarded same-binding GPU smoke')
    require_fields(train, {'smoke_source_sha256': smoke_sha}, 'Smoke source')
    require_fields(state, {'format': 'MAMBA2_RESURFACE_READAPTED_TRAINING_CHECKPOINT_V1',
        'binding': binding, 'schedule': schedule, 'pair_plan': plan, 'history': train['history'],
        'smoke_report_sha256': train['smoke_report_sha256'], 'smoke_source_sha256': smoke_sha,
        'base507_sha256': current_expected, 'teacher507_sha256': current_expected,
        'reason': 'final', 'resumable': False, **counts}, 'Actual final1536 checkpoint')
    directory = args.adapter.parent
    if args.adapter.name != 'adapter_fp16.pt' or set(path.name for path in directory.iterdir()) != {'manifest.json', 'adapter_fp16.pt'}:
        raise ValueError('Overlay must contain exactly manifest.json and adapter_fp16.pt')
    manifest_path = directory/'manifest.json'
    mp = train['manifest']
    if Path(mp['file']).resolve() != manifest_path.resolve():
        raise ValueError('Training receipt points to another overlay')
    manifest = read_bound(manifest_path, mp['sha256'], checked, mp['bytes'])
    require_fields(manifest, {'format': 'MAMBA2_RESURFACE_READAPTED_OVERLAY_V1', 'complete': True,
        'gate_mode': 'soft', 'variant': native.VARIANT, 'binding': binding,
        'tensors': inventory, 'base507_sha256': current_expected,
        'training_receipt': {**counts, 'checkpoint_sha256': checkpoint['sha256'], 'checkpoint_bytes': checkpoint['bytes'],
            'smoke_report_sha256': train['smoke_report_sha256'], 'smoke_source_sha256': smoke_sha}}, 'Overlay manifest')
    if set(manifest['files']) != {'adapter_fp16.pt'}:
        raise ValueError('Incorrect adapter file coverage')
    entry = manifest['files']['adapter_fp16.pt']
    bind(args.adapter, entry['sha256'], checked, entry['bytes'])
    payload = native.read_fp16(args.adapter, expected_binding=binding)
    if (payload['gate_mode'] != 'soft' or payload['geometry'] != [{'width':4096, 'heads':128, 'head_dim':64}]*56
            or set(payload['tensors']) != set(inventory) or set(state.get('masters', {})) != set(inventory)):
        raise ValueError('Actual adapter geometry/master inventory differs')
    hashes = {}
    for name, row in inventory.items():
        master, value = state['masters'][name], payload['tensors'][name]
        if (not isinstance(master, torch.Tensor) or master.dtype != torch.float32 or list(master.shape) != row['shape']
                or not torch.isfinite(master).all() or list(value.shape) != row['shape']):
            raise ValueError(f'Invalid final master/export tensor:{name}')
        hashes[name] = native.tensor_hash(value)
        if native.tensor_hash(master.half()) != hashes[name]:
            raise ValueError(f'Final checkpoint master.half differs from actual adapter:{name}')
    del state
    if hashes != manifest['fp16_tensor_sha256']:
        raise ValueError('Actual224 export hashes differ from manifest')
    require_fields(train['final_export'], {'file': str(args.adapter), 'bytes': entry['bytes'], 'sha256': entry['sha256'],
        'parameters': PARAMETERS, 'payload_bytes': 2*PARAMETERS, 'tensor_sha256': hashes, 'gate_mode': 'soft',
        'roundtrip_bitwise_equal': True, 'final1536_master_to_fp16_bitwise_equal': True}, 'Final export')
    parity = train['export_parity']
    require_fields(parity, {'input_tokens':128, 'logit_positions':8, 'hidden_bitwise_equal':True,
        'logits_bitwise_equal':True, 'gate_mode':'soft'}, 'Export parity')
    if parity['functional'] != parity['native_reloaded'] or set(parity['functional']) != {'hidden', 'logits'}:
        raise ValueError('Training/native export forward mismatch')
    for name in ('initial_base507_audit', 'initial_teacher507_audit', 'final_base507_audit', 'final_teacher507_audit'):
        require_fields(train[name], {'verified_tensor_count':507, 'parameter_count':8236999680,
            'unchanged_content_sha256':current_expected}, name)
    storage = {'base_logical_data_bytes':3138928792, 'adapter_parameter_count':PARAMETERS,
        'fp16_adapter_payload_bytes':2*PARAMETERS, 'adapter_file_bytes':entry['bytes'],
        'base_plus_adapter_data_bytes':3138928792+entry['bytes'], 'manifest_bytes':mp['bytes'],
        'complete_distribution_built':False}
    if manifest['storage'] != storage or train['storage'] != storage:
        raise ValueError('Actual storage ledger differs')
    return payload, {'candidate_identity': {'protocol_sha256': PROTOCOL_SHA, 'gate_mode':'soft',
        'adapter_sha256':entry['sha256'], 'overlay_manifest_sha256':mp['sha256'],
        'training_report_sha256':args.training_report_sha, 'checkpoint_sha256':checkpoint['sha256'],
        'training_manifest_sha256':TRAIN_MANIFEST_SHA, 'base_ppl_report_sha256':CURRENT_PPL_SHA},
        'final1536_accounting':counts, 'final1536_master_to_fp16_bitwise_equal':True,
        'adapter_tensor_count':224, 'adapter_parameter_count':PARAMETERS, 'adapter_tensor_sha256':hashes,
        'storage':storage, 'smoke_report_sha256':train['smoke_report_sha256'], 'smoke_source_sha256':smoke_sha,
        'training_raw_or_tokens_opened':False, 'training_binding_checked_as_metadata':True}

def prepare_current(args, shared, checked):
    """Minimal inference inputs; never open any TRAIN/DEV raw/token dataset."""
    previous = read_bound(ROOT / 'reports/axis_small_readaptation_v1_eval.json', CURRENT_PPL_SHA, checked)
    if previous.get('complete') is not True or not previous['full_validation']['gates']['source_plus5_percent_reference']['met']:
        raise ValueError('Completed accepted current PPL receipt required')
    package = previous['current_package_receipt']
    # Verify transitive repository decoder/scorer code from immutable receipts,
    # without invoking their expansive data/Hessian preparation functions.
    for ledger in (package['binding']['checked_input_sha256'], previous['integrity']['checked_input_sha256']):
        for path, digest in ledger.items():
            if Path(path).suffix == '.py':
                bind(path, digest, checked)
    old_expected = previous['direct_current_load_audit']['unchanged_content_sha256']
    expected = previous['integrity']['resolved507_candidate_fp16_sha256']
    if len(old_expected) != 507 or len(expected) != 507 or len(package['resolved_files']) != 117:
        raise ValueError('Incomplete current model receipt')
    for row in package['resolved_files'].values():
        bind(row['path'], row['sha256'], checked, row['bytes'])
    old_small = torch.load(package['resolved_files']['other_fp16.pt']['path'], map_location='cpu', weights_only=True)
    if shared.validate_small(old_small)['fp16_tensor_sha256'] != {key: old_expected[key] for key in shared.small_inventory()}:
        raise ValueError('Actual old small payload differs from direct loader identity')
    manifest_path = ROOT / 'artifacts/axis_small_readaptation_v1/manifest.json'
    manifest = read_bound(manifest_path, CURRENT_MANIFEST_SHA, checked)
    entry = manifest['files']['other_fp16.pt']
    path = bind(manifest_path.parent / 'other_fp16.pt', entry['sha256'], checked, entry['bytes'])
    values = torch.load(path, map_location='cpu', weights_only=True)
    digest = shared.validate_small(values)['fp16_tensor_sha256']
    if expected != {**old_expected, **digest} or digest != previous['integrity']['independent_final448_master_audit']['fp16_tensor_sha256']:
        raise ValueError('Actual readapted393 payload differs from accepted native507')
    checkpoint = previous['integrity']['training_checkpoint']
    state = torch.load(bind(checkpoint['file'], checkpoint['sha256'], checked, checkpoint['bytes']), map_location='cpu', weights_only=True)
    if (state.get('successful_updates') != 448 or set(state.get('masters', {})) != set(values)
            or any(shared.tensor_hash(state['masters'][key].half()) != digest[key] for key in values)):
        raise ValueError('Current final448 masters differ from actual FP16 small export')
    del state
    context = {'binding': package['binding'], 'resolved_files': package['resolved_files'],
               'current_expected507': old_expected, 'old_small': old_small,
               'fixed_large_keys': shared.LARGE_KEYS}
    bind(ROOT / 'models/source' / shared.runtime.TOKENIZER_FILENAME, shared.runtime.TOKENIZER_SHA256, checked)
    return context, values, expected, previous


def load_stage_data(args, data, tokenizer, current_ppl, checked):
    if args.stage == 'full':
        native = importlib.import_module('evaluate_input_axis_residual').native
        tokens, dataset = native.load_wikitext_tokens(tokenizer, 'validation', native.WIKITEXT_REVISION)
        raw_windows = native.ppl_windows(tokens, 2048, None)
        plan = native.window_plan(raw_windows)
        if dataset != current_ppl['full_validation']['dataset'] or plan != current_ppl['full_validation']['window_plan']:
            raise ValueError('Full validation identity differs from historical current')
        return {'windows': [window for _, window in raw_windows], 'plan': plan, 'dataset': dataset}
    if not args.split_manifest_sha:
        raise ValueError('Exact stage split manifest SHA required')
    split = 'dev' if args.stage == 'dev' else 'confirm'
    manifest, rows, tokens = data.load_evaluation(args.dataset_root, split, args.split_manifest_sha, tokenizer)
    if manifest['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('MK data were prepared for another protocol')
    bind(args.dataset_root / split / 'manifest.json', args.split_manifest_sha, checked)
    for name, row in manifest['files'].items():
        bind(args.dataset_root / split / name, row['sha256'], checked, row['bytes'])
    cases = [{**{key: raw[key] for key in ('id', 'N', 'template', 'query_position', 'key', 'answer', 'prompt', 'condition')},
              'prompt_tokens': encoded['prompt_tokens'],
              'prompt_token_sha256_int64le': encoded['prompt_token_sha256_int64le']}
             for raw, encoded in zip(rows, tokens)]
    replay_ids = [row['id'] for row in cases if row['id'].endswith('-s0') or row['id'].endswith('-s0-removed')]
    if len(cases) != 768 or len(replay_ids) != 12 or len({row['prompt_token_sha256_int64le'] for row in cases}) != 768:
        raise ValueError('Incorrect MK coverage/replay set')
    result = {'cases': cases, 'plan': {'replay_case_ids': replay_ids}, 'manifest': manifest,
              'case_plan': [{key: value for key, value in row.items() if key != 'prompt'} for row in cases]}
    if args.stage == 'dev':
        path = ROOT / 'training_data/teacher_kl_compensation_v1/heldout_tokens.pt'
        bind(path, 'a3fc0803052890d973b74f91581a6bfc1d2f93ed28db5725f23c16eb8ec20546', checked)
        windows = torch.load(path, map_location='cpu', weights_only=True)
        if windows.dtype != torch.long or tuple(windows.shape) != (64, 2048):
            raise ValueError('Expected64 frozen prose windows')
        plan = current_ppl['development_plan']
        if any(data.token_digest(window.tolist()) != row['token_sha256_int64le'] for window, row in zip(windows, plan)):
            raise ValueError('Prose heldout token identity differs')
        result.update(windows=list(windows), prose_plan=plan, dataset=current_ppl['dataset'])
    return result


def check_prior_stage(args, identity, checked):
    if args.stage == 'dev':
        if args.prior_report is not None or args.prior_report_sha is not None:
            raise ValueError('DEV does not consume a prior candidate stage')
        return None
    if args.prior_report is None or not args.prior_report_sha:
        raise ValueError('Passed previous-stage report and exact SHA required')
    previous = read_bound(args.prior_report, args.prior_report_sha, checked)
    required = 'dev' if args.stage == 'full' else 'full'
    if (previous.get('format') != FORMAT or previous.get('complete') is not True
            or previous.get('stage') != required or previous.get('gate', {}).get('passed') is not True
            or previous.get('candidate_identity') != identity or previous.get('protocol_sha256') != PROTOCOL_SHA
            or previous.get('script_sha256') != sha(__file__)):
        raise ValueError('Prior-stage identity/completion/joint gate differs')
    return {'file': str(args.prior_report), 'sha256': args.prior_report_sha,
            'stage': required, 'gate': previous['gate']}


def adapter_audit(bank, native, expected, before=None):
    actual = {name: native.tensor_hash(value) for name, value in bank.masters.items()}
    identity = {name: [id(value), value.data_ptr(), value._version] for name, value in bank.masters.items()}
    if actual != expected or any(value.dtype != torch.float16 or value.requires_grad for value in bank.parameters()):
        raise ValueError('Actual eval-only224 adapter tensors differ from serialized export')
    if before is not None and identity != before['identities']:
        raise ValueError('Evaluation mutated adapter identity/storage/version')
    return {'tensor_count': len(actual), 'tensor_sha256': actual, 'identities': identity,
            'gate_mode': bank.gate_mode, 'base_frozen': bank.assert_base_frozen()}


def score_ppl(model, scorer, mk, shared, expected, windows, plan, name, args, report, full):
    entry = report.setdefault('ppl', {})[name] = {}
    before = mk.identities(model)
    entry['identities507_before'] = before
    scorer.score_arm(model, windows, plan, expected, entry, args.report, report, name, full=full)
    entry['identities507_after'] = mk.identities(model)
    if entry['identities507_after'] != before:
        raise ValueError('PPL scoring changed base parameter identity/storage/version')


def mk_exact(first, second):
    return (first['rows'] == second['rows'] and first['summary'] == second['summary']
            and first['cells'] == second['cells'] and first['replay_exact'] is True and second['replay_exact'] is True)


def self_test():
    assert clopper_pearson(0, 1, True) == 0.
    assert clopper_pearson(1, 1, False) == 1.
    assert abs(clopper_pearson(1, 1, True) - .025) < 1e-12
    assert abs(clopper_pearson(0, 1, False) - .975) < 1e-12
    assert abs(clopper_pearson(0, 384, False) - (1 - .025 ** (1 / 384))) < 1e-12
    for k in (1, 15, 192, 383):
        assert abs(clopper_pearson(k, 384, True) - (1 - clopper_pearson(384 - k, 384, False))) < 1e-12
    assert stage_gate('full', current_ppl=7., active_ppl=7.)['passed']
    assert not stage_gate('full', current_ppl=7., active_ppl=math.nextafter(7., math.inf))['passed']
    assert len(expected_inventory()) == 224 and sum(math.prod(shape) for shape in expected_inventory().values()) == PARAMETERS
    assert sorted(expected_schedule()) == list(range(STEPS))
    plan = independent_plan({'answer_target_tokens':10752, 'answer_token_count_histogram':{'7':1536}})
    rows = []
    for j, item in enumerate(plan):
        multiplier = .1+.9*(1+math.cos(math.pi*j/1535))/2
        rows.append({**item, 'attempt':j+1, 'successful_updates_after':j+1, 'overflow':False,
            'mix_lr':1e-4*multiplier, 'router_lr':3e-4*multiplier, 'lr_multiplier':multiplier,
            'gradients':{'finite':True}})
    assert verify_history(rows, plan)['successful_prose_targets'] == 784896
    retry = {**rows[0], 'successful_updates_after':0, 'overflow':True,
             'overflow_optimizer_and_masters_unchanged':True}
    retried = [retry] + [{**row, 'attempt':row['attempt']+1} for row in rows]
    assert verify_history(retried, plan)['overflow_retries'] == 1
    for bad in ([{**rows[0], 'attempt':1.0}]+rows[1:],
                [{**rows[0], 'prose_targets':True}]+rows[1:],
                [{**rows[0], 'schedule_entry':-1}]+rows[1:],
                [{**retry, 'overflow_optimizer_and_masters_unchanged':False}]+retried[1:]):
        try:
            verify_history(bad, plan)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed final training accounting accepted')
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'cuda_initialized': False,
            'checks': ['Exact Clopper-Pearson endpoint/closed-form/symmetry cases',
                       'Strict no-epsilon PPL gate at equality/next-float',
                       'Independent224 tensor inventory and1536 permutation coverage',
                       'Full fixed1536 accounting and same-pair overflow retry',
                       'Reject float/bool counts, altered pair and mutated overflow']}


def run(args, report):
    if any(digest.startswith('PENDING') for digest in CODE_PINS.values()):
        raise ValueError('Trainer/data/native source pins are not frozen')
    checked = {}
    for name, digest in CODE_PINS.items():
        bind(ROOT / name, digest, checked)
    bind(ROOT / 'docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md', PROTOCOL_SHA, checked)
    bind(__file__, sha(__file__), checked)
    shared = importlib.import_module('mamba_e8w5.axis_small_base')
    native = importlib.import_module('mamba_e8w5.resurface_native')
    data = importlib.import_module('mamba_e8w5.resurface_data')
    mk = importlib.import_module('evaluate_readapted_mk')
    evaluation = importlib.import_module('mamba_e8w5.evaluation')
    scorer = importlib.import_module('evaluate_input_axis_residual')
    for pins in (shared.CODE, scorer.frozen.CODE):
        for name, digest in pins.items():
            if Path(name).suffix == '.py':
                bind(ROOT/name, digest, checked)
    context, current_values, expected, previous_ppl = prepare_current(args, shared, checked)
    train_manifest = read_bound(args.dataset_root / 'train/manifest.json', args.train_manifest_sha, checked)
    if (args.train_manifest_sha != TRAIN_MANIFEST_SHA or train_manifest.get('protocol_sha256') != PROTOCOL_SHA or train_manifest.get('split') != 'train'
            or train_manifest.get('row_count') != 1536 or train_manifest.get('contract') != data.CONTRACT['train']
            or train_manifest.get('helper_source_sha256') != CODE_PINS['mamba_e8w5/resurface_data.py']):
        raise ValueError('Frozen TRAIN manifest metadata differs')
    payload, proof = verify_training_export(args, shared, expected, train_manifest, checked, previous_ppl)
    identity = proof['candidate_identity']
    report.update(candidate_identity=identity, export_proof=proof,
                  prior_stage=check_prior_stage(args, identity, checked))
    # Gate checks happen before any raw/token evaluation split is loaded.
    tokenizer = shared.runtime.SentencePieceTokenizer(ROOT / 'models/source')
    stage_data = load_stage_data(args, data, tokenizer, previous_ppl, checked)
    baseline_mk = None
    if args.stage == 'dev':
        if args.baseline_mk_report_sha != BASELINE_MK_SHA:
            raise ValueError('Exact completed original/current MK baseline report SHA required')
        baseline_mk = read_bound(args.baseline_mk_report, BASELINE_MK_SHA, checked)
        if (baseline_mk.get('format') != mk.FORMAT or baseline_mk.get('complete') is not True
                or baseline_mk.get('script_sha256') != CODE_PINS['scripts/evaluate_readapted_mk.py']):
            raise ValueError('Completed frozen MK baseline identity differs')
    report['stage_data'] = {key: value for key, value in stage_data.items() if key not in ('windows', 'cases')}
    report['arms'] = {}
    report['checked_input_sha256'] = checked
    if args.verify_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise ValueError('CPU verify-only requires hidden/uninitialized CUDA')
        for path, digest in checked.items():
            bind(path, digest, {})
        report['final_bound_input_recheck'] = {'file_count': len(checked), 'checked_input_sha256': checked}
        report.update(complete=True, mode='verify_only', cuda_initialized=False)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    model = shared.load_model(context).eval().requires_grad_(False)
    report['direct_old_axis507_audit'] = model._axis_load_audit
    report['installed_readapted393_audit'] = shared.install_small(model, current_values, expected)
    base_identity = mk.identities(model)
    report['initial_current507_audit'] = shared.audit_model(model, expected)
    report['environment'] = shared.runtime.environment_receipt()
    bank = None
    try:
        for arm in (BASE, ACTIVE, RESTORED):
            if arm == ACTIVE:
                bank = native.install_fp16(model, args.adapter, expected_binding=payload['binding'], expected_base_hashes=expected)
                if bank.gate_mode != 'soft':
                    raise ValueError('All tasks require the identical soft-gated adapter')
                before_adapter = adapter_audit(bank, native, proof['adapter_tensor_sha256'])
                report['adapter_before'] = before_adapter
            elif arm == RESTORED:
                report['adapter_after'] = adapter_audit(bank, native, proof['adapter_tensor_sha256'], before_adapter)
                bank.close()
                bank = None
            if args.stage != 'full':
                mk.score_arm(model, tokenizer, evaluation, shared, expected, stage_data['cases'], stage_data['plan'], arm, args, report)
            if args.stage != 'confirm':
                plan = stage_data['plan'] if args.stage == 'full' else stage_data['prose_plan']
                score_ppl(model, scorer, mk, shared, expected, stage_data['windows'], plan, arm, args, report, args.stage == 'full')
            if mk.identities(model) != base_identity:
                raise ValueError('Stage changed the frozen507 model identity')
            write_json(args.report, report)
        if args.stage != 'full':
            if not mk_exact(report['arms'][BASE], report['arms'][RESTORED]):
                raise ValueError('Full768-prompt current MK restoration differs')
            report['mk_restoration_exact'] = True
            if args.stage == 'dev':
                if not mk_exact(report['arms'][BASE], baseline_mk['arms']['current_axis_readapted_small']):
                    raise ValueError('Current DEV MK output/rows differ from completed historical baseline')
                report['historical_current_mk_exact'] = True
                report['historical_mk_report_sha256'] = BASELINE_MK_SHA
            report['mk_comparison'] = mk.paired_comparison(report['arms'][BASE]['rows'], report['arms'][ACTIVE]['rows'])
        if args.stage != 'confirm':
            if not scorer.exact_repeat(report['ppl'][BASE], report['ppl'][RESTORED]):
                raise ValueError('Complete current PPL restoration differs')
            prior = (previous_ppl['full_validation']['arms']['current_axis_readapted_small'] if args.stage == 'full'
                     else previous_ppl['development']['current_axis_readapted_small'])
            if not scorer.exact_repeat(report['ppl'][BASE], prior):
                raise ValueError('Current PPL differs from strict historical per-window reference')
            report.update(ppl_restoration_exact=True, historical_current_ppl_exact=True)
        report['gate'] = stage_gate(args.stage, report.get('mk_comparison'),
            report.get('ppl', {}).get(BASE, {}).get('summary', {}).get('ppl'),
            report.get('ppl', {}).get(ACTIVE, {}).get('summary', {}).get('ppl'))
        report.update(complete=True, mode='native_fp16_staged_evaluation')
    finally:
        if bank is not None:
            bank.close()
        report['final_current507_audit'] = shared.audit_model(model, expected)
        report['final_current507_identity'] = mk.identities(model)
        if report['final_current507_identity'] != base_identity:
            raise ValueError('Final restored base identity differs')
        report['gpu_memory'] = shared.runtime.gpu_memory_receipt()
        for path, digest in checked.items():
            bind(path, digest, {}, None)
        report['final_bound_input_recheck'] = {'file_count': len(checked), 'checked_input_sha256': checked}
        del model
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('dev', 'full', 'confirm'))
    parser.add_argument('--adapter', type=Path, default=ROOT / 'artifacts/resurface_readapted_v1/adapter_fp16.pt')
    parser.add_argument('--training-report', type=Path, default=ROOT / 'reports/resurface_readapted_v1_train.json')
    parser.add_argument('--training-report-sha')
    parser.add_argument('--smoke-report', type=Path, default=ROOT/'reports/resurface_readapted_v1_smoke.json')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'training_data/resurface_readapted_v1')
    parser.add_argument('--train-manifest-sha')
    parser.add_argument('--split-manifest-sha')
    parser.add_argument('--prior-report', type=Path)
    parser.add_argument('--prior-report-sha')
    parser.add_argument('--baseline-mk-report', type=Path, default=ROOT / 'reports/readapted_mk_baseline_v1.json')
    parser.add_argument('--baseline-mk-report-sha', default=BASELINE_MK_SHA)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    for key in ('adapter', 'training_report', 'smoke_report', 'dataset_root', 'prior_report', 'baseline_mk_report', 'report'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if not args.self_test and (args.stage is None or not args.train_manifest_sha or not args.training_report_sha):
        parser.error('Stage, actual TRAIN manifest SHA and final training report SHA required')
    if args.self_test and args.verify_only:
        parser.error('Choose self-test or actual verify-only')
    if args.report.parent != ROOT / 'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('Fresh report directly under reports required')
    if (args.self_test or args.verify_only) and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU work requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(8)
    report = {'format': FORMAT, 'complete': False, 'stage': args.stage, 'script_sha256': sha(__file__),
              'protocol_sha256': PROTOCOL_SHA, 'gate_mode': 'soft',
              'fitting_performed': False, 'training_forward_used_for_evaluation': False,
              'publication_performed': False, 'candidate_promotion': False,
              'limits': ['Same active FP16 export/policy for every task; no task-triggered bypass.',
                         'Native expanded-weight reference; no compact-residency claim.']}
    started = time.monotonic()
    try:
        report['self_test'] = self_test()
        if args.self_test:
            report.update(complete=True, mode='self_test', cuda_initialized=False)
        else:
            run(args, report)
    except BaseException as error:
        report.update(complete=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(args.report, report)
    print(json.dumps({'complete': report['complete'], 'stage': args.stage, 'gate': report.get('gate')}), flush=True)


if __name__ == '__main__':
    main()
