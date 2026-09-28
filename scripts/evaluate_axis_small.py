#!/usr/bin/env python3
"""Independent final448 small-tensor export audit and native paired PPL.

The current all-axis model is loaded directly. No new trainer or training bank
is imported. Only serialized small tensors are installed for quality scoring.
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

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from mamba_e8w5.norm_overlay import other_shapes, sha
from mamba_e8w5.runtime import MODEL_CONFIG

PROTOCOL_SHA = 'a2509bec3d82748b18e452c9bbb5053a894c96ed5309eeb67d45f7dfa6029282'
CURRENT_EVAL_SHA = '23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4'
BASE_HELPER_SHA = '4e7026475fc97b005105b27043a15bae83c576934612b1e346576438eef4da70'
TRAINER_SHA = '7c03ccedc5bd1e5886aa82afac84640ac9d2c0e427455d8fb6bac327f42701af'
DIAGNOSTIC_SHA = 'fc35f16c8a6737a3b9f4ee79f1e91d3d765202f1f92186a89313b3478932bac5'
SCORER_WRAPPER_SHA = 'c8dece7ec209b74ef65244086c7c245f08a59ea344124559f964e44c8a2540f0'
NATIVE_SCORER_SHA = 'c1dd8343a204923d6799f5b4eed31a6f0ffee1bc0774b2deaa15f34e0d54091d'
FORMAT = 'MAMBA2_AXIS_SMALL_NATIVE_EVALUATION_V1'
BASE = 'current_axis_old_small'
PRIMARY = 'current_axis_readapted_small'
REPEAT = 'current_axis_old_small_repeat'
STEPS, TARGETS_PER_STEP, MAX_OVERFLOWS = 448, 2047, 8
BASE_RAW_BYTES = 3138928792
SMALL_PARAMETERS = 3580928
TRAINING_CODE = {
    'mamba_e8w5/axis_small_base.py': BASE_HELPER_SHA,
    'mamba_e8w5/small_training.py': '6e98858460534c4047f7bf75bd62eaf7769d7e46919cd9de6868fd6177a80086',
    'mamba_e8w5/norm_training.py': '25ac3ec6ba167902014018f99fbf8dbbc659e0e4903f80baf28e28a4a3e74087',
    'scripts/train_small_compensation.py': 'd92317e3157f374ef872e40f0de5ad81f9dc6ad9732b19f3df616a114fc5ca7a',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
    'mamba_e8w5/small_overlay.py': '3512c1b232c4039c16a26ffc0fe8c2eac1be6bb9c704ccbcb1d4e8e7788a1286',
    'scripts/prepare_teacher_kl_data.py': '4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13',
    'scripts/train_axis_small.py': TRAINER_SHA,
}
# An independent literal contract; no trainer import is used to establish it.
HYPERPARAMETERS = {
    'seed':20260928,'successful_updates':448,'maximum_attempts':456,'maximum_overflow_retries':8,
    'epochs':1,'windows':448,'stored_tokens':2048,'targets_per_window':2047,
    'optimizer':'AdamW','learning_rate':3e-5,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,
    'gradient_clip_norm':1.,'ce_coefficient':.5,'teacher_to_student_kl_coefficient':.5,'temperature':1.,
    'logits_chunk_tokens':64,'selected_tensors':393,'selected_parameters':3580928,
    'master_dtype':'float32','forward_dtype':'float16','native_eval_mode':True,'checkpoint_each_block':True,
    'loss_scaler':{'init_scale':1024.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000},
    'initialization':'current accepted trained393 FP16; fresh optimizer/scaler',
    'schedule':'prepared seed20260928 torch.randperm448 once','selection':'final448 only',
    'resume':False,'overflow_policy':'same window retry, no master/optimizer update; maximum8',
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def tensor_hash(value):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
        raise ValueError('Expected FP16 tensor for content identity')
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def expected_schedule():
    generator = torch.Generator(device='cpu').manual_seed(20260928)
    return torch.randperm(STEPS, generator=generator).tolist()


def verify_accounting(state):
    schedule = expected_schedule()
    actual = state.get('schedule')
    if (not isinstance(actual, list) or actual != schedule or any(type(v) is not int for v in actual)
            or type(state.get('successful_updates')) is not int or state['successful_updates'] != STEPS
            or type(state.get('attempts')) is not int or type(state.get('overflow_retries')) is not int
            or not isinstance(state.get('history'), list)):
        raise ValueError('Only the declared final448 schedule is eligible')
    successes = overflows = 0
    for number, row in enumerate(state['history'], 1):
        if (not isinstance(row, dict) or successes >= STEPS or type(row.get('attempt')) is not int
                or row['attempt'] != number or type(row.get('overflow')) is not bool
                or type(row.get('schedule_entry')) is not int or row['schedule_entry'] != schedule[successes]
                or type(row.get('targets')) is not int or row['targets'] != TARGETS_PER_STEP
                or type(row.get('successful_updates_after')) is not int):
            raise ValueError('Attempt history differs from fixed schedule/same-window retry')
        overflows += int(row['overflow'])
        successes += int(not row['overflow'])
        if row['successful_updates_after'] != successes:
            raise ValueError('Attempt history successful-update count differs')
    attempts = len(state['history'])
    if (successes != STEPS or overflows > MAX_OVERFLOWS or attempts != STEPS+overflows
            or state['attempts'] != attempts or state['overflow_retries'] != overflows):
        raise ValueError('Final attempt/overflow accounting differs')
    return {'successful_updates': STEPS, 'attempts': attempts, 'overflow_retries': overflows,
        'successful_target_exposures': STEPS*TARGETS_PER_STEP,
        'attempted_target_exposures': attempts*TARGETS_PER_STEP}


def verify_values(values, inventory=None):
    inventory = other_shapes() if inventory is None else inventory
    if not isinstance(values, dict) or set(values) != set(inventory):
        raise ValueError('Exact small-tensor inventory required')
    hashes, count = {}, 0
    for name, shape in inventory.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float16
                or value.layout != torch.strided or list(value.shape) != list(shape)
                or not torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite or incorrect FP16 tensor:{name}')
        if name.endswith('.A_log'):
            effective = -torch.exp(value.float())
            if not torch.isfinite(effective).all() or not (effective < 0).all():
                raise ValueError('Invalid native effective A')
        if name.endswith('.dt_bias'):
            effective = F.softplus(value.float())
            if not torch.isfinite(effective).all() or not (effective > 0).all():
                raise ValueError('Invalid native bias softplus')
        hashes[name] = tensor_hash(value)
        count += value.numel()
    return {'tensor_count': len(hashes), 'parameter_count': count, 'fp16_tensor_sha256': hashes}


def verify_rounded_masters(masters, values, inventory=None):
    inventory = other_shapes() if inventory is None else inventory
    receipt = verify_values(values, inventory)
    if not isinstance(masters, dict) or set(masters) != set(inventory):
        raise ValueError('Final FP32 master inventory differs from export')
    for name, shape in inventory.items():
        master = masters[name]
        if (not isinstance(master, torch.Tensor) or master.dtype != torch.float32
                or master.layout != torch.strided or list(master.shape) != list(shape)
                or not torch.isfinite(master).all()):
            raise ValueError(f'Invalid final FP32 master:{name}')
        rounded = master.half()
        if not torch.isfinite(rounded).all() or tensor_hash(rounded) != receipt['fp16_tensor_sha256'][name]:
            raise ValueError(f'Serialized FP16 tensor differs from final master.half():{name}')
    return receipt


def expected_candidate(base_hashes, exported_hashes):
    small_keys = set(other_shapes())
    if len(base_hashes) != 507 or set(exported_hashes) != small_keys or not small_keys.issubset(base_hashes):
        raise ValueError('Expected507 baseline /393 small replacement coverage')
    candidate = {**base_hashes, **exported_hashes}
    large = {name: digest for name, digest in base_hashes.items() if name not in small_keys}
    if len(large) != 114 or any(candidate[name] != digest for name, digest in large.items()):
        raise ValueError('The114 large parameters must remain fixed')
    # Replacement coverage is393; the changed subset may legitimately be empty.
    changed = sorted(name for name in small_keys if base_hashes[name] != candidate[name])
    return candidate, large, changed


def development_gate(arms):
    if set(arms) != {BASE, PRIMARY}:
        raise ValueError('Both declared native arms must finish before screening')
    base, candidate = (arms[name]['summary']['ppl'] for name in (BASE, PRIMARY))
    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in (base, candidate)):
        raise ValueError('Nonfinite development PPL')
    return {'baseline_arm': BASE, 'primary_arm': PRIMARY, 'required_reduction': .01,
        'baseline_ppl': base, 'primary_ppl': candidate, 'maximum_primary_ppl': .99*base,
        'passed': candidate <= .99*base}


def conditional_full_validation(gate, action):
    if gate.get('passed') is not True:
        return {'performed': False, 'reason': 'Fixed1% development gate failed; validation not loaded.'}
    return action()


def checked_file(path, expected_sha, expected_bytes=None):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or sha(path) != expected_sha
            or (expected_bytes is not None and (type(expected_bytes) is not int or path.stat().st_size != expected_bytes))):
        raise ValueError(f'Bound regular-file identity differs:{path}')
    return path


def load_helpers():
    if any(digest.startswith('PENDING') for digest in (BASE_HELPER_SHA, TRAINER_SHA)):
        raise ValueError('Shared-base/trainer pins must be frozen before native evaluation')
    for path, digest in {
        'docs/AXIS_SMALL_READAPTATION_PROTOCOL.md': PROTOCOL_SHA,
        'mamba_e8w5/axis_small_base.py': BASE_HELPER_SHA,
        'scripts/train_axis_small.py': TRAINER_SHA,
        'scripts/evaluate_input_axis_residual.py': SCORER_WRAPPER_SHA,
        'scripts/diagnose_vocab_w4.py': NATIVE_SCORER_SHA,
    }.items():
        checked_file(ROOT/path, digest)
    return (importlib.import_module('mamba_e8w5.axis_small_base'),
            importlib.import_module('evaluate_input_axis_residual'))


def verify_export(args, context):
    """Verify the actual serialized candidate against an independently loaded448 checkpoint."""
    directory = args.overlay_dir
    if (directory.is_symlink() or not directory.is_dir()
            or {p.name for p in directory.iterdir()} != {'manifest.json','other_fp16.pt'}):
        raise ValueError('Candidate must contain exactly manifest.json and other_fp16.pt')
    checked = {}

    def read_bound(path, digest=None, size=None, json_value=False):
        path = Path(path)
        digest = sha(path) if digest is None else digest
        checked_file(path, digest, size)
        name = str(path.resolve())
        if name in checked and checked[name] != digest:
            raise ValueError('Contradictory bound input identity')
        checked[name] = digest
        return json.loads(path.read_text()) if json_value else path

    for path,digest in {**TRAINING_CODE,
        'docs/AXIS_SMALL_READAPTATION_PROTOCOL.md':PROTOCOL_SHA,
        'scripts/diagnose_axis_small.py':DIAGNOSTIC_SHA,
        'scripts/evaluate_input_axis_residual.py':SCORER_WRAPPER_SHA,
        'scripts/diagnose_vocab_w4.py':NATIVE_SCORER_SHA}.items():
        read_bound(ROOT/path,digest)
    read_bound(Path(__file__))
    binding = {**context['binding'], 'protocol_sha256':PROTOCOL_SHA,
        'hyperparameters':HYPERPARAMETERS, 'trainer_source_sha256':TRAINER_SHA,
        'training_code_sha256':TRAINING_CODE}
    inventory = {name:{'shape':shape,'numel':math.prod(shape)} for name,shape in sorted(other_shapes().items())}
    manifest = read_bound(directory/'manifest.json',json_value=True)
    if (manifest.get('format') != 'MAMBA2_AXIS_SMALL_READAPTATION_V1' or manifest.get('complete') is not True
            or manifest.get('model_config') != MODEL_CONFIG or manifest.get('binding') != binding
            or manifest.get('selected_small_tensors') != inventory):
        raise ValueError('Candidate format, model, fixed recipe or small inventory differs')
    inherited = {name:row for name,row in context['resolved_files'].items() if name != 'other_fp16.pt'}
    large = {name:context['current_expected507'][name] for name in context['fixed_large_keys']}
    initialization = {'small_manifest_sha256':context['binding']['small_manifest_sha256'],
        'other_fp16_sha256':context['binding']['small_values_sha256']}
    if (len(inherited) != 116 or manifest.get('inherited_files') != inherited
            or manifest.get('initialization') != initialization
            or manifest.get('frozen_base_hash_ledger') != context['current_expected507']
            or manifest.get('frozen_large_hash_ledger') != large
            or set(manifest.get('files',{})) != {'other_fp16.pt'}):
        raise ValueError('Initialization,116 inherited files or frozen507/114 ledger differs')
    file_entry = manifest['files']['other_fp16.pt']
    if set(file_entry) != {'bytes','sha256'}:
        raise ValueError('Unexpected exported file ledger fields')
    export_path = read_bound(directory/'other_fp16.pt',file_entry['sha256'],file_entry['bytes'])
    values = torch.load(export_path,map_location='cpu',weights_only=True)
    exported = verify_values(values)
    if (exported['tensor_count'] != 393 or exported['parameter_count'] != SMALL_PARAMETERS
            or manifest.get('exported_small_fp16_sha256') != exported['fp16_tensor_sha256']):
        raise ValueError('Actual393 exported contents differ from candidate ledger')

    train = read_bound(args.training_report,json_value=True)
    if (train.get('format') != 'MAMBA2_AXIS_SMALL_TRAINING_REPORT_V1' or train.get('mode') != 'train'
            or train.get('complete') is not True or type(train.get('final_step')) is not int or train['final_step'] !=448
            or train.get('binding') != binding or train.get('script_sha256') != TRAINER_SHA
            or train.get('selected_small_tensors') != inventory
            or train.get('protocol_sha256') != PROTOCOL_SHA
            or train.get('training_split') != 'train' or train.get('heldout_used_for_fitting') is not False
            or train.get('validation_loaded') is not False or train.get('initialization_selected_by_diagnostic') is not False):
        raise ValueError('Complete fixed448 training receipt required')
    counts = verify_accounting(train)
    if any(type(train.get(k)) is not int or train[k] != v for k,v in counts.items()):
        raise ValueError('Final training exposure accounting differs')
    for row in train['history']:
        grad = row.get('gradients',{})
        if (type(grad.get('tensors')) is not int or grad['tensors'] != 393
                or type(grad.get('parameters')) is not int or grad['parameters'] != SMALL_PARAMETERS
                or grad.get('dtypes') != ['torch.float32']
                or (not row['overflow'] and grad.get('finite') is not True)
                or (row['overflow'] and row.get('overflow_optimizer_and_masters_unchanged') is not True)):
            raise ValueError('Recorded393 gradient coverage/overflow immutability differs')

    training = manifest.get('training_receipt',{})
    if set(training) != {'successful_updates','attempts','overflow_retries','checkpoint_sha256','checkpoint_bytes','smoke_report_sha256'}:
        raise ValueError('Final manifest training fields differ')
    for name in ('successful_updates','attempts','overflow_retries'):
        if type(training[name]) is not int or training[name] != counts[name]:
            raise ValueError('Manifest/checkpoint training counts differ')
    smoke = read_bound(args.smoke_report,training['smoke_report_sha256'],json_value=True)
    if (smoke.get('format') != train['format'] or smoke.get('mode') != 'smoke'
            or smoke.get('complete') is not True or smoke.get('passed') is not True
            or smoke.get('binding') != binding or smoke.get('trial_state_discarded') is not True
            or train.get('smoke_report_sha256') != training['smoke_report_sha256']):
        raise ValueError('Identical-binding passed discarded smoke required')
    diagnostic_digest = manifest.get('diagnostic_report_sha256')
    diagnostic = read_bound(args.diagnostic_report,diagnostic_digest,json_value=True)
    if (diagnostic.get('format') != 'MAMBA2_AXIS_SMALL_DIAGNOSTIC_V1'
            or diagnostic.get('complete') is not True or diagnostic.get('correctness_passed') is not True
            or diagnostic.get('baseline_repeat_exact') is not True or diagnostic.get('fixed114_verified') is not True
            or diagnostic.get('binding') != context['binding'] or diagnostic.get('script_sha256') != DIAGNOSTIC_SHA
            or diagnostic.get('training_performed') is not False or diagnostic.get('validation_loaded') is not False
            or diagnostic.get('candidate_advancement') is not False
            or diagnostic.get('final_current507_restored_audit',{}).get('unchanged_content_sha256') != context['current_expected507']
            or train.get('diagnostic_report_sha256') != diagnostic_digest):
        raise ValueError('Valid fixed descriptive diagnostic required without initialization selection')
    for name,path in (('manifest',directory/'manifest.json'),('diagnostic_receipt',args.diagnostic_report)):
        receipt = train.get(name,{})
        if (Path(receipt.get('file','')).resolve() != path.resolve()
                or receipt.get('sha256') != checked[str(path.resolve())]
                or type(receipt.get('bytes')) is not int or receipt['bytes'] != path.stat().st_size):
            raise ValueError(f'Training report actual receipt mismatch:{name}')

    cp_receipt = train.get('final_checkpoint',{})
    checkpoint_path = args.training_checkpoint or Path(cp_receipt.get('file',''))
    if (Path(cp_receipt.get('file','')).resolve() != checkpoint_path.resolve()
            or ROOT not in checkpoint_path.resolve().parents
            or checkpoint_path.name != f"checkpoint_attempt{counts['attempts']:03d}_step448_final.pt"
            or cp_receipt.get('sha256') != training['checkpoint_sha256']
            or cp_receipt.get('bytes') != training['checkpoint_bytes']):
        raise ValueError('Actual final448 checkpoint location/identity differs')
    read_bound(checkpoint_path,training['checkpoint_sha256'],training['checkpoint_bytes'])
    state = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if (state.get('format') != 'MAMBA2_AXIS_SMALL_TRAINING_CHECKPOINT_V1' or state.get('reason') != 'final'
            or state.get('resumable') is not False or state.get('binding') != binding
            or state.get('smoke_report_sha256') != training['smoke_report_sha256']
            or state.get('diagnostic_report_sha256') != diagnostic_digest
            or state.get('frozen_model_sha256') != context['current_expected507']
            or state.get('teacher_model_sha256') != context['source_expected507']):
        raise ValueError('Final checkpoint provenance/frozen source differs')
    if verify_accounting(state) != counts or any(state[key] != train[key] for key in
            ('schedule','history','successful_updates','attempts','overflow_retries')):
        raise ValueError('Final checkpoint and training report histories differ')
    master_proof = verify_rounded_masters(state.get('masters'),values)
    del state
    candidate,_,changed = expected_candidate(context['current_expected507'],exported['fp16_tensor_sha256'])
    final = train.get('final_export',{})
    frow = final.get('file',{})
    comparison = final.get('comparison',{})
    if (final.get('checkpoint448_master_to_fp16_bitwise_equal') is not True
            or final.get('tensor_count') !=393 or final.get('parameter_count') !=SMALL_PARAMETERS
            or Path(frow.get('file','')).resolve() != export_path.resolve()
            or {k:frow.get(k) for k in ('bytes','sha256')} != file_entry
            or comparison.get('fp16_tensor_sha256') != exported['fp16_tensor_sha256']
            or sorted(comparison.get('actual_changed_keys',[])) != changed
            or comparison.get('actual_changed_tensor_count') != len(changed)):
        raise ValueError('Final export/report actual changed-tensor accounting differs')
    for document in (train,smoke):
        for key in ('initial_underlying507_audit','final_underlying507_audit'):
            if document.get(key,{}).get('unchanged_content_sha256') != context['current_expected507']:
                raise ValueError('Recorded frozen underlying507 content differs')
        for key in ('initial_teacher507_audit','final_teacher507_audit'):
            if document.get(key,{}).get('unchanged_content_sha256') != context['source_expected507']:
                raise ValueError('Recorded original teacher507 content differs')
        if document.get('final_fixed114_audit',{}).get('unchanged_content_sha256') != large:
            raise ValueError('Recorded fixed114 content differs')
        parity = document.get('export_parity',{})
        if (parity.get('hidden_bitwise_equal') is not True or parity.get('logits_bitwise_equal') is not True
                or parity.get('input_tokens') !=128 or parity.get('logit_positions') !=8
                or parity.get('functional_hidden_sha256') != parity.get('native_hidden_sha256')
                or parity.get('functional_logits_sha256') != parity.get('native_logits_sha256')):
            raise ValueError('Recorded native export hidden/logit parity differs')
    if train.get('post_export_underlying507_audit',{}).get('unchanged_content_sha256') != context['current_expected507']:
        raise ValueError('Post-export frozen507 ledger differs')
    old_bytes = context['resolved_files']['other_fp16.pt']['bytes']
    storage = {'current_logical_data_bytes':BASE_RAW_BYTES,'replaced_other_bytes':old_bytes,
        'replacement_other_bytes':file_entry['bytes'],'archive_byte_delta':file_entry['bytes']-old_bytes,
        'logical_inference_data_bytes':BASE_RAW_BYTES+file_entry['bytes']-old_bytes,
        'fp16_parameter_payload_bytes':2*SMALL_PARAMETERS,'manifest_bytes':(directory/'manifest.json').stat().st_size,
        'complete_distribution_built':False}
    if manifest.get('storage') != storage or train.get('storage') != storage:
        raise ValueError('Actual storage byte scopes differ')
    proof = {'binding':binding,'overlay_manifest_sha256':checked[str((directory/'manifest.json').resolve())],
        'training_report_sha256':checked[str(args.training_report.resolve())],
        'training_checkpoint':{'file':str(checkpoint_path),'sha256':training['checkpoint_sha256'],'bytes':training['checkpoint_bytes']},
        'smoke_report_sha256':training['smoke_report_sha256'],'diagnostic_report_sha256':diagnostic_digest,
        'training_accounting':counts,'independent_final448_master_audit':master_proof,
        'resolved507_candidate_fp16_sha256':candidate,'fixed114_fp16_sha256':large,
        'actual_changed_small_keys':changed,'actual_changed_small_count':len(changed),
        'inherited_file_count':116,'storage':storage,'checked_input_sha256':checked,
        'scope':'CPU file/master/provenance verification; training GPU hashes/parity above are recorded evidence. Quality run independently audits freshly loaded native507 tensors.'}
    return manifest, values, proof


def run(args, report):
    shared, scorer = load_helpers()
    context = shared.verify_inputs()
    manifest, values, proof = verify_export(args, context)
    candidate, fixed_large, changed = expected_candidate(context['current_expected507'],
        proof['independent_final448_master_audit']['fp16_tensor_sha256'])
    report.update(integrity=proof, dataset=context['data_manifest']['dataset'],
        development_plan=context['heldout_plan'], development={},
        full_validation={'performed': False, 'reason': 'Development gate not yet evaluated'},
        actual_changed_small_keys=changed, actual_changed_small_count=len(changed),
        replaced_small_count=393, fixed_large_count=114)
    write_json(args.report, report)
    if args.verify_only:
        if torch.cuda.is_initialized():
            raise ValueError('CPU verification initialized CUDA')
        report['final_base_input_recheck'] = shared.final_recheck(context)
        for path,digest in proof['checked_input_sha256'].items():
            checked_file(path,digest)
        report.update(complete=True, mode='verify_only', cuda_initialized=False)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    model = shared.load_model(context, device='cuda').eval().requires_grad_(False)
    native = scorer.native
    report['environment'] = native.environment_receipt()
    report['direct_current_load_audit'] = model._axis_load_audit
    report['current_package_receipt'] = model._package_receipt
    report['reproducibility_before'] = scorer.reproducibility_snapshot()
    large_identity = shared.large_identity(model)
    modified = False

    def assert_large():
        return shared.verify_fixed_large(model,context,large_identity)

    def restore():
        nonlocal modified
        if modified:
            report['restored393_small_audit'] = shared.install_small(model, context['old_small'],
                expected=context['current_expected507'])
            modified = False

    def score(name, destination, expected, windows, plan, full=False):
        destination['fixed114_before'] = assert_large()
        scorer.score_arm(model, windows, plan, expected, destination, args.report, report, name, full=full)
        destination['fixed114_after'] = assert_large()

    def complete_validation():
        # This is the first validation-data access; the callback is only called on PASS.
        restore()
        tokenizer = native.SentencePieceTokenizer(ROOT/'models/source')
        ids, dataset = native.load_wikitext_tokens(tokenizer, 'validation', native.WIKITEXT_REVISION)
        all_windows = native.ppl_windows(ids, 2048, None)
        plan = native.window_plan(all_windows)
        previous = json.loads((ROOT/'reports/output_axis_residual_v1_eval.json').read_text())
        historical = previous['full_validation']
        if dataset != historical['dataset'] or plan != historical['window_plan']:
            raise ValueError('Established full-validation token/window identity changed')
        windows = [tokens for _, tokens in all_windows]
        full = {'performed': True, 'reason': 'Fixed1% development gate passed',
                'dataset': dataset, 'window_plan': plan, 'arms': {}}
        report['full_validation'] = full
        source = native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
        full['source_package_receipt'] = source._package_receipt
        entry = full['arms']['source_fp16'] = {}
        scorer.score_arm(source, windows, plan, context['source_expected507'], entry,
            args.report, report, 'source_fp16', full=True)
        del source
        gc.collect()
        torch.cuda.empty_cache()
        entry = full['arms'][BASE] = {}
        score(BASE, entry, context['current_expected507'], windows, plan, full=True)
        # Re-read serialized small values for this fresh native candidate installation.
        checked_file(args.overlay_dir/'other_fp16.pt', manifest['files']['other_fp16.pt']['sha256'],
            manifest['files']['other_fp16.pt']['bytes'])
        reloaded = torch.load(args.overlay_dir/'other_fp16.pt', map_location='cpu', weights_only=True)
        if verify_values(reloaded)['fp16_tensor_sha256'] != proof['independent_final448_master_audit']['fp16_tensor_sha256']:
            raise ValueError('Reloaded candidate small content changed')
        nonlocal modified
        modified = True
        full['reloaded393_small_audit'] = shared.install_small(model, reloaded, expected=candidate)
        del reloaded
        entry = full['arms'][PRIMARY] = {}
        score(PRIMARY, entry, candidate, windows, plan, full=True)
        full['comparison'] = scorer.paired_comparison(full['arms'], (PRIMARY,), BASE)
        src, old, new = [full['arms'][name]['summary']['ppl'] for name in ('source_fp16', BASE, PRIMARY)]
        full['gates'] = {
            'meaningful_improvement_reference': {'required_reduction': .01, 'met': new <= .99*old},
            'source_plus5_percent_reference': {'maximum_ppl': 1.05*src, 'met': new <= 1.05*src},
            'candidate_vs_source_ppl_relative_change': new/src-1}
        full['historical_baseline_drift'] = {
            'source_relative_ppl': src/historical['arms']['source_fp16']['summary']['ppl']-1,
            'current_relative_ppl': old/historical['arms']['axis_in_out_w4_w5']['summary']['ppl']-1,
            'scope': 'Descriptive only; quality gates use same-process paired scores.'}
        return full

    try:
        entry = report['development'][BASE] = {}
        score(BASE, entry, context['current_expected507'], context['heldout_windows'], context['heldout_plan'])
        modified = True
        report['installed393_candidate_audit'] = shared.install_small(model, values, expected=candidate)
        entry = report['development'][PRIMARY] = {}
        score(PRIMARY, entry, candidate, context['heldout_windows'], context['heldout_plan'])
        restore()
        entry = report['development'][REPEAT] = {}
        score(REPEAT, entry, context['current_expected507'], context['heldout_windows'], context['heldout_plan'])
        if not scorer.exact_repeat(report['development'][BASE], entry):
            raise ValueError('Same-process64-window baseline repetition differs')
        report['baseline_repeat_exact'] = True
        report['development_comparison'] = scorer.paired_comparison(report['development'], (PRIMARY,), BASE)
        report['development_gate'] = development_gate({name:report['development'][name] for name in (BASE, PRIMARY)})
        write_json(args.report, report)
        report['full_validation'] = conditional_full_validation(report['development_gate'], complete_validation)
        report.update(complete=True, mode='native_paired_development_conditional_full')
    finally:
        restore()
        report['final_fixed114_audit'] = assert_large()
        report['final_current507_restored_audit'] = shared.audit_model(model, context['current_expected507'])
        report['reproducibility_after'] = scorer.reproducibility_snapshot()
        report['gpu_memory'] = native.gpu_memory_receipt()
        report['final_base_input_recheck'] = shared.final_recheck(context)
        for path, digest in proof['checked_input_sha256'].items():
            checked_file(path, digest)


def self_test():
    schedule = expected_schedule()
    history = [{'attempt':i+1, 'overflow':False, 'schedule_entry':window, 'targets':2047,
                'successful_updates_after':i+1} for i,window in enumerate(schedule)]
    state = {'schedule':schedule, 'successful_updates':448, 'attempts':448, 'overflow_retries':0, 'history':history}
    assert verify_accounting(state)['successful_target_exposures'] == 917056
    retries = [{'attempt':i+1,'overflow':True,'schedule_entry':schedule[0],
                'targets':2047,'successful_updates_after':0} for i in range(8)]
    with_retries = {**state,'attempts':456,'overflow_retries':8,
        'history':retries+[{**row,'attempt':row['attempt']+8} for row in history]}
    assert verify_accounting(with_retries)['attempted_target_exposures'] ==456*2047
    for bad in ({**state,'successful_updates':447}, {**state,'attempts':True},
            {**state,'history':[{**history[0],'targets':2047.0}]+history[1:]},
            {**state,'history':[{**history[0],'schedule_entry':schedule[1]}]+history[1:]},
            {**with_retries,'overflow_retries':9}):
        try:
            verify_accounting(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed final accounting accepted')
    inv = {'toy': [2]}
    master = {'toy': torch.tensor([.2, -.3], dtype=torch.float32)}
    values = {name:value.half() for name,value in master.items()}
    verify_rounded_masters(master, values, inv)
    bad = {'toy': values['toy']+1}
    try:
        verify_rounded_masters(master,bad,inv)
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong rounded final master accepted')
    for bad in ({'toy':torch.tensor([float('inf'),0],dtype=torch.float16)},
                {'toy':torch.zeros(3,dtype=torch.float16)}, {},
                {'toy':torch.zeros(2,dtype=torch.float32)}):
        try:
            verify_values(bad,inv)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid export values accepted')
    try:
        verify_values({'x.A_log':torch.tensor([-200],dtype=torch.float16)}, {'x.A_log':[1]})
    except ValueError:
        pass
    else:
        raise AssertionError('Underflowed native A accepted')
    try:
        verify_rounded_masters({'toy':torch.tensor([-0.,1.],dtype=torch.float32)},
            {'toy':torch.tensor([0.,1.],dtype=torch.float16)},inv)
    except ValueError:
        pass
    else:
        raise AssertionError('Signed-zero master mismatch accepted')
    small_keys = set(other_shapes())
    large_keys = {f'backbone.layers.{i}.mixer.{part}.weight' for i in range(56) for part in ('in_proj','out_proj')}
    large_keys.update(('backbone.embedding.weight','lm_head.weight'))
    base = {name:hashlib.sha256(name.encode()).hexdigest() for name in small_keys|large_keys}
    unchanged = {name:base[name] for name in small_keys}
    candidate,large,changed = expected_candidate(base,unchanged)
    assert candidate == base and not changed and len(large)==114
    arms = {BASE:{'summary':{'ppl':100.}}, PRIMARY:{'summary':{'ppl':99.}}}
    assert development_gate(arms)['passed']
    arms[PRIMARY]['summary']['ppl'] = 99.00001
    gate = development_gate(arms)
    assert not gate['passed']
    def forbidden():
        raise AssertionError('Failed gate accessed full validation')
    assert conditional_full_validation(gate,forbidden)['performed'] is False
    assert not torch.cuda.is_initialized()
    return {'passed':True, 'cuda_initialized':False, 'checks':[
        'Fixed448 schedule and8 same-window overflow retries; strict count types',
        'Bitwise rounded master/export equality including signed zero',
        'Reject nonfinite/wrong dtype/shape/missing export and underflowed native A',
        '393 replacements may include unchanged tensors;114 fixed',
        'Exact1% threshold; failed gate never invokes full-validation callback']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay-dir', type=Path, default=ROOT/'artifacts/axis_small_readaptation_v1')
    parser.add_argument('--training-report', type=Path, default=ROOT/'reports/axis_small_readaptation_v1_train.json')
    parser.add_argument('--training-checkpoint', type=Path)
    parser.add_argument('--smoke-report', type=Path, default=ROOT/'reports/axis_small_readaptation_v1_smoke.json')
    parser.add_argument('--diagnostic-report', type=Path, default=ROOT/'reports/axis_small_diagnostic_v1.json')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/axis_small_readaptation_v1_eval.json')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    for key in ('overlay_dir','training_report','smoke_report','diagnostic_report','report'):
        setattr(args,key,getattr(args,key).resolve())
    if args.training_checkpoint is not None:
        args.training_checkpoint = args.training_checkpoint.resolve()
    if args.verify_only and args.self_test:
        parser.error('Choose self-test or verify-only')
    if args.report.parent != ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('A fresh report directly under reports is required')
    torch.set_num_threads(8)
    report = {'format':FORMAT, 'complete':False, 'script_sha256':sha(__file__),
        'protocol_sha256':PROTOCOL_SHA, 'development_arm_order':[BASE,PRIMARY,REPEAT],
        'primary_arm':PRIMARY, 'same_process':True, 'training_forward_used_for_ppl':False,
        'candidate_advancement':False, 'publication_performed':False, 'mk_performed':False,
        'limitations':['Development and validation have already informed research; no untouched test claim.',
            'A fixed small-tensor payload does not imply unchanged serialized/container bytes.',
            'Native FP16 reference execution is not compact inference residency.']}
    started = time.monotonic()
    try:
        if args.self_test:
            report.update(self_test=self_test(), complete=True, mode='self_test', cuda_initialized=False)
        else:
            run(args,report)
    except BaseException as error:
        report.update(complete=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(args.report,report)
    print(json.dumps({'complete':report['complete'], 'report':str(args.report),
        'development_gate':report.get('development_gate'),
        'full_validation_performed':report.get('full_validation',{}).get('performed',False)}),flush=True)


if __name__ == '__main__':
    main()
