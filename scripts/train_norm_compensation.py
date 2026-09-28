#!/usr/bin/env python3
"""Bounded train-only norm compensation; smoke first, fixed final128 export.

This file never evaluates validation/test/MK or uploads a model. Large data and
resumable optimizer checkpoints stay on the GPU host. Both modes bind the same
math, inputs, protocol and implementation. No frozen pipeline is modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from mamba_e8w5 import norm_overlay as overlay
from mamba_e8w5.norm_training import (HYPERPARAMETERS, NormMasters, selected_inventory,
    training_schedule, chunked_loss_backward, gradient_receipt, tensor_hash, export_parity)
from mamba_e8w5.runtime import (MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256,
    load_quantized_model, load_source_model, environment_receipt, gpu_memory_receipt, token_digest)

CALIBRATION_SHA = '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
SELECTED_TOKENS_SHA = '893bf7df37895a98e00116db9cb7a3189e5ee2e8f991e35594b970087958db34'
SMOKE_TOLERANCES = {'checkpoint_gradient_rtol': .005, 'checkpoint_gradient_atol': .00003,
                    'chunk_hidden_gradient_rtol': .005, 'chunk_hidden_gradient_atol': .001,
                    'chunk_objective_absolute': .00001}


def sha(path):
    return overlay.sha(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(path, state):
    """Unique durable checkpoint; no overwrite of an earlier attempt."""
    if path.exists():
        raise FileExistsError(f'Preserving existing checkpoint: {path}')
    temporary = path.with_suffix('.pt.tmp')
    if temporary.exists():
        raise FileExistsError(f'Preserving incomplete checkpoint: {temporary}')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return {'file': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: cpu_tree(item) for key,item in value.items()}
    if isinstance(value, list): return [cpu_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_tree(item) for item in value)
    return value


def tree_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), str(item.dtype))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple,list)):
            for child in item: visit(child)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def parameter_hashes(model, excluded=()):
    """Canonical FP16 tensor hashes with <=32MiB host transfer at once."""
    result = {}
    for name, parameter in model.named_parameters():
        if name in excluded: continue
        if parameter.dtype != torch.float16:
            raise ValueError(f'Non-FP16 model parameter: {name}')
        digest = hashlib.sha256()
        flat = parameter.detach().reshape(-1)
        for first in range(0, flat.numel(), 16 << 20):
            digest.update(flat[first:first+(16 << 20)].cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def verify_inputs(args):
    parent_path = args.parent_dir/'manifest.json'
    if sha(parent_path) != overlay.PARENT_MANIFEST_SHA256:
        raise ValueError('This experiment requires the original two-sweep parent')
    parent = json.loads(parent_path.read_text())
    if (not parent['complete'] or parent['model_config'] != MODEL_CONFIG
            or parent['source_checkpoint_sha256'] != SOURCE_CHECKPOINT_SHA256
            or parent['tokenizer_sha256'] != TOKENIZER_SHA256 or parent['binding']['tune_iters'] != 2):
        raise ValueError('Parent architecture/source/recipe differs')
    for name, expected in parent['binding']['code_sha256'].items():
        if sha(ROOT/'mamba_e8w5'/name) != expected:
            raise ValueError(f'Frozen implementation changed: {name}')
    for name, expected in parent['binding']['quip_sha256'].items():
        if sha(ROOT/'third_party/quip-sharp'/name) != expected:
            raise ValueError(f'Frozen QuIP source changed: {name}')
    if {p.name for p in args.parent_dir.iterdir()} != set(parent['files'])|{'manifest.json'}:
        raise ValueError('Parent file inventory differs')
    for name, entry in parent['files'].items(): overlay.checked_file(args.parent_dir, name, entry)
    if sha(args.protocol) != overlay.PROTOCOL_SHA256:
        raise ValueError('Protocol differs')
    calibration_path = args.calibration_dir/'manifest.json'
    token_path = args.calibration_dir/'calibration_tokens.pt'
    if sha(calibration_path) != CALIBRATION_SHA or parent['binding']['hessian_manifest_sha256'] != CALIBRATION_SHA:
        raise ValueError('Calibration manifest differs')
    if sha(token_path) != overlay.CALIBRATION_TOKENS_SHA256:
        raise ValueError('Calibration token file differs')
    calibration = json.loads(calibration_path.read_text())
    if (not calibration['complete'] or calibration['calibration_split'] != 'train'
            or calibration['evaluation_data_used'] is not False or calibration['nwin'] != 32
            or calibration['seqlen'] != 2048 or calibration['source_checkpoint_sha256'] != SOURCE_CHECKPOINT_SHA256
            or calibration['dataset']['tokenizer_sha256'] != TOKENIZER_SHA256):
        raise ValueError('Not the declared train-only calibration stream')
    windows = torch.load(token_path, map_location='cpu', weights_only=True)
    if windows.dtype != torch.int64 or list(windows.shape) != [32,2048] or token_digest(windows.flatten().numpy()) != SELECTED_TOKENS_SHA:
        raise ValueError('Training token tensor differs')
    if int(windows.min()) < 0 or int(windows.max()) >= 256000:
        raise ValueError('Invalid training token IDs')
    if selected_inventory() != overlay.selected_norms(): raise ValueError('Norm inventories disagree')
    binding = {'source_checkpoint_sha256': SOURCE_CHECKPOINT_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
        'hessian_manifest_sha256': CALIBRATION_SHA, 'calibration_tokens_sha256': sha(token_path),
        'protocol_sha256': sha(args.protocol), 'trainer_source_sha256': sha(__file__),
        'training_helper_source_sha256': sha(ROOT/'mamba_e8w5/norm_training.py'),
        'norm_overlay_source_sha256': sha(overlay.__file__), 'hyperparameters': HYPERPARAMETERS,
        'frozen_code_sha256': parent['binding']['code_sha256'],
        'evaluation_source_sha256': sha(ROOT/'mamba_e8w5/evaluation.py')}
    return parent, windows, binding


def optimizer_for(bank):
    return torch.optim.AdamW(bank.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)


def scaler_for():
    return torch.amp.GradScaler('cuda', init_scale=1024., growth_factor=2., backoff_factor=.5, growth_interval=2000)


def attempt_update(bank, teacher, window, optimizer, scaler):
    """Exactly one forward/backward attempt; master/state mutation only if finite."""
    optimizer.zero_grad(set_to_none=True)
    ids, targets = window[None,:-1].cuda(), window[None,1:].cuda()
    with torch.no_grad(): teacher_hidden = teacher.backbone(ids)
    student_hidden = bank.forward(ids, use_checkpoint=True)
    scale_before = scaler.get_scale()
    values = chunked_loss_backward(student_hidden, teacher_hidden, targets,
                                  bank.model.lm_head, teacher.lm_head, scaler, 64)
    del student_hidden, teacher_hidden
    scaler.unscale_(optimizer)
    gradients = gradient_receipt(bank)
    overflow = not gradients['finite'] or not values['scaled_hidden_gradient_finite']
    if not overflow and gradients['nonzero'] == 0:
        raise RuntimeError('All selected gradients are zero; refusing silent underflow')
    if overflow:
        # Public update(new_scale) also clears per-optimizer unscale bookkeeping.
        # No optimizer.step is called, including the rare case where only the
        # staged hidden gradient overflowed while master gradients are finite.
        master_before = tree_digest(bank.state_dict())
        optimizer_before = tree_digest(optimizer.state_dict())
        scaler.update(new_scale=scale_before*.5)
        if master_before != tree_digest(bank.state_dict()) or optimizer_before != tree_digest(optimizer.state_dict()):
            raise RuntimeError('Overflow retry mutated masters or optimizer state')
        gradient_norm = None
    else:
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), 1., error_if_nonfinite=True))
        scaler.step(optimizer); scaler.update()
        if any(not torch.isfinite(p).all() for p in bank.parameters()):
            raise FloatingPointError('Optimizer produced nonfinite norm masters')
    bank.assert_frozen()
    optimizer.zero_grad(set_to_none=True)
    return {**values, 'overflow': overflow, 'loss_scale_before': scale_before, 'loss_scale_after': scaler.get_scale(),
            'gradient_norm_before_clip': gradient_norm,
            'gradients': {key:value for key,value in gradients.items() if key != 'per_tensor'},
            'overflow_optimizer_and_masters_unchanged': True if overflow else None}


def gradient_comparison(first, second, *, rtol, atol):
    differences = {}
    for name in first:
        a,b = first[name].float(), second[name].float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError(f'Nonfinite smoke gradient: {name}')
        differences[name] = float((a-b).abs().max())
        if not torch.allclose(a,b,rtol=rtol,atol=atol):
            raise RuntimeError(f'Smoke gradient mismatch: {name}, maxabs={differences[name]}')
    return {'passed': True, 'rtol': rtol, 'atol': atol, 'maximum_absolute_difference': max(differences.values()),
            'tensor_count': len(differences)}


def smoke(bank, teacher, windows, parent_other, args, report):
    ids = windows[0,None,:128].cuda()
    print('[smoke] initial native/functional hidden and output parity', flush=True)
    with torch.no_grad():
        native = bank.model.backbone(ids)
        functional = bank.forward(ids, use_checkpoint=False)
        if not torch.equal(native, functional): raise RuntimeError('Initial native/functional hidden mismatch')
        native_logits = bank.model.lm_head(native[:,-1:])
        functional_logits = bank.model.lm_head(functional[:,-1:])
        if not torch.equal(native_logits, functional_logits): raise RuntimeError('Initial output mismatch')
    report['initial_forward'] = {'hidden_bitwise_equal': True, 'logits_bitwise_equal': True,
                                 'input_tokens': 128, 'hidden_sha256': tensor_hash(native)}
    del native, functional, native_logits, functional_logits
    report['tolerances'] = SMOKE_TOLERANCES
    print('[smoke] checkpoint recomputation and selected gradient parity', flush=True)
    # Changed masters explicitly test replay's functional parameter mapping.
    initial = bank.state_dict()
    with torch.no_grad():
        for p in bank.parameters(): p.add_(torch.linspace(-.003,.003,p.numel(),device=p.device))
    grad_sets, hidden_hashes = [], []
    for enabled in (False, True):
        for p in bank.parameters(): p.grad = None
        hidden = bank.forward(ids, use_checkpoint=enabled)
        hidden_hashes.append(tensor_hash(hidden))
        tangent = torch.linspace(-1,1,hidden.shape[-1],device=hidden.device)
        (1024*(hidden.float()*tangent).mean()).backward()
        receipt = gradient_receipt(bank)
        if not receipt['finite'] or any(x['nonzero'] == 0 for x in receipt['per_tensor'].values()):
            raise RuntimeError('Checkpoint smoke needs finite nonzero gradient for every norm')
        grad_sets.append({name:p.grad.detach().cpu().clone() for name,p in bank.masters.items()})
        del hidden
    if hidden_hashes[0] != hidden_hashes[1]: raise RuntimeError('Checkpoint forward mismatch')
    report['checkpoint_parity'] = gradient_comparison(*grad_sets,
        rtol=SMOKE_TOLERANCES['checkpoint_gradient_rtol'], atol=SMOKE_TOLERANCES['checkpoint_gradient_atol'])
    report['checkpoint_parity'].update({'hidden_bitwise_equal':True, 'perturbed_masters':True, 'gradient_coverage':receipt})
    bank.load_state_dict(initial)
    for p in bank.parameters(): p.grad = None
    del grad_sets
    torch.cuda.empty_cache()
    print('[smoke] chunked versus full vocabulary loss and hidden gradient', flush=True)
    with torch.no_grad():
        student_h = bank.forward(ids[:,:5], False)
        teacher_h = teacher.backbone(ids[:,:5])
    targets = windows[0,None,1:6].cuda()
    staged = student_h.detach().clone().requires_grad_()
    staged_receipt = chunked_loss_backward(staged, teacher_h, targets, bank.model.lm_head, teacher.lm_head, None, 3)
    full = student_h.detach().clone().requires_grad_()
    logits = bank.model.lm_head(full).float()
    with torch.no_grad(): teacher_logp = F.log_softmax(teacher.lm_head(teacher_h).float(), -1)
    ce = F.cross_entropy(logits.flatten(0,1), targets.flatten(), reduction='sum')/5
    kl = F.kl_div(F.log_softmax(logits,-1), teacher_logp, log_target=True, reduction='sum')/5
    loss = .5*(ce+kl); loss.backward()
    difference = abs(staged_receipt['loss']-float(loss.detach()))
    if difference > SMOKE_TOLERANCES['chunk_objective_absolute']: raise RuntimeError('Chunk loss mismatch')
    report['chunked_loss_parity'] = gradient_comparison({'hidden':staged.grad},{'hidden':full.grad},
        rtol=SMOKE_TOLERANCES['chunk_hidden_gradient_rtol'], atol=SMOKE_TOLERANCES['chunk_hidden_gradient_atol'])
    report['chunked_loss_parity'].update({'tokens':5,'chunk_tokens':3,'objective_absolute_difference':difference,
        'chunked_loss':staged_receipt['loss'],'unchunked_loss':float(loss.detach())})
    del staged, full, logits, teacher_logp, teacher_h, student_h, loss, ce, kl
    torch.cuda.empty_cache()
    print('[smoke] full 2047-target backward, trial update and FP16 export', flush=True)
    optimizer, scaler = optimizer_for(bank), scaler_for()
    trials = []
    for trial in range(9):
        values = attempt_update(bank, teacher, windows[0], optimizer, scaler)
        trials.append(values)
        if not values['overflow']: break
    if values['overflow']: raise RuntimeError('Smoke exceeded bounded overflow retries')
    report['full_window_trial'] = {'attempts':trials, 'successful_updates':1, 'discard_before_training':True}
    exported = bank.export_other(parent_other)
    comparison = overlay.compare_other_tensors(parent_other, exported)
    if comparison['actual_changed_norm_tensor_count'] == 0: raise RuntimeError('Smoke export has no changed FP16 norms')
    export_dir = args.work_dir/'smoke_export'
    export_dir.mkdir(parents=True, exist_ok=False)
    path = export_dir/'other_fp16.pt'
    torch.save(exported, path)
    report['export_parity'] = export_parity(bank, path, ids)
    report['export_parity'].update({'tensor_comparison':comparison, 'bytes':path.stat().st_size, 'sha256':sha(path)})
    report['export_parity']['archive_byte_delta'] = path.stat().st_size-(args.parent_dir/'other_fp16.pt').stat().st_size
    bank.load_state_dict(initial)
    bank.assert_frozen()
    report['trial_state_discarded'] = True


def train(bank, teacher, windows, parent, parent_other, binding, args, report, frozen_before):
    smoke_report = json.loads(args.smoke_report.read_text())
    if (smoke_report.get('complete') is not True or smoke_report.get('passed') is not True
            or smoke_report.get('binding') != binding
            or smoke_report.get('parent_manifest_sha256') != overlay.PARENT_MANIFEST_SHA256):
        raise ValueError('Successful smoke does not match final inputs and code')
    report['smoke_report_sha256'] = sha(args.smoke_report)
    schedule = [entry['window'] for entry in training_schedule()]
    optimizer, scaler = optimizer_for(bank), scaler_for()
    successes = attempts = overflows = 0
    final_checkpoint = None
    history = []
    if args.resume:
        latest = json.loads((args.work_dir/'latest.json').read_text())
        final_checkpoint = latest
        checkpoint = Path(latest['file'])
        if checkpoint.parent.resolve() != args.work_dir.resolve() or checkpoint.name != Path(checkpoint.name).name:
            raise ValueError('Resume checkpoint is outside this work directory')
        if sha(checkpoint) != latest['sha256']: raise ValueError('Resume checkpoint hash mismatch')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (state['binding'] != binding or state['parent_manifest_sha256'] != overlay.PARENT_MANIFEST_SHA256
                or state['schedule'] != schedule or state['smoke_report_sha256'] != report['smoke_report_sha256']
                or state['frozen_model_sha256'] != frozen_before):
            raise ValueError('Resume provenance differs')
        bank.load_state_dict(state['masters'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        successes, attempts, overflows = state['successful_updates'],state['attempts'],state['overflow_retries']
        if not 0 <= successes <= 128 or not 0 <= overflows <= 8 or attempts != successes+overflows:
            raise ValueError('Invalid resume progress')
        history = state['history']
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(f'[resume] {successes} successful updates, {attempts} attempts', flush=True)
    elif (args.work_dir/'latest.json').exists():
        raise FileExistsError('Existing training checkpoint requires explicit --resume')
    report.update(schedule=schedule, successful_updates=successes, attempts=attempts,
                  overflow_retries=overflows, history=history,
                  successful_target_exposures=successes*2047, attempted_target_exposures=attempts*2047)
    while successes < 128:
        began = time.time()
        window_id = schedule[successes]
        values = attempt_update(bank, teacher, windows[window_id], optimizer, scaler)
        attempts += 1
        if values['overflow']: overflows += 1
        else: successes += 1
        history.append({'attempt':attempts, 'successful_updates_after':successes, 'window':window_id,
                        'seconds':time.time()-began, **values})
        state = {'format':'MAMBA2_NORM_TRAINING_CHECKPOINT_V1','binding':binding,
            'parent_manifest_sha256':overlay.PARENT_MANIFEST_SHA256,'smoke_report_sha256':report['smoke_report_sha256'],
            'schedule':schedule,'successful_updates':successes,'attempts':attempts,'overflow_retries':overflows,
            'history':history,'masters':bank.state_dict(),'optimizer':cpu_tree(optimizer.state_dict()),
            'scaler':scaler.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'frozen_model_sha256':frozen_before}
        path = args.work_dir/f'checkpoint_attempt{attempts:04d}_step{successes:04d}.pt'
        final_checkpoint = save_checkpoint(path, state)
        write_json(args.work_dir/'latest.json', final_checkpoint)
        report.update(successful_updates=successes, attempts=attempts, overflow_retries=overflows,
                      successful_target_exposures=successes*2047, attempted_target_exposures=attempts*2047,
                      final_checkpoint=final_checkpoint, gpu_memory=gpu_memory_receipt())
        write_json(args.report, report)
        print(f'[train] update {successes}/128 attempt {attempts} window {window_id} loss {values["loss"]:.6f} '
              f'scale {values["loss_scale_after"]:g} overflow={values["overflow"]} {time.time()-began:.1f}s',flush=True)
        if overflows > 8 or attempts > 136:
            raise RuntimeError('Fixed overflow/attempt budget exceeded; no candidate export')
    print('[train] verifying all394 frozen GPU parameter values', flush=True)
    frozen_after = parameter_hashes(bank.model, bank.inventory)
    if frozen_after != frozen_before: raise RuntimeError('Frozen GPU parameter contents changed')
    report['frozen_parameter_audit'] = {'tensor_count':len(frozen_after), 'before_sha256':frozen_before,
                                      'after_sha256':frozen_after,'bitwise_unchanged':True}
    # Validate a private staging directory before making a complete candidate
    # visible. Failed or interrupted exports remain available for inspection.
    export_dir = args.work_dir/f'final_export_{time.time_ns()}'
    export_dir.mkdir(parents=True, exist_ok=False)
    other_path = export_dir/'other_fp16.pt'
    exported = bank.export_other(parent_other)
    comparison = overlay.compare_other_tensors(parent_other, exported)
    torch.save(exported, other_path)
    report['archive_byte_delta'] = other_path.stat().st_size-parent['files']['other_fp16.pt']['bytes']
    report['export_parity'] = export_parity(bank, other_path, windows[0,None,:128].cuda())
    report['export_tensor_comparison'] = comparison
    report['replacement_file'] = {'bytes':other_path.stat().st_size,'sha256':sha(other_path)}
    report.update(complete=True, final_step=128, final_checkpoint=final_checkpoint,
                  elapsed_seconds=time.time()-report['started_at_unix'], gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    manifest = {'format':overlay.FORMAT,'complete':True,'parent_manifest_sha256':overlay.PARENT_MANIFEST_SHA256,
        'model_config':MODEL_CONFIG,'binding':binding,'files':{'other_fp16.pt':report['replacement_file']},
        'inherited_files':{name:entry for name,entry in parent['files'].items() if name!='other_fp16.pt'},
        'selected_norms':overlay.selected_norms(),'parameter_mapping':parent['parameter_mapping'],
        'training_receipt':{'final_step':128,'checkpoint_sha256':final_checkpoint['sha256'],
                            'report_sha256':sha(args.report),'smoke_report_sha256':report['smoke_report_sha256']}}
    write_json(export_dir/'manifest.json',manifest)
    _,_,receipt = overlay.verify_overlay(args.parent_dir,export_dir,
        calibration_manifest=args.calibration_dir/'manifest.json',calibration_tokens=args.calibration_dir/'calibration_tokens.pt',
        protocol=args.protocol,training_report=args.report,training_checkpoint=Path(final_checkpoint['file']),smoke_report=args.smoke_report)
    write_json(args.work_dir/'final_overlay_integrity.json',receipt)
    args.overlay_dir.parent.mkdir(parents=True,exist_ok=True)
    export_dir.rename(args.overlay_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','train'),required=True)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'models/source')
    parser.add_argument('--parent-dir',type=Path,default=ROOT/'artifacts/e8w5_v1')
    parser.add_argument('--calibration-dir',type=Path,default=ROOT/'calibration/v1')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/NORM_COMPENSATION_PROTOCOL.md')
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--overlay-dir',type=Path,default=ROOT/'artifacts/norm_compensation_v1')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--smoke-report',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve(); args.report = args.report.resolve()
    if args.mode == 'train' and args.smoke_report is None: parser.error('--train requires --smoke-report')
    if args.mode == 'smoke' and args.resume: parser.error('Smoke cannot resume')
    if args.report.exists() and not args.resume: raise FileExistsError('Preserving existing report; choose a new path')
    if args.overlay_dir.exists() and args.mode == 'train': raise FileExistsError('Preserving existing candidate directory')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    # Keep lock files as history. A resumed job must verify the previous PID is dead.
    lock_path=args.work_dir/'active_job.json'
    if lock_path.exists():
        old=json.loads(lock_path.read_text())
        try: os.kill(old['pid'],0)
        except ProcessLookupError: pass
        else: raise RuntimeError(f'Prior process {old["pid"]} is alive; refusing concurrent work')
        if not args.resume: raise RuntimeError('Existing job record requires explicit resume or a new work directory')
        previous_lock = args.work_dir/f'job_pid{old["pid"]}_{time.time_ns()}.json'
        previous_lock.write_bytes(lock_path.read_bytes())
    if args.report.exists():
        # Keep the interrupted/failed receipt immutable before resume writes.
        previous_report = args.report.with_name(args.report.stem+f'.before_resume_{time.time_ns()}.json')
        previous_report.write_bytes(args.report.read_bytes())
    write_json(lock_path,{'pid':os.getpid(),'mode':args.mode,'started_at_unix':time.time()})
    report={'complete':False,'passed':False if args.mode=='smoke' else None,'mode':args.mode,
        'parent_manifest_sha256':overlay.PARENT_MANIFEST_SHA256,'selected_norms':overlay.selected_norms(),
        'started_at_unix':time.time(),'pid':os.getpid(),'evaluation_data_used':False,'calibration_split':'train'}
    write_json(args.report,report)
    try:
        parent,windows,binding=verify_inputs(args)
        report['binding']=binding
        torch.set_num_threads(8); torch.manual_seed(20260927); torch.cuda.manual_seed_all(20260927)
        torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats()
        report['environment']=environment_receipt()
        write_json(args.report,report)
        print(f'[{args.mode}] verified train-only inputs; PID={os.getpid()}; loading original parent',flush=True)
        student=load_quantized_model(args.parent_dir)
        bank=NormMasters(student)
        if len(dict(student.named_parameters())) != 507 or sum(p.numel() for p in student.parameters()) != 8236999680:
            raise ValueError('Student full parameter coverage differs')
        print(f'[{args.mode}] hashing all394 frozen GPU tensors',flush=True)
        frozen_before=parameter_hashes(student,bank.inventory)
        if len(frozen_before) != 394: raise ValueError('Frozen tensor count differs')
        print(f'[{args.mode}] loading original FP16 teacher',flush=True)
        teacher=load_source_model(args.source_dir)
        if any(p.requires_grad for p in teacher.parameters()): raise ValueError('Teacher not frozen')
        parent_other=torch.load(args.parent_dir/'other_fp16.pt',map_location='cpu',weights_only=True)
        if args.mode=='smoke':
            smoke(bank,teacher,windows,parent_other,args,report)
            frozen_after=parameter_hashes(student,bank.inventory)
            if frozen_after != frozen_before: raise RuntimeError('Frozen GPU weights changed during smoke')
            report['frozen_parameter_audit']={'tensor_count':394,'before_sha256':frozen_before,
                                            'after_sha256':frozen_after,'bitwise_unchanged':True}
            report.update(complete=True,passed=True,elapsed_seconds=time.time()-report['started_at_unix'],
                          gpu_memory=gpu_memory_receipt())
            write_json(args.report,report)
        else:
            train(bank,teacher,windows,parent,parent_other,binding,args,report,frozen_before)
        print(json.dumps({'complete':True,'mode':args.mode,'report':str(args.report),'report_sha256':sha(args.report)}),flush=True)
    except BaseException as error:
        report.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc(),elapsed_seconds=time.time()-report['started_at_unix'])
        if torch.cuda.is_available(): report['gpu_memory']=gpu_memory_receipt()
        write_json(args.report,report)
        raise


if __name__=='__main__':
    main()
