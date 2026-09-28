#!/usr/bin/env python3
"""Fixed all-projection rank-4 CE/KL compensation; smoke first, no resume or selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from mamba_e8w5 import small_overlay
from mamba_e8w5.low_rank_residual import write_factors, read_factors, validate_factors
from mamba_e8w5.norm_training import chunked_loss_backward, tensor_hash
from mamba_e8w5.low_rank_training import (LowRankMasters, selected_inventory, training_schedule,
    gradient_receipt, export_parity)
from scripts.prepare_low_rank_training_data import (BOUND as DATA_UPSTREAM, CODE_BOUND as DATA_CODE,
    previous_intervals, select_starts)
from mamba_e8w5.runtime import (load_quantized_model, load_source_model, MODEL_CONFIG,
    SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256, token_digest, environment_receipt, gpu_memory_receipt)
from scripts.train_norm_compensation import (sha, write_json, save_checkpoint, cpu_tree,
    tree_digest, parameter_hashes, scaler_for, gradient_comparison)

PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
BASELINE_REPORT_SHA = '957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
DATA_SHA = 'e168e1d90ea7cd9c14c8210f13b5e6b569612c82dcaf832100c250cb72d24bef'
TOKENS_SHA = '539684ba450894a2e56d4f958b69942bacd7e6260ef349684c2c965d273012b1'
PROTOCOL_SHA = 'b732c24692ae5dc91c5c67c1a0e0a0c435be3402959bfe5db874c67246836cf2'
FROZEN = {
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/small_overlay.py': '3512c1b232c4039c16a26ffc0fe8c2eac1be6bb9c704ccbcb1d4e8e7788a1286',
    'mamba_e8w5/norm_training.py': '25ac3ec6ba167902014018f99fbf8dbbc659e0e4903f80baf28e28a4a3e74087',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
}
HYPERPARAMETERS = {
    'successful_updates': 1024, 'maximum_overflow_retries': 8, 'maximum_attempts': 1032,
    'epochs': 4, 'windows_per_epoch': 256, 'stored_window_tokens': 2048, 'targets_per_update': 2047,
    'seed': 20260928, 'initialization_seed': 20260928, 'rank': 4,
    'sampler': 'four successive torch.randperm(256) draws from one seeded CPU Generator',
    'optimizer': 'AdamW', 'learning_rate': 3e-4, 'betas': [.9, .999], 'epsilon': 1e-8,
    'weight_decay': 0., 'gradient_clip_norm': 1., 'ce_coefficient': .5,
    'teacher_to_student_kl_coefficient': .5, 'temperature': 1., 'logits_chunk_tokens': 64,
    'masters': '224 FP32 factor tensors; rank4 B=0, A=CPU seeded normal/sqrt(n)',
    'initialization_order': 'numeric layer0..55, in_proj then out_proj; separate CPU generator, only A draws RNG',
    'forward': 'half(W0.detach().float()+B_master.half().float()@A_master.half().float()); casts inside replay',
    'frozen': 'all507 accepted all-small baseline tensors; E8/W5 metadata and indices unchanged',
    'loss_scaler': {'init_scale': 1024., 'growth_factor': 2., 'backoff_factor': .5, 'growth_interval': 2000},
    'overflow_policy': 'retry same window with halved scale, no parameter/optimizer update; abort after8 overflows',
    'selection': 'final1024 successful updates only; no evaluation/checkpoint selection',
    'resume': False, 'checkpoint_every_successful_updates': 32,
}
FACTOR_PAYLOAD_BYTES = 15654912
FACTOR_FILE_BYTES = 15658496
INVENTORY = selected_inventory()

LABELS = tuple(f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj'))
STOP_REQUESTED = False


class RequestedStop(RuntimeError):
    pass


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def file_receipt(path):
    return {'file': str(path), 'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def validate_masters(masters):
    expected = {label+'.'+part: entry[part+'_shape']
                for label,entry in INVENTORY.items() for part in ('B','A')}
    if set(masters) != set(expected):
        raise ValueError('Expected exactly224 factor masters')
    for name,value in masters.items():
        if (not isinstance(value,torch.Tensor) or value.dtype != torch.float32
                or list(value.shape) != expected[name] or not torch.isfinite(value).all()
                or not torch.isfinite(value.half()).all()):
            raise ValueError(f'Invalid FP32/FP16 factor master: {name}')
    return {'tensors':224,'parameters':7827456,'rank':4}


def factor_statistics(factors):
    result = {}
    for label,(B,A) in factors.items():
        result[label] = {part:{'nonzero':int(torch.count_nonzero(value)),
            'minimum':float(value.min()),'maximum':float(value.max())}
            for part,value in (('B',B),('A',A))}
    return result


def accounting(history, schedule):
    """Strict attempt accounting, including repeated windows after overflow."""
    successes = overflows = 0
    if (not isinstance(schedule,list) or len(schedule) != 1024
            or any(type(value) is not int or not 0 <= value < 256 for value in schedule)):
        raise ValueError('Fixed schedule must contain1024 updates')
    for number, row in enumerate(history, 1):
        if (not isinstance(row,dict) or type(row.get('attempt')) is not int or row['attempt'] != number
                or type(row.get('successful_updates_after')) is not int
                or type(row.get('overflow')) is not bool or successes >= 1024
                or type(row.get('schedule_entry')) is not int or row['schedule_entry'] != schedule[successes]
                or type(row.get('targets')) is not int or row['targets'] != 2047):
            raise ValueError('Attempt history does not match fixed schedule')
        if row['overflow']:
            overflows += 1
        else:
            successes += 1
        if row['successful_updates_after'] != successes:
            raise ValueError('Successful-update count differs')
    if overflows > 8 or len(history) > 1032 or len(history) != successes+overflows:
        raise ValueError('Overflow/attempt budget exceeded')
    return {'successful_updates': successes, 'attempts': len(history), 'overflow_retries': overflows,
            'successful_target_exposures': successes*2047, 'attempted_target_exposures': len(history)*2047}


def verify_inputs(args):
    if args.protocol_sha256 != PROTOCOL_SHA:
        raise ValueError('Not the predeclared low-rank protocol')
    fixed_paths = {args.parent_dir/'manifest.json':PARENT_SHA,
        args.small_dir/'manifest.json':SMALL_SHA, args.small_dir/'other_fp16.pt':SMALL_VALUES_SHA,
        ROOT/'reports/small_compensation_v1_eval.json':BASELINE_REPORT_SHA,
        args.protocol:PROTOCOL_SHA}
    fixed_paths.update({ROOT/name:digest for name,digest in {**FROZEN,**DATA_UPSTREAM,**DATA_CODE}.items()})
    for path,digest in fixed_paths.items():
        if sha(path) != digest:
            raise ValueError(f'Pinned input differs: {path}')
    raw = json.loads((args.parent_dir/'manifest.json').read_text())
    small = json.loads((args.small_dir/'manifest.json').read_text())
    baseline = json.loads((ROOT/'reports/small_compensation_v1_eval.json').read_text())
    data_path = args.data_dir/'manifest.json'
    if sha(data_path) != DATA_SHA:
        raise ValueError('Prepared256-window manifest differs')
    data = json.loads(data_path.read_text())
    if (data.get('format') != 'MAMBA2_LOW_RANK_TRAIN_WINDOWS_V1' or data.get('complete') is not True
            or data.get('split') != 'train' or data.get('evaluation_data_used') is not False
            or data.get('nwin') != 256 or data.get('seqlen') != 2048
            or data.get('dataset',{}).get('tokenizer_sha256') != TOKENIZER_SHA256
            or data.get('token_windows_file') != 'training_tokens.pt'
            or data.get('protocol_sha256') != PROTOCOL_SHA
            or data.get('preparation_source_sha256') != sha(ROOT/'scripts/prepare_low_rank_training_data.py')
            or data.get('excluded_manifests_sha256') != DATA_UPSTREAM
            or data.get('schedule') != training_schedule()):
        raise ValueError('Wrong prepared TRAIN data declaration')
    token_path = args.data_dir/'training_tokens.pt'
    if sha(token_path) != data['token_windows_file_sha256'] or sha(token_path) != TOKENS_SHA:
        raise ValueError('Training-token binary identity differs')
    windows = torch.load(token_path,map_location='cpu',weights_only=True)
    if (not isinstance(windows,torch.Tensor) or windows.dtype != torch.int64
            or tuple(windows.shape) != (256,2048) or windows.min() < 0 or windows.max() >= 256000
            or token_digest(windows.flatten().numpy()) != data['selected_tokens_sha256_int64le']):
        raise ValueError('Training-token tensor/content differs')
    documents = [json.loads((ROOT/name).read_text()) for name in DATA_UPSTREAM]
    excluded = previous_intervals(documents)
    starts,overlap = select_starts(excluded)
    if (data['excluded_starts'] != excluded or data['starts'] != starts or data['overlap'] != overlap
            or any(data['dataset'] != doc['dataset'] for doc in documents)):
        raise ValueError('Fresh256-window provenance/selection differs')
    for start,window,receipt in zip(starts,windows,data['windows']):
        if receipt != {'start':start,'stored_tokens':2048,'targets':2047,
                       'token_sha256_int64le':token_digest(window.numpy())}:
            raise ValueError('Per-window training token identity differs')
    if len(data['windows']) != 256:
        raise ValueError('Per-window receipt coverage differs')
    code_paths = ['scripts/train_low_rank_compensation.py','mamba_e8w5/low_rank_training.py',
                  'mamba_e8w5/low_rank_residual.py','scripts/prepare_low_rank_training_data.py']
    binding = {'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'small_values_sha256':SMALL_VALUES_SHA,'source_checkpoint_sha256':SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256':TOKENIZER_SHA256,'baseline_report_sha256':BASELINE_REPORT_SHA,
        'protocol_sha256':PROTOCOL_SHA,'training_data_manifest_sha256':DATA_SHA,
        'training_tokens_sha256':TOKENS_SHA,'hyperparameters':HYPERPARAMETERS,
        'frozen_code_sha256':FROZEN,
        'code_sha256':{name:sha(ROOT/name) for name in code_paths}}
    end_paths = dict(fixed_paths)
    end_paths.update({data_path:DATA_SHA,token_path:TOKENS_SHA})
    end_paths.update({ROOT/name:digest for name,digest in binding['code_sha256'].items()})
    return raw,small,baseline,data,windows,binding,end_paths


def optimizer_for(bank):
    return torch.optim.AdamW(bank.parameters(), lr=3e-4, betas=(.9,.999), eps=1e-8, weight_decay=0.)


def attempt_update(bank, teacher, window, optimizer, scaler):
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    ids, targets = window[None, :-1].cuda(), window[None, 1:].cuda()
    with torch.no_grad():
        teacher_hidden = teacher.backbone(ids)
    hidden = bank.forward(ids, use_checkpoint=True)
    scale = scaler.get_scale()
    values = chunked_loss_backward(hidden, teacher_hidden, targets,
        bank.model.lm_head, teacher.lm_head, scaler, 64)
    del hidden, teacher_hidden
    scaler.unscale_(optimizer)
    gradients = gradient_receipt(bank)
    if gradients['tensors'] != 224 or gradients['parameters'] != 7827456:
        raise RuntimeError('Low-rank gradient coverage differs')
    overflow = not gradients['finite'] or not values['scaled_hidden_gradient_finite']
    if not overflow and gradients['nonzero'] == 0:
        raise RuntimeError('All low_rank gradients are zero')
    if overflow:
        masters_before, optimizer_before = tree_digest(bank.state_dict()), tree_digest(optimizer.state_dict())
        scaler.update(new_scale=scale*.5)
        if masters_before != tree_digest(bank.state_dict()) or optimizer_before != tree_digest(optimizer.state_dict()):
            raise RuntimeError('Overflow changed masters or optimizer')
        magnitude = None
    else:
        magnitude = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), 1., error_if_nonfinite=True))
        scaler.step(optimizer)
        scaler.update()
    validate_masters(bank.masters)
    bank.assert_frozen()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {**values, 'overflow': overflow, 'loss_scale_before': scale, 'loss_scale_after': scaler.get_scale(),
        'gradient_norm_before_clip': magnitude, 'gradients': gradients,
        'attempt_seconds':time.perf_counter()-started,'gpu_memory':gpu_memory_receipt(),
        'overflow_optimizer_and_masters_unchanged': True if overflow else None}


def serialize_factors(bank, directory):
    directory.mkdir(parents=True)
    factors = bank.export_factors()
    files,loaded = {},{}
    for label in LABELS:
        name = label+'.lrf'
        B,A = factors[label]
        files[name] = write_factors(directory/name,B,A)
        loaded[label] = read_factors(directory/name,expected_shape=INVENTORY[label]['shape'],expected_rank=4)
        if any(tensor_hash(a) != tensor_hash(b) for a,b in zip(loaded[label],(B,A))):
            raise RuntimeError('Exported factors differ from rounded masters')
    if sum(v['bytes'] for v in files.values()) != FACTOR_FILE_BYTES:
        raise RuntimeError('Unexpected factor file capacity')
    return loaded,files


def replay_smoke(bank, ids):
    """Compare one real block at a time; never retain56 decoded backward graphs."""
    chosen = (0, 18, 55)
    captures, handles = {}, []
    for index in chosen:
        def capture(_module, args, index=index):
            captures[index] = (args[0].detach().clone(),
                               None if args[1] is None else args[1].detach().clone())
        handles.append(bank.model.backbone.layers[index].register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            bank.model.backbone(ids)
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(chosen):
        raise RuntimeError('Native block-input captures incomplete')
    initial = bank.state_dict()
    with torch.no_grad():
        for master in bank.parameters():
            master.add_(torch.linspace(-.002,.002,master.numel(),device=master.device).reshape_as(master))
    results = {}
    try:
        for index in chosen:
            pairs, output_hashes = [], []
            for enabled in (False, True):
                hidden = captures[index][0].detach().clone().requires_grad_(True)
                residual = captures[index][1]
                residual = None if residual is None else residual.detach().clone().requires_grad_(True)
                output, new_residual = bank.forward_block(index, hidden, residual, use_checkpoint=enabled)
                tangent = torch.linspace(-1,1,output.shape[-1],device=output.device)
                objective = 1024*((output.float()*tangent).mean()+.1*(new_residual.float()*tangent).mean())
                factor_names = [f'layer{index}.{part}.{family}' for part in ('in_proj','out_proj') for family in ('B','A')]
                names = factor_names+['hidden']
                parameters = [bank.masters[n] for n in factor_names]+[hidden]
                if residual is not None:
                    names.append('residual'); parameters.append(residual)
                gradients = torch.autograd.grad(objective, parameters)
                values = {name:value.detach().cpu().clone() for name,value in zip(names, gradients)}
                if any(not torch.isfinite(v).all() for v in values.values()):
                    raise FloatingPointError('Nonfinite block smoke gradient')
                if any(torch.count_nonzero(values[n]) == 0 for n in factor_names):
                    raise RuntimeError('Missing nonzero low_rank block gradient')
                pairs.append(values)
                output_hashes.append({'hidden':tensor_hash(output),'residual':tensor_hash(new_residual)})
                del output,new_residual,objective,gradients,hidden,residual,parameters
            if any(tensor_hash(pairs[0][n]) != tensor_hash(pairs[1][n]) for n in pairs[0]):
                raise RuntimeError('Checkpoint replay changes gradient bits')
            if output_hashes[0] != output_hashes[1]:
                raise RuntimeError('Checkpoint replay changes block forward')
            results[str(index)] = {'forward_bitwise_equal':True,
                'gradients':gradient_comparison(pairs[0],pairs[1],
                    rtol=0.,atol=0.)}
            del pairs
    finally:
        bank.load_state_dict(initial)
        for parameter in bank.parameters():
            parameter.grad = None
    bank.assert_frozen()
    return {'all224_factors_perturbed_before_test':True,'blocks':results,
        'scope':'Three actual blocks independently,128 tokens; full56-block uncheckpointed graph is not retained.'}


def smoke(bank, teacher, windows, args, report):
    ids = windows[0,None,:128].cuda()
    initial = bank.state_dict()
    with torch.no_grad():
        native, functional = bank.model.backbone(ids), bank.forward(ids,use_checkpoint=False)
        if not torch.isfinite(native).all() or not torch.isfinite(functional).all():
            raise FloatingPointError('Nonfinite zero-B hidden state')
        if tensor_hash(native) != tensor_hash(functional):
            raise RuntimeError('Zero-B native hidden mismatch')
        logits_native, logits_functional = bank.model.lm_head(native[:,-8:]), bank.model.lm_head(functional[:,-8:])
        if not torch.isfinite(logits_native).all() or not torch.isfinite(logits_functional).all():
            raise FloatingPointError('Nonfinite zero-B logits')
        if tensor_hash(logits_native) != tensor_hash(logits_functional):
            raise RuntimeError('Zero-B native logit mismatch')
    report['zero_native_parity'] = {'hidden_bitwise_equal':True,'last8_full_vocab_logits_bitwise_equal':True,
        'input_tokens':128,'hidden_sha256':tensor_hash(native),'logits_sha256':tensor_hash(logits_native)}
    del native,functional,logits_native,logits_functional
    write_json(args.report,report)
    print('[low_rank smoke] Local block replay gradients',flush=True)
    report['checkpoint_parity'] = replay_smoke(bank,ids)
    write_json(args.report,report)
    optimizer,scaler = optimizer_for(bank),scaler_for()
    trials = []
    print('[low_rank smoke] Full2047-target CE/KL trial',flush=True)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(9):
        values = attempt_update(bank,teacher,windows[0],optimizer,scaler)
        trials.append(values)
        report['full_window_trial'] = {'attempts':trials,'successful_updates':0 if values['overflow'] else 1}
        write_json(args.report,report)
        if not values['overflow']:
            break
    if values['overflow']:
        raise RuntimeError('Smoke exceeds8 overflow retries')
    initial_grads = values['gradients']['per_tensor']
    if any(not v['finite'] or (v['nonzero'] == 0 if n.endswith('.B') else v['nonzero'] != 0)
           for n,v in initial_grads.items()):
        raise RuntimeError('First update must have all112 nonzero B gradients and all112 zero A gradients')
    report['full_window_trial'].update(successful_target_exposures=2047,
        attempted_target_exposures=2047*len(trials),overflow_retries=len(trials)-1,
        full_update_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        full_update_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        full_update_peak_scope='Counter reset immediately before full-window trial; includes resident teacher and baseline.')
    print('[low_rank smoke] Post-update128-token gradient check',flush=True)
    optimizer.zero_grad(set_to_none=True)
    post_ids,post_targets = windows[0,None,:128].cuda(),windows[0,None,1:129].cuda()
    with torch.no_grad():
        teacher_hidden = teacher.backbone(post_ids)
    hidden = bank.forward(post_ids,use_checkpoint=True)
    losses = chunked_loss_backward(hidden,teacher_hidden,post_targets,
        bank.model.lm_head,teacher.lm_head,scaler,64)
    scaler.unscale_(optimizer)
    post_gradients = gradient_receipt(bank)
    if (not losses['scaled_hidden_gradient_finite'] or
            any(not v['finite'] or v['nonzero'] == 0 for v in post_gradients['per_tensor'].values())):
        raise RuntimeError('Post-update backward requires all224 nonzero finite factor gradients')
    report['post_update_short_backward'] = {'input_tokens':128,'losses':losses,'gradients':post_gradients,
                                           'optimizer_step_performed':False}
    optimizer.zero_grad(set_to_none=True)
    del hidden,teacher_hidden,post_ids,post_targets
    factors,files = serialize_factors(bank,args.work_dir/'smoke_export')
    changed = sum(bool(torch.count_nonzero(pair[0])) for pair in factors.values())
    if changed != 112:
        raise RuntimeError('Smoke update must produce nonzero FP16 B for all112 projections')
    report['export_parity'] = export_parity(bank,factors,ids)
    report['export_parity'].update(files=files,changed_B_files=changed,
        factor_payload_bytes=FACTOR_PAYLOAD_BYTES,factor_file_bytes=sum(v['bytes'] for v in files.values()),
        factor_statistics=factor_statistics(factors))
    bank.load_state_dict(initial)
    optimizer.zero_grad(set_to_none=True)
    if tree_digest(bank.state_dict()) != tree_digest(initial):
        raise RuntimeError('Smoke trial state restoration differs')
    bank.assert_frozen()
    report['trial_state_discarded'] = True


class TrainingRun:
    def __init__(self,bank,teacher,windows,binding,args,report,frozen_before):
        self.bank,self.teacher,self.windows = bank,teacher,windows
        self.binding,self.args,self.report,self.frozen_before = binding,args,report,frozen_before
        self.optimizer,self.scaler = optimizer_for(bank),scaler_for()
        self.schedule = training_schedule()
        self.history = []
        self.successes = self.attempts = self.overflows = 0
        self.final_checkpoint = None

    def journal(self,event):
        with (self.args.work_dir/'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**event},allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())

    def checkpoint(self,reason):
        state = {'format':'MAMBA2_LOW_RANK_TRAINING_CHECKPOINT_V1','binding':self.binding,
            'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
            'smoke_report_sha256':self.report['smoke_report_sha256'],'schedule':self.schedule,
            'successful_updates':self.successes,'attempts':self.attempts,'overflow_retries':self.overflows,
            'history':self.history,'masters':self.bank.state_dict(),'optimizer':cpu_tree(self.optimizer.state_dict()),
            'scaler':self.scaler.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'frozen_model_sha256':self.frozen_before,'resumable':False,'reason':reason}
        path = self.args.work_dir/f'checkpoint_attempt{self.attempts:03d}_step{self.successes:03d}_{reason}.pt'
        receipt = save_checkpoint(path,state)
        if reason == 'final':
            self.final_checkpoint = receipt
        return receipt

    def run(self):
        self.report.update(schedule=self.schedule,history=self.history,
            successful_updates=0,attempts=0,overflow_retries=0,successful_target_exposures=0,
            attempted_target_exposures=0,resumption_policy='No resume or replay; fresh paths required')
        try:
            while self.successes < 1024:
                if STOP_REQUESTED:
                    self.checkpoint('termination')
                    raise RequestedStop('Stop requested between updates; no resume is authorized')
                item = self.schedule[self.successes]
                self.journal({'event':'attempt_start','attempt':self.attempts+1,
                    'successful_updates_before':self.successes,'schedule_entry':item})
                t0 = time.perf_counter()
                values = attempt_update(self.bank,self.teacher,self.windows[item],self.optimizer,self.scaler)
                self.attempts += 1
                if values['overflow']:
                    self.overflows += 1
                else:
                    self.successes += 1
                row = {'attempt':self.attempts,'successful_updates_after':self.successes,
                    'schedule_entry':item,'seconds':time.perf_counter()-t0,**values}
                self.history.append(row)
                self.journal({'event':'attempt_complete',**row})
                counts = accounting(self.history,self.schedule)
                if not values['overflow'] and self.successes % 32 == 0:
                    cp = self.checkpoint('final' if self.successes == 1024 else 'periodic')
                    self.report.setdefault('checkpoints',[]).append(cp)
                self.report.update(**counts,gpu_memory=gpu_memory_receipt(),final_checkpoint=self.final_checkpoint)
                write_json(self.args.report,self.report)
                if self.successes % 8 == 0 or values['overflow']:
                    print(f'[low_rank train] {self.successes}/1024 attempt{self.attempts} '
                          f'loss={values["loss"]:.6f} scale={values["loss_scale_after"]:g} overflow={values["overflow"]}',flush=True)
            return self.final_checkpoint
        except RequestedStop:
            raise
        except BaseException:
            self.report['failure_snapshot'] = self.checkpoint('failure')
            raise


def finalize(run,args,report,raw,end_paths):
    counts = accounting(run.history,run.schedule)
    if counts['successful_updates'] != 1024 or not run.final_checkpoint:
        raise RuntimeError('Final1024 checkpoint is required')
    frozen_after = parameter_hashes(run.bank.model)
    if frozen_after != run.frozen_before:
        raise RuntimeError('Frozen baseline tensors changed')
    stage = args.work_dir/'final_export'
    factors,files = serialize_factors(run.bank,stage)
    if sha(run.final_checkpoint['file']) != run.final_checkpoint['sha256']:
        raise RuntimeError('Final saved checkpoint hash differs')
    checkpoint = torch.load(run.final_checkpoint['file'],map_location='cpu',weights_only=True)
    if (checkpoint['binding'] != run.binding
            or type(checkpoint['successful_updates']) is not int or checkpoint['successful_updates'] != 1024
            or any(type(checkpoint[key]) is not int or checkpoint[key] != counts[key]
                   for key in ('attempts','overflow_retries'))
            or checkpoint.get('reason') != 'final' or checkpoint.get('resumable') is not False
            or checkpoint['history'] != run.history or checkpoint['schedule'] != run.schedule
            or checkpoint['frozen_model_sha256'] != frozen_after):
        raise RuntimeError('Final saved checkpoint identity/accounting differs')
    validate_masters(checkpoint['masters'])
    for name, master in checkpoint['masters'].items():
        label,part = name.rsplit('.',1)
        if tensor_hash(master.half()) != tensor_hash(factors[label][0 if part == 'B' else 1]):
            raise RuntimeError('Final checkpoint rounding differs from exported factors')
    report['export_parity'] = export_parity(run.bank,factors,run.windows[0,None,:128].cuda())
    if parameter_hashes(run.bank.model) != frozen_after:
        raise RuntimeError('Export parity modified baseline tensors')
    for path,digest in end_paths.items():
        if sha(path) != digest:
            raise RuntimeError(f'Bound input changed during training: {path}')
    report['frozen_parameter_audit'] = {'tensor_count':507,'before_sha256':run.frozen_before,
        'after_sha256':frozen_after,'bitwise_unchanged':True}
    report['final_export'] = {'files':files,'changed_B_files':sum(bool(torch.count_nonzero(pair[0])) for pair in factors.values()),'factor_statistics':factor_statistics(factors),
        'factor_payload_bytes':15654912,'factor_file_bytes':sum(x['bytes'] for x in files.values()),
        'checkpoint1024_master_to_fp16_bitwise_equal':True,'checkpoint_sha256':run.final_checkpoint['sha256']}
    report.update(complete=True,passed=None,final_step=1024,final_checkpoint=run.final_checkpoint,
        elapsed_seconds=time.time()-report['started_at_unix'],gpu_memory=gpu_memory_receipt(),**counts)
    write_json(args.report,report)
    manifest = {'format':'MAMBA2_LOW_RANK_COMPENSATION_V1','complete':True,
        'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'small_values_sha256':SMALL_VALUES_SHA,'model_config':MODEL_CONFIG,'binding':run.binding,
        'files':files,'factors_inventory':{name:{'file':name+'.lrf',**entry} for name,entry in INVENTORY.items()},
        'frozen_base_hash_ledger':frozen_after,
        'training_receipt':{'successful_updates':1024,'attempts':run.attempts,'overflow_retries':run.overflows,
            'checkpoint_sha256':run.final_checkpoint['sha256'],'report_sha256':sha(args.report),
            'smoke_report_sha256':report['smoke_report_sha256']},
        'storage':{'base_logical_data_bytes':raw['data_file_bytes'],'factor_payload_bytes':15654912,
            'factor_file_bytes':FACTOR_FILE_BYTES,'logical_inference_data_bytes':raw['data_file_bytes']+FACTOR_FILE_BYTES,
            'manifest_bytes':0,'scope':'Original E8/W5 plus full393-small replacement plus112 rank4 factor files; tokenizer/software/reports/training files excluded.'},
        'representation':'All112 rank4 FP16 factors; effective weights use half(W0.float()+B16.float()@A16.float()); all-small base frozen.'}
    while True:
        write_json(stage/'manifest.json',manifest)
        size = (stage/'manifest.json').stat().st_size
        if manifest['storage']['manifest_bytes'] == size:
            break
        manifest['storage']['manifest_bytes'] = size
    args.out_dir.parent.mkdir(parents=True,exist_ok=True)
    if args.out_dir.exists():
        raise FileExistsError('Refusing to replace an existing candidate')
    stage.rename(args.out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','train'),required=True)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'models/source')
    parser.add_argument('--parent-dir',type=Path,default=ROOT/'artifacts/e8w5_v1')
    parser.add_argument('--small-dir',type=Path,default=ROOT/'artifacts/small_compensation_v1')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'training_data/low_rank_compensation_v1')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/LOW_RANK_COMPENSATION_PROTOCOL.md')
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--smoke-report',type=Path)
    args = parser.parse_args()
    if args.mode == 'train' and args.smoke_report is None:
        parser.error('Training requires --smoke-report from matching successful smoke')
    for field in ('source_dir','parent_dir','small_dir','data_dir','protocol','work_dir','out_dir','report'):
        setattr(args,field,getattr(args,field).resolve())
    if any(path.exists() for path in (args.work_dir,args.out_dir,args.report)):
        raise FileExistsError('Existing work/output/report cannot be resumed or overwritten')
    outputs = (args.work_dir,args.out_dir,args.report)
    if any(a == b or a in b.parents or b in a.parents
           for i,a in enumerate(outputs) for b in outputs[i+1:]):
        raise ValueError('Work/output/report paths must be distinct and must not contain one another')
    for output in (args.work_dir,args.out_dir,args.report):
        if any(output == source or source in output.parents for source in (args.source_dir,args.parent_dir,args.small_dir,args.data_dir)):
            raise ValueError('Outputs must not be inside immutable input directories')
    args.work_dir.mkdir(parents=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.work_dir/'job.json',{'pid':os.getpid(),'mode':args.mode,'started_at_unix':time.time(),'resume':False})
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    report = {'format':'MAMBA2_LOW_RANK_COMPENSATION_TRAIN_V1','complete':False,
        'passed':False if args.mode == 'smoke' else None,'mode':args.mode,'pid':os.getpid(),
        'started_at_unix':time.time(),'evaluation_data_used':False,'split':'train','selected_projections':112,'factor_master_tensors':224,'rank':4,
        'trainable_parameters':7827456,'frozen_baseline_tensors':507}
    write_json(args.report,report)
    try:
        raw,small,baseline,data,windows,binding,end_paths = verify_inputs(args)
        report.update(binding=binding,data_manifest=data)
        if args.mode == 'train':
            smoke_report = json.loads(args.smoke_report.read_text())
            if (smoke_report.get('complete') is not True or smoke_report.get('passed') is not True
                    or smoke_report.get('mode') != 'smoke' or smoke_report.get('binding') != binding
                    or smoke_report.get('trial_state_discarded') is not True):
                raise ValueError('Successful smoke binding differs')
            report['smoke_report_sha256'] = sha(args.smoke_report)
            end_paths[args.smoke_report] = report['smoke_report_sha256']
        torch.set_num_threads(8);torch.manual_seed(20260928);torch.cuda.manual_seed_all(20260928)
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        report['environment'] = environment_receipt();write_json(args.report,report)
        print(f'[low_rank {args.mode}] Loading verified baseline; PID={os.getpid()}',flush=True)
        model = load_quantized_model(args.parent_dir)
        apply = small_overlay.apply_overlay(model,args.small_dir,small)
        if apply is not model:
            raise RuntimeError('Unexpected baseline overlay return')
        bank = LowRankMasters(model)
        validate_masters(bank.masters)
        if bank.inventory != INVENTORY or any(bool(torch.count_nonzero(v)) for n,v in bank.masters.items() if n.endswith('.B')):
            raise ValueError('Factor geometry or zero-B initialization differs')
        report['initial_master_sha256'] = {n:tensor_hash(v) for n,v in bank.masters.items()}
        report['initial_factor_statistics'] = factor_statistics(bank.export_factors())
        frozen_before = parameter_hashes(model)
        expected = {e['name']:e['decoded_fp16_sha256'] for e in baseline['parent_loaded_tensor_audit']['tensors']}
        expected.update(baseline['small_loaded_tensor_audit']['small']['loaded_and_export_fp16_sha256'])
        if len(frozen_before) != 507 or frozen_before != expected:
            raise RuntimeError('Actual baseline differs from pinned best model')
        report['baseline_gpu_sha256'] = frozen_before
        zero_hashes = {}
        with torch.no_grad():
            for label in LABELS:
                merged = bank.merge(label)
                base = bank.base_weights[label]
                changed = merged.view(torch.int16) != base.view(torch.int16)
                if not torch.equal(merged,base) or torch.any(changed & ((merged != 0) | (base != 0))):
                    raise RuntimeError(f'Zero-B merge changes a base value: {label}')
                zero_hashes[label] = {'merged_sha256':tensor_hash(merged),
                    'base_sha256':frozen_before[INVENTORY[label]['source_key']],
                    'signed_zero_only_bit_changes':int(torch.count_nonzero(changed))}
                del merged,changed
        report['zero_merged_projection_audit'] = {'verified':112,'all_values_equal':True,'per_projection':zero_hashes}
        write_json(args.report,report)
        teacher = load_source_model(args.source_dir)
        if args.mode == 'smoke':
            smoke(bank,teacher,windows,args,report)
            frozen_after = parameter_hashes(model)
            if frozen_after != frozen_before:
                raise RuntimeError('Smoke changed frozen baseline values')
            for path,digest in end_paths.items():
                if sha(path) != digest:
                    raise RuntimeError(f'Bound smoke input changed: {path}')
            report['frozen_parameter_audit'] = {'tensor_count':507,'before_sha256':frozen_before,
                'after_sha256':frozen_after,'bitwise_unchanged':True}
            report.update(complete=True,passed=True,elapsed_seconds=time.time()-report['started_at_unix'],gpu_memory=gpu_memory_receipt())
            write_json(args.report,report)
        else:
            run = TrainingRun(bank,teacher,windows,binding,args,report,frozen_before)
            run.run()
            finalize(run,args,report,raw,end_paths)
        print(json.dumps({'complete':True,'mode':args.mode,'report':str(args.report),'sha256':sha(args.report)}),flush=True)
    except BaseException as error:
        report.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),elapsed_seconds=time.time()-report['started_at_unix'])
        if torch.cuda.is_available():
            report['gpu_memory'] = gpu_memory_receipt()
        write_json(args.report,report)
        raise


if __name__ == '__main__':
    main()
