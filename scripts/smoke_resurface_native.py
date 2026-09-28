#!/usr/bin/env python3
"""Once-only native8B adapter correctness smoke; all trial updates are discarded."""
from __future__ import annotations
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from mamba_e8w5 import resurface_native as native
from mamba_e8w5.resurface_loss import staged_loss_backward
from mamba_e8w5.runtime import make_cache, environment_receipt, gpu_memory_receipt
from train_norm_compensation import gradient_comparison, SMOKE_TOLERANCES

PROTOCOL_SHA = '469063147d892282a2bdad8558afca8466d6f365f1efe14ecbba7417e9857dd6'
DATA_SHA = 'dfe076ef18d5b0610e016d41d00a38948f3878b7c67d9cd8f6f46237e09d7e11'
TRAINER_SHA = 'ddc9aacf26b37d318b20e9409f8c7619a6bd8d3d1688f7736e2acab6ae12ec0b'
CODE_PINS = {
    'mamba_e8w5/resurface_native.py': '2dd08c7ee8958c832f0ae9da7cf5f261dceeff3e528e493c905c7c52df7166d7',
    'mamba_e8w5/resurface_loss.py': 'ec045368583b6e49f484341b2d718658bb59e8d11050710377742752c1c67294',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
}


def write(path, report):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_inputs(args):
    pins = {**CODE_PINS, 'scripts/train_resurface_readapted.py': TRAINER_SHA,
            'docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md': PROTOCOL_SHA}
    for name, digest in pins.items():
        if native.file_hash(ROOT/name) != digest:
            raise ValueError(f'Smoke source/protocol differs: {name}')
    if args.train_manifest_sha256 != DATA_SHA:
        raise ValueError('Use exactly the fixed TRAIN manifest')
    trainer = importlib.import_module('train_resurface_readapted')
    inputs = trainer.verify_inputs(args)
    return trainer, inputs, pins


def parameter_gradients(bank):
    values, receipt, families = {}, {}, {}
    for name, p in bank.masters.items():
        grad = p.grad
        if grad is None or grad.dtype != torch.float32 or not torch.isfinite(grad).all():
            raise RuntimeError(f'Missing/nonfinite FP32 adapter gradient: {name}')
        values[name] = grad.detach().cpu().clone()
        receipt[name] = {'shape': list(grad.shape), 'sha256': native.tensor_hash(grad),
            'nonzero': int(torch.count_nonzero(grad)), 'minimum': float(grad.min()),
            'maximum': float(grad.max()), 'maximum_absolute': float(grad.abs().max())}
        family = name.rsplit('.', 1)[1]
        families[family] = families.get(family, 0)+receipt[name]['nonzero']
    bank.assert_base_frozen()
    return values, {'tensors': len(values), 'parameters': sum(v.numel() for v in values.values()),
                    'per_tensor': receipt, 'family_nonzero_elements': families, 'finite': True}


def clear_gradients(bank):
    for p in bank.parameters(): p.grad = None


@torch.no_grad()
def native_probe(model, ids):
    """128-token prefill plus3 forced native decode steps; same inputs in each arm."""
    if ids.shape != (1, 131): raise ValueError('Fixed131-token smoke input required')
    hidden = model.backbone(ids[:, :128])
    logits = model.lm_head(hidden[:, -8:])
    if not torch.isfinite(hidden).all() or not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite native probe')
    result = {'hidden_sha256': native.tensor_hash(hidden), 'last8_logits_sha256': native.tensor_hash(logits),
              'input_tokens': 128, 'steps': []}
    cache = make_cache(model, 131, state_dtype=torch.float16)
    for step in range(4):
        cache.seqlen_offset = 0 if step == 0 else 127+step
        tokens = ids[:, :128] if step == 0 else ids[:, 127+step:128+step]
        h = model.backbone(tokens, inference_params=cache)
        z = model.lm_head(h[:, -1:])
        tensors = {f'layer{index}.{kind}': value for index, pair in cache.key_value_memory_dict.items()
                   for kind, value in zip(('conv', 'ssm'), pair)}
        if len(tensors) != 112 or any(v.dtype != torch.float16 for v in tensors.values()):
            raise RuntimeError('Native cache inventory differs')
        if not torch.isfinite(h).all() or not torch.isfinite(z).all() or not all(torch.isfinite(v).all() for v in tensors.values()):
            raise FloatingPointError('Nonfinite hidden/logits/cache')
        result['steps'].append({'offset': cache.seqlen_offset, 'hidden_sha256': native.tensor_hash(h),
            'logits_sha256': native.tensor_hash(z), 'argmax': int(z.argmax(-1)),
            'cache_sha256': {key: native.tensor_hash(value) for key, value in tensors.items()},
            'cache_bytes': sum(v.numel()*v.element_size() for v in tensors.values())})
    return result


def loss_parity(bank, teacher, pair):
    """Small actual-vocabulary objectives; direct oracle has no staged backward."""
    result = {}
    for kind in ('mk', 'prose'):
        ids = pair[f'{kind}_ids'].cuda()
        targets = pair[f'{kind}_targets'].cuda()
        if kind == 'prose': ids, targets = ids[:, :5], targets[:, :5]
        with torch.no_grad():
            h, gates = bank.forward_hidden(ids, use_checkpoint=False)
            th = teacher.backbone(ids) if kind == 'prose' else None
        selected = pair['answer_mask'].cuda() if kind == 'mk' else torch.ones_like(targets, dtype=torch.bool)
        if kind == 'mk' and int(selected.sum()) != 7:
            raise ValueError('Actual first TRAIN answer must supervise all7 suffix tokens')
        gradients, objectives, metrics = [], [], None
        for staged in (False, True):
            hidden = h.detach().clone().requires_grad_()
            logits = gates.logits.detach().clone().requires_grad_()
            capture = native.GateCapture(logits)
            if staged:
                metrics = staged_loss_backward(hidden, targets, bank.model.lm_head, gates=capture,
                    teacher_hidden=th, teacher_head=teacher.lm_head if kind == 'prose' else None,
                    answer_mask=selected, ce_weight=.5 if kind == 'prose' else 1.,
                    kl_weight=.5 if kind == 'prose' else 0., closure_weight=3. if kind == 'prose' else 0.,
                    chunk_tokens=3)
                objectives.append(metrics['loss'])
            else:
                z = bank.model.lm_head(hidden[selected]).float()
                loss = F.cross_entropy(z, targets[selected], reduction='sum')/selected.sum()
                if kind == 'prose':
                    with torch.no_grad(): tlp = F.log_softmax(teacher.lm_head(th[selected]).float(), -1)
                    kl = F.kl_div(F.log_softmax(z, -1), tlp, reduction='sum', log_target=True)/selected.sum()
                    loss = .5*loss+.5*kl+3*capture.closure_loss()
                loss.backward(); objectives.append(float(loss.detach()))
            gradients.append({'hidden': hidden.grad.detach().cpu(),
                              **({'gate_logits': logits.grad.detach().cpu()} if kind == 'prose' else {})})
        delta = abs(objectives[0]-objectives[1])
        if delta > SMOKE_TOLERANCES['chunk_objective_absolute']:
            raise RuntimeError(f'{kind} objective parity failed')
        comparison = gradient_comparison(*gradients, rtol=SMOKE_TOLERANCES['chunk_hidden_gradient_rtol'],
                                         atol=SMOKE_TOLERANCES['chunk_hidden_gradient_atol'])
        result[kind] = {**comparison, 'objective_absolute_difference': delta, 'direct_objective': objectives[0],
            'staged_metrics': metrics, 'vocabulary': 256000, 'actual_training_labels': True}
    return result


def run(args, report):
    trainer, inputs, pins = verify_inputs(args)
    report.update(binding=inputs['binding'], code_sha256=pins, base_integrity=inputs['base_integrity'],
                  tolerances=SMOKE_TOLERANCES)
    if args.cpu_preflight:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise ValueError('CPU preflight requires CUDA hidden/uninitialized')
        report.update(complete=True, mode='cpu_preflight', cuda_initialized=False)
        trainer.final_recheck(inputs)
        return
    args.work_dir.mkdir()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report['environment'] = environment_receipt()
    student = teacher = bank = None
    initial = None
    expected = inputs['base_expected507']
    shared = importlib.import_module('mamba_e8w5.axis_small_base')
    try:
        student, teacher, bank, audits = trainer.load_models_and_bank(inputs)
        report['load_audits'] = audits
        report['initial_student507'] = shared.audit_model(student, expected)
        report['initial_teacher507'] = shared.audit_model(teacher, expected)
        bank.close()  # Baseline is this same model without any adapter hooks.
        probe_ids = inputs['prose_windows'][0, None, :131].cuda()
        baseline = native_probe(student, probe_ids)
        bank = native.ResurfaceNative(student, gate_mode='soft', expected_base_hashes=expected)
        initial = bank.state_dict()
        zero = native_probe(student, probe_ids)
        report['zero_adapter'] = {'native_without_adapter': baseline, 'native_zero_adapter': zero,
                                  'exact': baseline == zero}
        write(args.report, report)
        if baseline != zero: raise RuntimeError('Zero adapter changed native hidden/logits/FP16 caches')
        pair = trainer.make_pair(inputs, 0)
        clear_gradients(bank)
        h, gates = bank.forward_hidden(pair['mk_ids'].cuda(), use_checkpoint=True)
        class Scale1024:
            def scale(self, value): return value*1024
        mk = staged_loss_backward(h, pair['mk_targets'].cuda(), student.lm_head,
                                  gates=gates, answer_mask=pair['answer_mask'].cuda(), scaler=Scale1024())
        _, zero_grads = parameter_gradients(bank)
        if any(bool(row['nonzero']) != name.endswith('.V_read') for name, row in zero_grads['per_tensor'].items()):
            raise RuntimeError('Soft zero-V answer gradients violate staggered pattern')
        report['zero_V_gradients'] = {'loss': mk, 'gradients': zero_grads,
                                     'expected': '56 V tensors nonzero; router/g tensors exactly zero'}
        with torch.no_grad():
            for name, p in bank.masters.items():
                p.add_(torch.linspace(-.003, .003, p.numel(), device=p.device).reshape(p.shape))
        forward_hashes, gradients, grad_reports = [], [], []
        for enabled in (False, True):
            clear_gradients(bank)
            hidden, gates = bank.forward_hidden(probe_ids[:, :128], use_checkpoint=enabled)
            forward_hashes.append({'hidden': native.tensor_hash(hidden), 'gates': native.tensor_hash(gates.logits)})
            tangent = torch.linspace(-1, 1, hidden.shape[-1], device=hidden.device)
            (1024*((hidden.float()*tangent).mean()+.01*gates.closure_loss())).backward()
            values, receipt = parameter_gradients(bank)
            if not all(receipt['family_nonzero_elements'].values()): raise RuntimeError('A perturbed family has zero gradients')
            gradients.append(values); grad_reports.append(receipt)
        report['checkpoint_parity'] = {'forward': forward_hashes, 'gradients': grad_reports,
            'gradient_comparison': gradient_comparison(*gradients,
                rtol=SMOKE_TOLERANCES['checkpoint_gradient_rtol'], atol=SMOKE_TOLERANCES['checkpoint_gradient_atol'])}
        if forward_hashes[0] != forward_hashes[1]: raise RuntimeError('Checkpoint hidden/gate output differs')
        with torch.no_grad():
            h, _ = bank.forward_hidden(probe_ids[:, :128], use_checkpoint=False)
            direct = student.backbone(probe_ids[:, :128])
            if native.tensor_hash(h) != native.tensor_hash(direct): raise RuntimeError('Native/backbone orchestration differs')
        report['checkpoint_parity']['native_hidden_exact'] = True
        del gradients, hidden, h, direct
        report['full_vocabulary_loss_parity'] = loss_parity(bank, teacher, pair)
        write(args.report, report)
        bank.load_state_dict(initial); clear_gradients(bank)
        optimizer, scaler = trainer.optimizer_for(bank), trainer.scaler_for()
        trials = []
        for _ in range(9):
            trial = trainer.attempt_update(bank, teacher, inputs, 0, optimizer, scaler)
            trials.append(trial)
            report['trial_update'] = {'attempts': trials, 'discard_before_training': True}
            write(args.report, report)
            if not trial['overflow']: break
        if trials[-1]['overflow']: raise RuntimeError('Smoke exhausted8 overflow retries')
        report['trial_update']['successful_updates'] = 1
        clear_gradients(bank)
        path = args.work_dir/'adapter_fp16.pt'
        report['export'] = bank.export_fp16(path, binding=inputs['binding'])
        if report['export']['parameters'] != 1154104 or report['export']['payload_bytes'] != 2308208:
            raise RuntimeError('Adapter export capacity differs')
        before = native_probe(student, probe_ids)
        bank.close()
        with native.install_fp16(student, path, expected_binding=inputs['binding'], expected_base_hashes=expected) as deployed:
            after = native_probe(student, probe_ids)
            deployed.assert_base_frozen(check_values=True)
        report['export_parity'] = {'before_export': before, 'independent_native_reload': after, 'exact': before == after}
        if before != after: raise RuntimeError('Serialized FP16 native prefill/step/cache parity failed')
        bank.load_state_dict(initial)
        report['trial_state_discarded'] = all(native.tensor_hash(bank.masters[n]) == native.tensor_hash(v) for n, v in initial.items())
        if not report['trial_state_discarded']: raise RuntimeError('Smoke masters were not reset')
        restored = native_probe(student, probe_ids)
        report['hooks_removed_base_restored_exact'] = restored == baseline
        if restored != baseline: raise RuntimeError('Base not restored after hook removal')
        report.update(complete=True, mode='gpu_smoke', passed=True, formal_training_started=False,
                      gpu_memory=gpu_memory_receipt())
    finally:
        if bank is not None:
            bank.close()
            if initial is not None: bank.load_state_dict(initial)
            clear_gradients(bank)
            report['base_bank_identity_audit'] = bank.assert_base_frozen(check_values=True)
        if student is not None:
            report['final_student507'] = shared.audit_model(student, expected)
        if teacher is not None:
            report['final_teacher507'] = shared.audit_model(teacher, expected)
        trainer.final_recheck(inputs)
        for name, digest in pins.items():
            if native.file_hash(ROOT/name) != digest: raise RuntimeError('Smoke input changed')
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/resurface_readapted_v1_smoke.json')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'training_runs/resurface_readapted_v1_smoke')
    parser.add_argument('--data-root', type=Path, default=ROOT/'training_data/resurface_readapted_v1')
    parser.add_argument('--train-manifest-sha256', default=DATA_SHA)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    for key in ('report', 'work_dir', 'data_root'): setattr(args, key, getattr(args, key).resolve())
    if (args.report.parent != ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists()
            or args.work_dir.exists()): raise ValueError('Fresh exclusive report/work paths required')
    torch.set_num_threads(8)
    report = {'format': 'MAMBA2_RESURFACE_NATIVE_SMOKE_V1', 'complete': False, 'passed': False,
              'script_sha256': native.file_hash(__file__), 'protocol_sha256': PROTOCOL_SHA,
              'data_manifest_sha256': DATA_SHA, 'quality_measured': False, 'candidate_advancement': False}
    started = time.monotonic()
    try:
        run(args, report)
    except BaseException as exc:
        report.update(complete=False, passed=False, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write(args.report, report)


if __name__ == '__main__': main()
