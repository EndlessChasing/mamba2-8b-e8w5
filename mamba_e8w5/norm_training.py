"""Experimental FP32 norm masters with native FP16 forward and checkpointing.

The backbone orchestration follows the installed Apache-2.0 Mamba MixerModel:
embedding, native (hidden,residual) blocks, final residual add and final norm.
No quantization/runtime implementation is changed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint


HYPERPARAMETERS = {
    'epochs': 4, 'windows_per_epoch': 32, 'window_tokens': 2048, 'targets_per_window': 2047,
    'successful_updates': 128, 'maximum_skipped_attempts': 8, 'maximum_attempts': 136,
    'seed': 20260927, 'sampler': 'four sequential torch.randperm(32) draws from one seeded CPU Generator',
    'optimizer': 'AdamW', 'learning_rate': 1e-4, 'betas': [0.9, 0.999], 'epsilon': 1e-8,
    'weight_decay': 0.0, 'gradient_clip_norm': 1.0,
    'ce_coefficient': 0.5, 'teacher_to_student_kl_coefficient': 0.5, 'temperature': 1.0,
    'logits_chunk_tokens': 64, 'model_mode': 'eval with gradients enabled',
    'master_dtype': 'float32', 'forward_parameter_dtype': 'float16', 'checkpoint_each_native_block': True,
    'loss_scaler': {'device': 'cuda', 'init_scale': 1024.0, 'growth_factor': 2.0,
                    'backoff_factor': 0.5, 'growth_interval': 2000},
    'overflow_policy': 'retry same scheduled window without optimizer update; stop above8 skipped attempts',
    'selection': 'final128 successful updates only; no validation during training',
}


def selected_inventory(layers=56, width=4096, inner=8192):
    result = {}
    for index in range(layers):
        result[f'backbone.layers.{index}.norm.weight'] = {'shape': [width], 'numel': width}
        result[f'backbone.layers.{index}.mixer.norm.weight'] = {'shape': [inner], 'numel': inner}
    result['backbone.norm_f.weight'] = {'shape': [width], 'numel': width}
    return dict(sorted(result.items()))


def training_schedule():
    generator = torch.Generator(device='cpu').manual_seed(HYPERPARAMETERS['seed'])
    return [{'epoch': epoch, 'position_in_epoch': position, 'window': index}
            for epoch in range(4) for position, index in enumerate(torch.randperm(32, generator=generator).tolist())]


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class NormMasters:
    """Keep the model frozen; ordinary autograd sees only FP32 master leaves."""
    def __init__(self, model, inventory=None):
        self.model = model.eval().requires_grad_(False)
        self.inventory = selected_inventory() if inventory is None else inventory
        parameters = dict(model.named_parameters())
        self.masters = {}
        for name, entry in self.inventory.items():
            original = parameters[name]
            if list(original.shape) != entry['shape'] or original.numel() != entry['numel'] or original.dtype != torch.float16:
                raise ValueError(f'Wrong selected norm geometry/dtype: {name}')
            self.masters[name] = torch.nn.Parameter(original.detach().float().clone(), requires_grad=True)
        self.frozen_identities = {name: id(value) for name, value in parameters.items()}
        self.block_frozen = [dict(layer.named_parameters()) for layer in model.backbone.layers]
        self.final_frozen = dict(model.backbone.norm_f.named_parameters())
        if model.backbone.fused_add_norm or model.backbone.residual_in_fp32:
            raise ValueError('Norm compensation requires the frozen unfused FP16 residual path')
        for layer in model.backbone.layers:
            if layer.fused_add_norm or layer.residual_in_fp32 or layer.mlp is not None or layer.mixer.use_mem_eff_path:
                raise ValueError('Unsupported block path')

    def parameters(self):
        return list(self.masters.values())

    def state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.masters.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.masters):
            raise ValueError('Checkpoint selected norm keys differ')
        with torch.no_grad():
            for name, value in state.items():
                if value.dtype != torch.float32 or value.shape != self.masters[name].shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid checkpoint norm master: {name}')
                self.masters[name].copy_(value)

    def assert_frozen(self):
        for name, parameter in self.model.named_parameters():
            if id(parameter) != self.frozen_identities[name] or parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError(f'Frozen model parameter changed identity/gradient state: {name}')

    def forward(self, input_ids, use_checkpoint=True):
        hidden = self.model.backbone.embedding(input_ids)
        residual = None
        for index, layer in enumerate(self.model.backbone.layers):
            # Default captures prevent a late-bound layer index in recomputation.
            # FP16 casts and the functional swap happen INSIDE every replay.
            def run_block(h, r, index=index, layer=layer):
                values = dict(self.block_frozen[index])
                prefix = f'backbone.layers.{index}.'
                values['norm.weight'] = self.masters[prefix+'norm.weight'].to(torch.float16)
                values['mixer.norm.weight'] = self.masters[prefix+'mixer.norm.weight'].to(torch.float16)
                return functional_call(layer, values, (h, r), {'inference_params': None}, tie_weights=False, strict=True)
            if use_checkpoint and torch.is_grad_enabled():
                hidden, residual = checkpoint(run_block, hidden, residual, use_reentrant=False, preserve_rng_state=True)
            else:
                hidden, residual = run_block(hidden, residual)
        residual = hidden+residual if residual is not None else hidden
        final = dict(self.final_frozen)
        final['weight'] = self.masters['backbone.norm_f.weight'].to(torch.float16)
        return functional_call(self.model.backbone.norm_f, final,
                               (residual.to(dtype=self.model.backbone.norm_f.weight.dtype),),
                               tie_weights=False, strict=True)

    def export_other(self, parent_other):
        if not set(self.masters).issubset(parent_other):
            raise ValueError('Selected norms absent from parent small-tensor file')
        result = {name: value.detach().cpu().clone() for name, value in parent_other.items()}
        for name, master in self.masters.items():
            value = master.detach().cpu().half()
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite FP16 export: {name}')
            result[name] = value
        return result


def chunked_loss_backward(student_hidden, teacher_hidden, targets, student_head, teacher_head,
                          scaler, chunk_tokens=64):
    """Full-vocabulary CE/KL with one backbone backward and bounded logits memory.

Each detached student hidden chunk obtains its scaled activation derivative.
The concatenated derivative is then backpropagated through the retained backbone
graph once. All chunk sums divide by TOTAL targets, including the final short chunk.
"""
    if student_hidden.shape != teacher_hidden.shape or student_hidden.shape[:2] != targets.shape:
        raise ValueError('Hidden/target geometry mismatch')
    if not torch.isfinite(student_hidden).all() or not torch.isfinite(teacher_hidden).all():
        raise FloatingPointError('Nonfinite teacher/student hidden states')
    count = targets.numel()
    gradient = torch.empty_like(student_hidden)
    ce_sum = kl_sum = 0.0
    for first in range(0, student_hidden.shape[1], chunk_tokens):
        end = min(first+chunk_tokens, student_hidden.shape[1])
        chunk = student_hidden[:, first:end].detach().requires_grad_(True)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_hidden[:, first:end]).float()
            if not torch.isfinite(teacher_logits).all():
                raise FloatingPointError('Nonfinite teacher logits')
            teacher_logp = F.log_softmax(teacher_logits, dim=-1)
            del teacher_logits
        student_logits = student_head(chunk).float()
        if not torch.isfinite(student_logits).all():
            raise FloatingPointError('Nonfinite student logits')
        ce = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), targets[:, first:end].reshape(-1), reduction='sum')
        student_logp = F.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(student_logp, teacher_logp, reduction='sum', log_target=True)
        loss = (.5*ce+.5*kl)/count
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite CE/KL objective')
        scaled = scaler.scale(loss) if scaler is not None else loss
        grad, = torch.autograd.grad(scaled, chunk)
        gradient[:, first:end].copy_(grad)
        ce_sum += float(ce.detach()); kl_sum += float(kl.detach())
        del chunk, grad, student_logits, student_logp, teacher_logp, ce, kl, loss, scaled
    # Nonfinite backward values can be loss-scale overflow. Return them to the
    # scaler/optimizer guard; a nonfinite FORWARD value has already failed above.
    student_hidden.backward(gradient)
    return {'ce': ce_sum/count, 'teacher_to_student_kl': kl_sum/count,
            'loss': (.5*ce_sum+.5*kl_sum)/count, 'targets': count,
            'scaled_hidden_gradient_finite': bool(torch.isfinite(gradient).all()),
            'scaled_hidden_gradient_nonzero': int(torch.count_nonzero(gradient)),
            'scaled_hidden_gradient_elements': gradient.numel()}


def gradient_receipt(bank):
    finite, nonzero, elements, dtypes = True, 0, 0, set()
    per_tensor = {}
    for name, parameter in bank.masters.items():
        if parameter.grad is None:
            raise RuntimeError(f'Missing norm gradient: {name}')
        grad = parameter.grad
        if grad.dtype != torch.float32:
            raise RuntimeError(f'Master gradient is not FP32: {name}')
        ok = bool(torch.isfinite(grad).all())
        nz = int(torch.count_nonzero(grad))
        per_tensor[name] = {'finite': ok, 'nonzero': nz, 'numel': grad.numel()}
        finite &= ok; nonzero += nz; elements += grad.numel(); dtypes.add(str(grad.dtype))
    bank.assert_frozen()
    return {'finite': finite, 'tensors': len(per_tensor), 'parameters': elements, 'nonzero': nonzero,
            'dtypes': sorted(dtypes), 'per_tensor': per_tensor}


def install_other(model, other):
    """Temporary readback parity only; caller restores references in finally."""
    previous = {}
    for name, value in other.items():
        owner, leaf = name.rsplit('.', 1)
        module = model.get_submodule(owner)
        old = getattr(module, leaf)
        if value.shape != old.shape or value.dtype != torch.float16:
            raise ValueError(f'Invalid FP16 export geometry: {name}')
        previous[name] = old
        setattr(module, leaf, torch.nn.Parameter(value.to(old.device), requires_grad=False))
    return previous


def restore_parameters(model, previous):
    for name, parameter in previous.items():
        owner, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(owner), leaf, parameter)


@torch.no_grad()
def export_parity(bank, other_path, ids):
    expected = bank.forward(ids, use_checkpoint=False)
    other = torch.load(other_path, map_location='cpu', weights_only=True)
    previous = install_other(bank.model, other)
    try:
        actual = bank.model.backbone(ids)
        result = {'bitwise_equal': bool(torch.equal(expected, actual)),
                  'maximum_absolute_difference': float((expected-actual).abs().max()),
                  'functional_hidden_sha256': tensor_hash(expected), 'reloaded_hidden_sha256': tensor_hash(actual),
                  'input_tokens': ids.numel()}
        if not result['bitwise_equal']:
            raise RuntimeError(f'Functional/exported FP16 forward mismatch: {result}')
    finally:
        restore_parameters(bank.model, previous)
    bank.assert_frozen()
    return result
