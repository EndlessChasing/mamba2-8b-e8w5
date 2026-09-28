"""Isolated rank-4 FP16-merged projection training on an immutable native base.

Only FP32 A/B factor masters are trainable. Each native block replay casts them
to FP16, forms the dense FP32 sum with detached W0, then rounds once to FP16.
There is no E8 decode in this graph and no separate two-linear residual branch.
The base's existing decoded FP16 storage is retained once, without a clone.

CUDA callers must explicitly configure highest FP32 matmul precision/TF32 off.
This module never changes global precision flags; local autocast is disabled.
Ordinary autograd through casts is a surrogate for discrete FP16 rounding.
"""
from __future__ import annotations

import copy
import math

import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .low_rank_residual import factor_layout, validate_factors
from .norm_training import (HYPERPARAMETERS as NORM_HYPERPARAMETERS,
                            chunked_loss_backward, tensor_hash)


RANK = 4
HYPERPARAMETERS = copy.deepcopy(NORM_HYPERPARAMETERS)
HYPERPARAMETERS.update({
    'seed': 20260928, 'initialization_seed': 20260928, 'rank': RANK,
    'windows_per_epoch': 256, 'successful_updates': 1024, 'maximum_attempts': 1032,
    'learning_rate': 3e-4, 'selected_tensors': 224, 'selected_parameters': 7827456,
    'selected_projections': 112,
    'sampler': 'four sequential torch.randperm(256) draws from one seeded CPU Generator',
    'selection': 'final1024 successful updates only; no validation during training',
    'initialization': 'B zero; A CPU FP32 normal times 1/sqrt(n); separate seeded CPU generator',
    'initialization_order': 'numeric layer order, in_proj then out_proj; only A consumes RNG',
    'base': 'accepted all-small FP16 model; no prototype corrections',
    'forward_parameter_dtype': 'FP16 factors; FP32 merge; native FP16 projection weights',
    'merge': 'half(W0.detach().float() + B16.float() @ A16.float())',
    'rounding_backward': 'ordinary autograd through FP16 casts; surrogate for discrete rounding',
    'checkpoint_interval_successes': 32,
    'training_data': '256 fresh TRAIN windows excluding calibration32, small256, crossmoment48 and prototype32',
})


def selected_inventory(layers=56, width=4096, in_rows=18560, inner=8192):
    """Canonical numeric order; each entry describes one B-then-A factor file."""
    if type(layers) is not int or layers < 1:
        raise ValueError('layers must be a positive exact integer')
    result = {}
    for index in range(layers):
        for part, (m, n) in (('in_proj', (in_rows, width)), ('out_proj', (width, inner))):
            result[f'layer{index}.{part}'] = {
                **factor_layout(m, n, RANK),
                'source_key': f'backbone.layers.{index}.mixer.{part}.weight',
            }
    return result


def training_schedule():
    generator = torch.Generator(device='cpu').manual_seed(HYPERPARAMETERS['seed'])
    return [index for _ in range(4) for index in torch.randperm(256, generator=generator).tolist()]


def _identity(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.dtype, str(value.device))


def _check_precision(device):
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA native merge paths are supported')
    if device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32 or
                                 torch.get_float32_matmul_precision() != 'highest'):
        raise RuntimeError('Caller must explicitly set highest FP32 matmul precision and TF32 off')


def merge_fp16_weight(base, B, A):
    """Native merge for same-device FP16 inputs; no global precision mutation.

    Shape/dtype checks are cheap and run on every replay. Full finite/content
    checks belong to initialization/export receipts and the training loss guard,
    avoiding full dense-weight scans or synchronizations in every block replay.
    No zero-factor shortcut: signed-zero behavior follows the declared addition.
    """
    if any(not isinstance(v, torch.Tensor) or v.ndim != 2 or
           v.dtype != torch.float16 or v.layout != torch.strided for v in (base, B, A)):
        raise ValueError('Base and factors must be strided FP16 matrices')
    m, n = base.shape
    if B.shape != (m, RANK) or A.shape != (RANK, n):
        raise ValueError('Base/factor shape or rank differs from the rank-4 contract')
    if B.device != base.device or A.device != base.device:
        raise ValueError('Base and factors must be on the same device')
    _check_precision(base.device)
    with torch.autocast(device_type=base.device.type, enabled=False):
        return (base.detach().float() + B.float() @ A.float()).half()


class LowRankMasters:
    """224 FP32 leaves for the real model, with immutable decoded FP16 W0 refs.

    Model/base provenance is the driver's responsibility. Identity/version
    checks detect normal in-place changes; explicit frozen_hashes() receipts
    independently cover contents, including writes made outside autograd.
    """
    def __init__(self, model, *, rank=RANK, seed=20260928):
        if type(rank) is not int or rank != RANK:
            raise ValueError('This fixed experiment supports rank 4 only')
        if type(seed) is not int or seed != HYPERPARAMETERS['initialization_seed']:
            raise ValueError('Initialization seed differs from the fixed recipe')
        self.model = model.eval().requires_grad_(False)
        if model.backbone.fused_add_norm or model.backbone.residual_in_fp32:
            raise ValueError('Low-rank compensation requires native unfused FP16 residuals')
        parameters = dict(model.named_parameters())
        if any(p.grad is not None for p in parameters.values()):
            raise ValueError('Frozen model has pre-existing parameter gradients')
        self.inventory = {}
        self.masters = {}
        self.base_weights = {}
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.block_frozen = []
        self.block_buffers = []
        for index, layer in enumerate(model.backbone.layers):
            if (layer.fused_add_norm or layer.residual_in_fp32 or layer.mlp is not None
                    or layer.mixer.use_mem_eff_path):
                raise ValueError('Unsupported native block path')
            self.block_frozen.append(dict(layer.named_parameters()))
            self.block_buffers.append(dict(layer.named_buffers()))
            for part in ('in_proj', 'out_proj'):
                module = getattr(layer.mixer, part)
                if not isinstance(module, torch.nn.Linear) or module.bias is not None:
                    raise ValueError(f'Expected bias-free native Linear: layer{index}.{part}')
                label = f'layer{index}.{part}'
                key = f'backbone.layers.{index}.mixer.{part}.weight'
                original = parameters[key]
                if original.dtype != torch.float16 or original.ndim != 2:
                    raise ValueError(f'Expected native FP16 projection: {label}')
                _check_precision(original.device)
                m, n = original.shape
                self.inventory[label] = {**factor_layout(m, n, rank), 'source_key': key}
                self.base_weights[label] = original.detach()
                # Only A draws from this generator; B creation cannot move RNG.
                initial_a = torch.randn((rank, n), generator=generator, dtype=torch.float32) * (1 / math.sqrt(n))
                self.masters[label + '.B'] = torch.nn.Parameter(torch.zeros(m, rank, device=original.device, dtype=torch.float32))
                self.masters[label + '.A'] = torch.nn.Parameter(initial_a.to(original.device))
        if not self.inventory:
            raise ValueError('At least one native block is required')
        self.frozen_parameters = parameters
        self.frozen_identities = {name: _identity(value) for name, value in parameters.items()}
        self.frozen_buffers = {name: _identity(value) for name, value in model.named_buffers()}
        self.base_identities = {name: _identity(value) for name, value in self.base_weights.items()}

    def parameters(self):
        return list(self.masters.values())

    def state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.masters.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.masters):
            raise ValueError('Checkpoint factor master keys differ')
        # Validate every tensor first, so rejection cannot partially load state.
        for name, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                    value.layout != torch.strided or value.shape != self.masters[name].shape or
                    not torch.isfinite(value).all() or not torch.isfinite(value.half()).all()):
                raise ValueError(f'Invalid FP32/FP16-representable factor master: {name}')
        with torch.no_grad():
            for name, value in state.items():
                self.masters[name].copy_(value)

    def frozen_lookup(self, name):
        """Return the retained original parameter reference; callers must not mutate it."""
        return self.frozen_parameters[name]

    def frozen_hashes(self):
        self.assert_frozen()
        return {name: tensor_hash(value) for name, value in self.frozen_parameters.items()}

    def assert_frozen(self):
        parameters = dict(self.model.named_parameters())
        if set(parameters) != set(self.frozen_identities):
            raise RuntimeError('Frozen parameter coverage changed')
        for name, value in parameters.items():
            if (_identity(value) != self.frozen_identities[name] or value.requires_grad
                    or value.grad is not None):
                raise RuntimeError(f'Frozen native parameter changed: {name}')
        if {name: _identity(value) for name, value in self.model.named_buffers()} != self.frozen_buffers:
            raise RuntimeError('Frozen buffer identity/version changed')
        if {name: _identity(value) for name, value in self.base_weights.items()} != self.base_identities:
            raise RuntimeError('Retained base weight identity/version changed')

    def merge(self, label, factors=None):
        """Explicit factor half casts execute inside every block checkpoint replay."""
        base = self.base_weights[label]
        with torch.autocast(device_type=base.device.type, enabled=False):
            B, A = ((self.masters[label + '.B'].half(), self.masters[label + '.A'].half())
                    if factors is None else factors)
            return merge_fp16_weight(base, B, A)

    def forward_block(self, index, hidden, residual, use_checkpoint=True):
        if type(index) is not int or not 0 <= index < len(self.block_frozen):
            raise ValueError('Block index is out of range')
        layer = self.model.backbone.layers[index]
        def run_block(h, r):
            with torch.autocast(device_type=h.device.type, enabled=False):
                values = dict(self.block_frozen[index])
                for part in ('in_proj', 'out_proj'):
                    values[f'mixer.{part}.weight'] = self.merge(f'layer{index}.{part}')
                return functional_call(layer, (values, self.block_buffers[index]), (h, r),
                                       {'inference_params': None}, tie_weights=False, strict=True)
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(run_block, hidden, residual, use_reentrant=False, preserve_rng_state=True)
        return run_block(hidden, residual)

    def forward(self, input_ids, use_checkpoint=True):
        """Backbone hidden states; use frozen chunked_loss_backward for CE/KL.

        Training checkpoints each block. Dense merged weights/weight gradients
        are transient during replay; this is not compressed GPU residency.
        """
        with torch.autocast(device_type=self.model.backbone.embedding.weight.device.type, enabled=False):
            hidden = self.model.backbone.embedding(input_ids)
            residual = None
            for index in range(len(self.block_frozen)):
                hidden, residual = self.forward_block(index, hidden, residual, use_checkpoint)
            residual = hidden + residual if residual is not None else hidden
            return self.model.backbone.norm_f(residual.to(self.model.backbone.norm_f.weight.dtype))

    def export_factors(self):
        """Return label -> (B16, A16) independent CPU tensors for write_factors."""
        result = {}
        for label in self.inventory:
            masters = [self.masters[label + '.' + part] for part in ('B', 'A')]
            if any(not torch.isfinite(value).all() for value in masters):
                raise ValueError(f'Nonfinite FP32 factor master: {label}')
            B, A = [value.detach().cpu().half() for value in masters]
            validate_factors(B, A)
            result[label] = (B, A)
        return result


def gradient_receipt(bank):
    per_tensor = {}
    for name, master in bank.masters.items():
        grad = master.grad
        if grad is None or grad.dtype != torch.float32:
            raise RuntimeError(f'Missing or non-FP32 factor gradient: {name}')
        finite = bool(torch.isfinite(grad).all())
        per_tensor[name] = {'finite': finite,
                            'nonzero': int(torch.count_nonzero(grad)), 'numel': grad.numel(),
                            'minimum': float(grad.min()) if finite else None,
                            'maximum': float(grad.max()) if finite else None,
                            'maximum_absolute': float(grad.abs().max()) if finite else None}
    bank.assert_frozen()
    return {'finite': all(v['finite'] for v in per_tensor.values()),
            'tensors': len(per_tensor), 'parameters': sum(v['numel'] for v in per_tensor.values()),
            'nonzero': sum(v['nonzero'] for v in per_tensor.values()),
            'dtypes': ['torch.float32'], 'per_tensor': per_tensor}


@torch.no_grad()
def export_parity(bank, factors, ids):
    """Compare live masters with read-back FP16 pairs installed as native weights.

    All parameter references are restored in finally, including on failure.
    Temporarily installing all projections requires additional dense GPU memory.
    """
    if set(factors) != set(bank.inventory):
        raise ValueError('Export factor coverage differs')
    for label, (B, A) in factors.items():
        layout = validate_factors(B, A)
        if layout['shape'] != bank.inventory[label]['shape'] or layout['rank'] != RANK:
            raise ValueError(f'Export factor geometry differs: {label}')
    expected = bank.forward(ids, use_checkpoint=False)
    expected_logits = bank.model.lm_head(expected[:, -8:])
    previous = {}
    try:
        for label, entry in bank.inventory.items():
            owner, leaf = entry['source_key'].rsplit('.', 1)
            module = bank.model.get_submodule(owner)
            previous[entry['source_key']] = getattr(module, leaf)
            device = bank.base_weights[label].device
            merged = bank.merge(label, tuple(v.to(device) for v in factors[label]))
            setattr(module, leaf, torch.nn.Parameter(merged, requires_grad=False))
        actual = bank.model.backbone(ids)
        actual_logits = bank.model.lm_head(actual[:, -8:])
        if not all(torch.isfinite(v).all() for v in (expected, actual, expected_logits, actual_logits)):
            raise FloatingPointError('Nonfinite native/export parity output')
        expected_hash, actual_hash = tensor_hash(expected), tensor_hash(actual)
        expected_logits_hash, actual_logits_hash = tensor_hash(expected_logits), tensor_hash(actual_logits)
        hidden_equal, logits_equal = expected_hash == actual_hash, expected_logits_hash == actual_logits_hash
        result = {'bitwise_equal': hidden_equal and logits_equal,
                  'hidden_bitwise_equal': hidden_equal,
                  'last8_logits_bitwise_equal': logits_equal,
                  'maximum_absolute_difference': float((expected - actual).abs().max()),
                  'functional_hidden_sha256': expected_hash, 'reloaded_hidden_sha256': actual_hash,
                  'functional_last8_logits_sha256': expected_logits_hash,
                  'reloaded_last8_logits_sha256': actual_logits_hash,
                  'input_tokens': ids.numel(), 'factor_files': len(factors)}
        if not result['bitwise_equal']:
            raise RuntimeError(f'Low-rank export/native forward mismatch: {result}')
    finally:
        for name, value in previous.items():
            owner, leaf = name.rsplit('.', 1)
            setattr(bank.model.get_submodule(owner), leaf, value)
    bank.assert_frozen()
    return result
