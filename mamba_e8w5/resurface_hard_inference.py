"""Prepared fixed-hard0 fallback, NOT an evaluated or promoted candidate.

Reuse an unchanged FP16 soft-adapter export with logit>0 -> gate1, otherwise0.
There is no threshold/scale/task selector or training/STE path. All-closed norm
calls return the original y object; mixed calls preserve closed-position bits.
An earlier open upstream adapter can change downstream cached states: a later
closed gate proves local identity only, not equality with the original model.

Use ``with install_fp16(model, path, expected_sha256=..., expected_binding=...,
expected_base_hashes=...) as bank: model.backbone(ids)``. The native model remains
the inference path; inspect/reset ``bank.gate_stats()`` between calls. No token
logits or masks are retained. The caller must serialize forwards. A complete
PPL claim requires independent paired scoring and actual gate-coverage checks.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from . import resurface_native as native

POLICY = 'RESURFACE_POST_D_HARD0_GATE1_EXACT_CLOSED_V1'
NATIVE_SHA256 = '2dd08c7ee8958c832f0ae9da7cf5f261dceeff3e528e493c905c7c52df7166d7'


@torch.no_grad()
def hard0_readout(y, u, *, V_read, g_read, router_w, router_b, heads, head_dim):
    """Pure inference primitive returning (output, scalar-only call statistics)."""
    if (type(heads) is not int or type(head_dim) is not int or heads <= 0 or head_dim <= 0
            or y.ndim not in (2, 3) or y.dtype != torch.float16 or not y.numel()
            or y.shape[-1] != heads*head_dim or u.ndim != y.ndim
            or u.shape[:-1] != y.shape[:-1] or u.dtype != torch.float16):
        raise ValueError('Invalid FP16 head/input geometry')
    shapes = ((heads, heads), (heads,), (u.shape[-1],), ())
    for value, shape in zip((V_read, g_read, router_w, router_b), shapes):
        if value.dtype != torch.float16 or tuple(value.shape) != shape or value.device != y.device:
            raise ValueError('Fixed-hard0 requires actual FP16 adapter tensors on the input device')
    if u.device != y.device:
        raise ValueError('Mixer input/readout devices differ')
    logits = (torch.einsum('...d,d->...', u, router_w)+router_b).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite fixed-hard0 router logits')
    opened = logits > 0
    count, nopen = logits.numel(), int(opened.sum())
    stats = {'token_layer_decisions': count, 'opened': nopen, 'closed': count-nopen,
             'zero_logits': int((logits == 0).sum()), 'minimum_logit': float(logits.min()),
             'maximum_logit': float(logits.max()), 'all_closed': nopen == 0,
             'all_open': nopen == count, 'mixed': 0 < nopen < count}
    if nopen == 0:
        # No add/multiply/copy/view: preserves object, storage, strides and -0.
        return y, stats
    yh = y.reshape(*y.shape[:-1], heads, head_dim)
    chosen = yh[opened]
    mixed = torch.einsum('ij,kjp->kip', V_read, chosen)
    corrected = chosen+g_read[None, :, None]*mixed
    if not torch.isfinite(corrected).all():
        raise FloatingPointError('Nonfinite open-position hard0 correction')
    if nopen == count:
        return corrected.reshape_as(y), stats
    candidate = torch.zeros_like(yh)
    candidate[opened] = corrected
    # Closed rows never enter V@y. where copies their exact original FP16 bits,
    # including signed zero; mixed output need not retain the original strides.
    return torch.where(opened[..., None, None], candidate, yh).reshape_as(y), stats


def _empty():
    return {'calls': 0, 'token_layer_decisions': 0, 'opened': 0, 'closed': 0, 'zero_logits': 0,
            'fully_closed_calls': 0, 'fully_open_calls': 0, 'mixed_calls': 0,
            'minimum_logit': None, 'maximum_logit': None, 'failed_calls': 0}


class HardResurfaceInference(native.ResurfaceNative):
    """Inference-only native hook bank; construct through install_fp16()."""
    def __init__(self, model, *, expected_base_hashes=None, production=True):
        self._stats = {}
        self._ready = False
        self._adapter_identity = self._adapter_hashes = None
        self.source_receipt = None
        super().__init__(model, gate_mode='hard', trainable=False,
                         expected_base_hashes=expected_base_hashes, production=production)
        self._stats = {f'layer{i}': _empty() for i in range(len(self.geometry))}

    def _norm_hook(self, index):
        def hook(module, args, kwargs):
            if not self._ready:
                raise RuntimeError('Hard0 bank must be loaded/sealed from an actual FP16 export')
            frame = self._frames.get(index)
            if frame is None or frame['consumed']:
                raise RuntimeError('Hard0 norm call has no unique mixer input')
            y = args[0] if args else kwargs.get('x')
            geo = self.geometry[index]
            if not isinstance(y, torch.Tensor) or y.ndim not in (2, 3):
                raise ValueError('Native norm input must have rank2/3')
            u = frame['u']
            if u.numel()//geo['width'] != y.numel()//y.shape[-1]:
                raise ValueError('Mixer/readout token counts differ')
            u = u.reshape(*y.shape[:-1], geo['width'])
            params = {name: getattr(self.adapters[f'layer{index}'], name) for name in native.FIELDS}
            try:
                adapted, row = hard0_readout(y, u, heads=geo['heads'], head_dim=geo['head_dim'], **params)
            except BaseException:
                self._stats[f'layer{index}']['failed_calls'] += 1
                raise
            stat = self._stats[f'layer{index}']
            stat['calls'] += 1
            for key in ('token_layer_decisions', 'opened', 'closed', 'zero_logits'):
                stat[key] += row[key]
            for output, source in (('fully_closed_calls', 'all_closed'), ('fully_open_calls', 'all_open'), ('mixed_calls', 'mixed')):
                stat[output] += int(row[source])
            for key, reduce in (('minimum_logit', min), ('maximum_logit', max)):
                stat[key] = row[key] if stat[key] is None else reduce(stat[key], row[key])
            # No captured per-token tensor survives the call, even transiently
            # in bank state. Native end-hook clears this frame after the mixer.
            frame.update(u=None, consumed=True, logits=None)
            if args:
                return (adapted, *args[1:]), kwargs
            return args, {**kwargs, 'x': adapted}
        return hook

    def load_state_dict(self, values):
        if self._ready:
            raise RuntimeError('Loaded hard0 tables are immutable; install from a new bound file')
        return super().load_state_dict(values)

    def export_fp16(self, *args, **kwargs):
        raise RuntimeError('Reuse the unchanged soft export; hard0 policy needs a separate explicit manifest')

    @torch.no_grad()
    def forward_hidden(self, input_ids, use_checkpoint=False):
        """Convenience only: native backbone plus scalar stats, not GateCapture."""
        self._require_open()
        if use_checkpoint:
            raise ValueError('Fixed-hard0 is inference-only; checkpoint/training is unsupported')
        hidden = self.model.backbone(input_ids)
        return hidden, self.gate_stats()

    def gate_stats(self):
        per_layer = copy.deepcopy(self._stats)
        total = {key: sum(row[key] for row in per_layer.values()) for key in
                 ('calls', 'token_layer_decisions', 'opened', 'closed', 'zero_logits',
                  'fully_closed_calls', 'fully_open_calls', 'mixed_calls', 'failed_calls')}
        for key, reduce in (('minimum_logit', min), ('maximum_logit', max)):
            values = [row[key] for row in per_layer.values() if row[key] is not None]
            total[key] = reduce(values) if values else None
        total['all_observed_gates_closed'] = total['opened'] == 0 if total['token_layer_decisions'] else None
        return {'policy': POLICY, 'per_layer': per_layer, 'total': total,
                'per_token_tensors_retained': False, 'no_extra_recurrent_state': True}

    def reset_gate_stats(self):
        if self._frames or self._requests:
            raise RuntimeError('Cannot reset statistics during a native invocation')
        previous = self.gate_stats()
        self._stats = {name: _empty() for name in self._stats}
        return previous

    def validate_gate_coverage(self, *, calls_per_layer, tokens_per_layer):
        if any(type(x) is not int or x < 1 for x in (calls_per_layer, tokens_per_layer)):
            raise ValueError('Positive integer coverage required')
        receipt = self.gate_stats()
        for name, row in receipt['per_layer'].items():
            if (row['calls'] != calls_per_layer or row['token_layer_decisions'] != tokens_per_layer
                    or row['opened']+row['closed'] != tokens_per_layer or row['failed_calls']):
                raise RuntimeError(f'Incomplete/failed native hard0 gate coverage: {name}')
        return receipt

    def assert_adapter_unchanged(self, recheck_file=True):
        if not self._ready:
            raise RuntimeError('No sealed FP16 export')
        for name, p in self.masters.items():
            if ((id(p), p.data_ptr(), p._version) != self._adapter_identity[name]
                    or p.dtype != torch.float16 or p.requires_grad or p.grad is not None
                    or native.tensor_hash(p) != self._adapter_hashes[name]):
                raise RuntimeError(f'Hard0 adapter changed: {name}')
        if recheck_file and native.file_hash(self.source_receipt['file']) != self.source_receipt['sha256']:
            raise RuntimeError('Reused soft export changed')
        return {'policy': POLICY, 'source_export': copy.deepcopy(self.source_receipt),
                'fp16_tensor_sha256': dict(self._adapter_hashes), 'all_actual_tensors_unchanged': True}


def install_fp16(model, path, *, expected_sha256=None, expected_binding=None,
                 expected_base_hashes=None, production=True):
    """Bind unchanged SOFT-export bytes; explicitly select the fixed hard0 policy."""
    if native.file_hash(Path(native.__file__)) != NATIVE_SHA256:
        raise ValueError('Frozen native hook/readback dependency differs')
    path = Path(path).resolve()
    if production and (expected_sha256 is None or expected_binding is None or expected_base_hashes is None):
        raise ValueError('Production install requires export SHA, binding and complete base hash ledger')
    before = native.file_hash(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError('Soft export identity differs')
    payload = native.read_fp16(path, expected_binding)
    if payload['gate_mode'] != 'soft':
        raise ValueError('This conditional conversion requires an unchanged soft-gate export')
    bank = HardResurfaceInference(model, expected_base_hashes=expected_base_hashes, production=production)
    try:
        if bank.geometry != payload['geometry']:
            raise ValueError('FP16 adapter geometry differs from actual native model')
        bank.load_state_dict(payload['tensors'])
        bank._adapter_hashes = {name: native.tensor_hash(value) for name, value in payload['tensors'].items()}
        bank._adapter_identity = {name: (id(p), p.data_ptr(), p._version) for name, p in bank.masters.items()}
        bank.source_receipt = {'file': str(path), 'bytes': path.stat().st_size, 'sha256': before,
            'stored_gate_mode': 'soft', 'effective_policy': POLICY, 'threshold': 0, 'open_gate_value': 1,
            'weights_modified': False, 'parameters': sum(v.numel() for v in payload['tensors'].values()),
            'fp16_payload_bytes': sum(v.numel()*v.element_size() for v in payload['tensors'].values())}
        bank._ready = True
        bank.assert_adapter_unchanged()
    except BaseException:
        bank.close()
        raise
    return bank
