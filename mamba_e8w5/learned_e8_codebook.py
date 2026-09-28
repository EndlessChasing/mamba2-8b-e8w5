"""Fixed-index, per-projection E8-derived prototype corrections.

The pinned E8P code c uses prototype p=c>>8 and eight sign/coset bits. A
256x8 FP16 table delta is indexed in *decoded coordinate order*. The expanded
value is exactly base.grid[c,j] + sigma(c,j)*delta[p,j], with sigma derived
from the pinned packed absolute prototype and original sign/parity/shuffle
rules. Original +/-1/4 offsets are unchanged; parity is never recomputed from
learned values. Arbitrary corrections no longer form a strict E8 lattice.

No indices, balance, scale, rotations or frozen codec code are changed. This
module is representation infrastructure; it does not train or change a model.
Sign/layout interpretation derives from the GPL-3.0 QuIP# source pinned by this
repository (lib/codebook/latticee8_padded12.py, get_full_grid).

File format: 32-byte little-endian <8sHHHHQQ header: magic, version, rows,
columns, dtype (1=IEEE FP16), payload byte count, reserved zero. Then exactly
4096 bytes of row-major little-endian FP16. Total4128 bytes per projection;
112 separately stored tables require462336 bytes, excluding manifests.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import torch

from .codec import decode_e8

PROTOTYPES = 256
COORDINATES = 8
CODEWORDS = 65536
SHUFFLE = (0, 4, 1, 5, 2, 6, 3, 7)
PACKED_BASE_SHA256 = 'efc2c03c60acd955dc812ff2bade9ec6cc31259807e48473506161f3a9a230ce'
MAGIC = b'ME8P0001'
HEADER = struct.Struct('<8sHHHHQQ')
PAYLOAD_BYTES = PROTOTYPES * COORDINATES * 2
FILE_BYTES = HEADER.size + PAYLOAD_BYTES
EXPECTED_HEADER = (MAGIC, 1, PROTOTYPES, COORDINATES, 1, PAYLOAD_BYTES, 0)


def validate_prototypes(delta):
    """Require an explicit finite FP16 table; never silently quantize inputs."""
    if not isinstance(delta, torch.Tensor):
        raise TypeError('Prototype correction must be a torch.Tensor')
    if delta.dtype != torch.float16 or tuple(delta.shape) != (PROTOTYPES, COORDINATES):
        raise ValueError('Expected prototype corrections with shape256x8 and dtypeFP16')
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite prototype correction')
    return delta


def _packed_base(cb, device):
    packed = getattr(cb, 'grid_packed_abs', None)
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.int32 or tuple(packed.shape) != (256,):
        raise ValueError('Expected pinned256-entry packed int32 codebook')
    raw = packed.detach().cpu().numpy().astype('<i4', copy=False).tobytes()
    if hashlib.sha256(raw).hexdigest() != PACKED_BASE_SHA256:
        raise ValueError('Packed base codebook differs from pinned E8P')
    return packed.to(device=device, dtype=torch.int64)


def prototype_mapping(cb, device='cpu'):
    """Return high8 prototype IDs and exact signed correction directions.

    The returned directions have shape65536x8 in decoded coordinate order.
    The packed base already includes the original prototype-dependent sign of
    its last packed coordinate, which must be included in sigma.
    """
    packed = _packed_base(cb, device)
    codes = torch.arange(CODEWORDS, dtype=torch.int64, device=device)
    prototypes = codes >> 8
    low = codes & 255
    parity = torch.zeros_like(low)
    for bit in range(8):
        parity = parity ^ ((low >> bit) & 1)
    corrected_sign_bits = low ^ parity
    shuffle = torch.tensor(SHUFFLE, dtype=torch.int64, device=device)
    nibbles = ((packed[prototypes, None] >> (4*shuffle)[None]) & 15) - 8
    packed_sign = torch.where(nibbles < 0, -1., 1.)
    bit_sign = 1 - 2*((corrected_sign_bits[:, None] >> shuffle[None]) & 1)
    directions = packed_sign.float() * bit_sign.float()
    return prototypes, directions


def corrected_grid(cb, delta, device=None):
    """Construct the learned65536x8 FP32 grid, preserving delta gradients.

    Zero corrections add exact signed zeros to nonzero quarter-valued base
    entries, preserving every base bit. No special-case branch suppresses the
    derivative at zero. This makes a future differentiable fit possible, but
    does not implement or authorize such training.
    """
    validate_prototypes(delta)
    return _corrected_grid(cb, delta, device)


def corrected_grid_from_rounded_fp32(cb, delta, device=None):
    """Optional adjoint primitive with FP32 gradients and FP16-representable values.

    Callers may supply an FP32 straight-through rounded tensor. This primitive
    never rounds silently and never relaxes the strict FP16 inference/file API.
    It is a pure tensor operation, not a training loop or optimizer.
    """
    if (not isinstance(delta, torch.Tensor) or delta.dtype != torch.float32
            or tuple(delta.shape) != (256, 8) or not torch.isfinite(delta).all()
            or not torch.equal(delta, delta.half().float())):
        raise ValueError('Expected finite FP32 table with exactly FP16-representable values')
    return _corrected_grid(cb, delta, device)


def _corrected_grid(cb, delta, device):
    base = getattr(cb, 'grid', None)
    if not isinstance(base, torch.Tensor) or base.dtype != torch.float32 or tuple(base.shape) != (CODEWORDS, COORDINATES):
        raise ValueError('Expected original65536x8 FP32 E8P grid')
    device = base.device if device is None else device
    prototypes, directions = prototype_mapping(cb, device=device)
    # Independently reconstruct the base entries to reject a stale/mutated grid
    # even when the1024-byte packed codebook still has the correct identity.
    packed = cb.grid_packed_abs.to(device=device, dtype=torch.int64)
    shuffle = torch.tensor(SHUFFLE, dtype=torch.int64, device=device)
    magnitudes = (((packed[prototypes, None] >> (4*shuffle)[None]) & 15)-8).abs().float()*.5
    low = torch.arange(CODEWORDS, device=device, dtype=torch.int64) & 255
    parity = torch.zeros_like(low)
    for bit in range(8):
        parity = parity ^ ((low >> bit) & 1)
    expected = directions*magnitudes + (1-2*parity[:, None]).float()*.25
    base = base.to(device)
    if not torch.equal(base, expected):
        raise ValueError('Expanded base grid differs from pinned packed decode')
    return base + directions*delta.to(device=device, dtype=torch.float32)[prototypes]


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError('Expected frozen E8 payload dictionary')
    indices = payload.get('indices')
    if (not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.uint16, torch.int32, torch.int64)
            or indices.ndim != 2 or min(indices.shape) <= 0):
        raise ValueError('Expected nonempty2D uint16/int32/int64 E8 indices')
    values = indices.to(dtype=torch.int64)
    if values.min() < 0 or values.max() >= CODEWORDS:
        raise ValueError('E8 indices must fit uint16')
    if payload.get('axis_residual_amplitude', 0.0) != 0.0:
        raise ValueError('Prototype correction supports original16-bit E8 indices only')
    m, groups = indices.shape
    n = groups*8
    balance = payload.get('balance')
    if (not isinstance(balance, torch.Tensor) or balance.dtype != torch.float16
            or tuple(balance.shape) != (n,) or not torch.isfinite(balance).all() or not (balance > 0).all()):
        raise ValueError('Expected finite positive FP16 balance vector')
    for key, size in (('input_sign', n), ('output_sign', m)):
        sign = payload.get(key)
        if (not isinstance(sign, torch.Tensor) or sign.dtype != torch.int8 or tuple(sign.shape) != (size,)
                or not ((sign == 1) | (sign == -1)).all()):
            raise ValueError(f'Invalid frozen sign vector: {key}')
    scale = payload.get('scale')
    if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or scale.numel() != 1
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError('Expected positive finite FP32 scalar scale')


def decode_e8_with_prototypes(payload, cb, delta, device='cuda'):
    """Reuse frozen inverse-transform order and final FP16 rounding exactly."""
    _validate_payload(payload)
    grid = corrected_grid(cb, delta, device=device)
    result = decode_e8(payload, SimpleNamespace(grid=grid), device=device)
    if not torch.isfinite(result).all():
        raise ValueError('Learned E8-derived decode overflows FP16')
    return result


def read_prototypes(path, device='cpu'):
    path = Path(path)
    if path.stat().st_size != FILE_BYTES:
        raise ValueError(f'Prototype file must contain exactly{FILE_BYTES} bytes')
    raw = path.read_bytes()
    if len(raw) != FILE_BYTES or HEADER.unpack(raw[:HEADER.size]) != EXPECTED_HEADER:
        raise ValueError('Invalid prototype file header or length')
    values = np.frombuffer(raw[HEADER.size:], dtype='<f2').copy().reshape(PROTOTYPES, COORDINATES)
    delta = torch.from_numpy(values).to(device)
    return validate_prototypes(delta)


def write_prototypes(path, delta):
    """Write one finite FP16 table; preserve existing files; verify every bit."""
    validate_prototypes(delta)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    raw = delta.detach().cpu().contiguous().numpy().astype('<f2', copy=False).tobytes()
    encoded = HEADER.pack(*EXPECTED_HEADER) + raw
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    restored = read_prototypes(temporary)
    restored_bytes = restored.contiguous().numpy().astype('<f2', copy=False).tobytes()
    if restored_bytes != raw:
        raise RuntimeError('Prototype disk readback differs bitwise')
    # Exclusive hard-link publication avoids overwriting a concurrently created
    # destination; the temporary stays available if any publication step fails.
    os.link(temporary, path)
    temporary.unlink()
    return {'format': 'MAMBA2_LEARNED_E8_PROTOTYPE_TABLE_V1', 'bytes': len(encoded),
        'header_bytes': HEADER.size, 'payload_bytes': len(raw), 'shape': [256, 8], 'dtype': 'float16',
        'sha256': hashlib.sha256(encoded).hexdigest(), 'payload_sha256': hashlib.sha256(raw).hexdigest(),
        'disk_roundtrip_bitwise_equal': True,
        'note': 'E8-derived prototype correction; fixed original indices; not a strict E8 lattice.'}
