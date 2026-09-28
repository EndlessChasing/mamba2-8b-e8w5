"""CPU preparation for strictly serialized FP16 low-rank weight corrections.

One projection stores B[m, r], then A[r, n], as row-major little-endian FP16.
The 32-byte <8sHHIIIQ header contains magic, version, dtype, m, n, r and
payload bytes. No pickle, padding, scaling field, bias or trailing data.

The merge contract is half(W0.detach().float() + B16.float() @ A16.float()).
It is not a two-linear residual branch. FP32 optimizer masters, if used later,
must be explicitly cast to half by the caller inside every checkpoint replay.
Ordinary autograd traverses these casts; the frozen base has no gradient path.
This module deliberately accepts CPU tensors only and changes no global torch
precision state. It makes no CUDA, model-quality or compact-residency claim.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import struct

import numpy as np
import torch

MAGIC = b'ME8LR001'
VERSION = 1
DTYPE_FP16 = 1
HEADER = struct.Struct('<8sHHIIIQ')
FORMAT = 'MAMBA2_LOW_RANK_FP16_FACTORS_V1'

# Explicit resource limits checked before a file payload is read/allocated.
# They cover every current projection and ranks4/8/16. Enlarging these limits
# is an explicit future decision, not something controlled by an input file.
MAX_DIMENSION = 65_536
MAX_RANK = 256
MAX_MATRIX_ELEMENTS = 100_000_000
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


def factor_layout(m: int, n: int, rank: int) -> dict:
    """Validate geometry and return exact factor/header capacity, without allocation."""
    if any(type(v) is not int for v in (m, n, rank)):
        raise ValueError('Dimensions and rank must be exact integers')
    if not (1 <= m <= MAX_DIMENSION and 1 <= n <= MAX_DIMENSION):
        raise ValueError('Dimensions outside configured bounds')
    if not (1 <= rank <= min(m, n, MAX_RANK)):
        raise ValueError('Rank outside configured bounds')
    if m * n > MAX_MATRIX_ELEMENTS:
        raise ValueError('Merged matrix exceeds configured element limit')
    payload_bytes = 2 * rank * (m + n)
    if payload_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError('Factor payload exceeds configured byte limit')
    return {'shape': [m, n], 'rank': rank, 'B_shape': [m, rank],
        'A_shape': [rank, n], 'dtype': 'float16', 'value_count': rank * (m + n),
        'header_bytes': HEADER.size, 'payload_bytes': payload_bytes,
        'bytes': HEADER.size + payload_bytes}


def _cpu_half_matrix(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    if (value.dtype != torch.float16 or value.ndim != 2
            or value.device.type != 'cpu' or value.layout != torch.strided):
        raise ValueError(f'{name} must be a strided CPU FP16 matrix')


def validate_factors(B, A):
    """Reject implicit dtype/device conversions and nonfinite factor values."""
    _cpu_half_matrix(B, 'B')
    _cpu_half_matrix(A, 'A')
    if B.shape[1] != A.shape[0]:
        raise ValueError('B and A have different ranks')
    layout = factor_layout(B.shape[0], A.shape[1], B.shape[1])
    if not torch.isfinite(B).all() or not torch.isfinite(A).all():
        raise ValueError('Nonfinite factor values')
    return layout


def _factor_bytes(value):
    return value.detach().contiguous().numpy().astype('<f2', copy=False).tobytes(order='C')


def read_factors(path, *, expected_shape=None, expected_rank=None):
    """Read a bounded, exact-length regular file into independent CPU B/A tensors.

    Optional expected geometry should be supplied by a future model manifest;
    a self-consistent header alone cannot identify the intended projection.
    """
    path = Path(path)
    # O_NONBLOCK avoids waiting on named pipes before the regular-file check.
    # No-follow prevents accidentally accepting a symlink as the artifact.
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Factor artifact must be a regular file')
        if not HEADER.size <= info.st_size <= HEADER.size + MAX_PAYLOAD_BYTES:
            raise ValueError('Invalid factor file length')
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError('Truncated factor header')
        magic, version, dtype, m, n, rank, payload_bytes = HEADER.unpack(header)
        if (magic, version, dtype) != (MAGIC, VERSION, DTYPE_FP16):
            raise ValueError('Invalid factor magic, version or dtype')
        layout = factor_layout(m, n, rank)
        if payload_bytes != layout['payload_bytes'] or info.st_size != layout['bytes']:
            raise ValueError('Factor dimensions, payload length and file length disagree')
        if expected_shape is not None:
            if (not isinstance(expected_shape, (tuple, list)) or len(expected_shape) != 2
                    or any(type(v) is not int for v in expected_shape)
                    or list(expected_shape) != [m, n]):
                raise ValueError('Factor dimensions differ from expected shape')
        if expected_rank is not None:
            if type(expected_rank) is not int or expected_rank != rank:
                raise ValueError('Factor rank differs from expected rank')
        raw = stream.read(payload_bytes)
        if len(raw) != payload_bytes or stream.read(1):
            raise ValueError('Truncated factor payload or extra bytes')
    # Separate copies keep B/A independent of each other and the file buffer.
    split = m * rank * 2
    b = np.frombuffer(raw[:split], dtype='<f2').astype(np.float16, copy=True).reshape(m, rank)
    a = np.frombuffer(raw[split:], dtype='<f2').astype(np.float16, copy=True).reshape(rank, n)
    B, A = torch.from_numpy(b), torch.from_numpy(a)
    validate_factors(B, A)
    return B, A


def write_factors(path, B, A):
    """Publish deterministic factor bytes exclusively, after exact disk readback.

    Existing destinations and .partial files are preserved. A failed write or
    publication leaves its partial file available for inspection.
    """
    layout = validate_factors(B, A)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    m, n = layout['shape']
    raw_b, raw_a = _factor_bytes(B), _factor_bytes(A)
    payload = raw_b + raw_a
    encoded = HEADER.pack(MAGIC, VERSION, DTYPE_FP16, m, n, layout['rank'], len(payload)) + payload
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    restored_b, restored_a = read_factors(temporary,
        expected_shape=(m, n), expected_rank=layout['rank'])
    if _factor_bytes(restored_b) != raw_b or _factor_bytes(restored_a) != raw_a:
        raise RuntimeError('Factor disk readback differs bitwise')
    os.link(temporary, path)  # Do not overwrite a concurrent destination.
    temporary.unlink()
    return {'format': FORMAT, **layout, 'sha256': hashlib.sha256(encoded).hexdigest(),
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'B_sha256': hashlib.sha256(raw_b).hexdigest(), 'A_sha256': hashlib.sha256(raw_a).hexdigest(),
        'disk_roundtrip_bitwise_equal': True}


def merge_weight(base, B, A):
    """Exact declared CPU merge, with factor gradients and no frozen-base gradient.

    There is intentionally no zero-factor shortcut. Native FP32 addition can
    turn base -0 plus a +0 correction into +0. The base itself is never changed.
    The returned FP16 matrix has fresh storage, including for zero corrections.
    """
    layout = validate_factors(B, A)
    _cpu_half_matrix(base, 'base')
    if list(base.shape) != layout['shape']:
        raise ValueError('Base and factor dimensions disagree')
    if not torch.isfinite(base).all():
        raise ValueError('Nonfinite base weight')
    if torch.is_autocast_enabled('cpu'):
        raise ValueError('CPU autocast must be disabled for the declared FP32 merge')
    merged = (base.detach().float() + B.float() @ A.float()).half()
    if not torch.isfinite(merged).all():
        raise ValueError('Merged weight overflows FP16')
    return merged
