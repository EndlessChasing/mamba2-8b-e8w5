"""Standalone E8/W5 codecs. E8 primitives derive from GPL-3.0 QuIP#.

Projection quantization is lossy. Raw-file and Huffman roundtrips are exact.
"""
import ast
import functools
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn


def load_reference_primitives(source=None):
    source = Path(source) if source else Path(__file__).resolve().parents[1] / "third_party/quip-sharp"
    provenance = source / "SOURCE.json"
    if provenance.exists():
        for name, digest in json.loads(provenance.read_text())["files"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"QuIP# source hash mismatch: {name}")
    book_path = source / "lib/codebook/latticee8_padded12.py"
    tree = ast.parse(book_path.read_text())
    # Keep original definitions and codebook construction; the CUDA extension
    # is only used by inference methods which this experiment does not call.
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ClassDef) and node.name != "E8P12_codebook":
            continue
        keep.append(node)
    scope = {"torch": torch, "nn": nn, "math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(book_path), "exec"), scope)
    cb = scope["E8P12_codebook"]()
    algo_path = source / "lib/algo/quip.py"
    tree = ast.parse(algo_path.read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "LDLQ_buffered_lowmem"]
    algo = {"torch": torch, "utils": SimpleNamespace(clean=lambda: None)}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(algo_path), "exec"), algo)
    return cb, algo["LDLQ_buffered_lowmem"]

@functools.lru_cache(maxsize=32)
def rotation_factors(n, device):
    p = n & -n
    k = n // p
    h = torch.ones((1, 1), device=device)
    while h.shape[0] < p:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= math.sqrt(p)
    # 8B in_proj has 18560 rows = 145 * 128. Computing the DCT's
    # trigonometric arguments in FP32 gives an ~8e-6 inverse error for k=145.
    # Construct the small factor in FP64, then retain FP32 execution/storage.
    i = torch.arange(k, device=device, dtype=torch.float64)
    dct = torch.cos(math.pi / k * (i[None, :] + .5) * i[:, None]) * math.sqrt(2 / k)
    dct[0] /= math.sqrt(2)
    return dct.float(), h

def rotate_last(x, inverse=False):
    """Row-vector forward x @ (DCT.T kron H); inverse x @ (DCT kron H)."""
    n = x.shape[-1]
    c, h = rotation_factors(n, str(x.device))
    y = x.reshape(-1, c.shape[0], h.shape[0]) @ h
    y = torch.matmul(c.T if inverse else c, y)
    return y.reshape_as(x)

def block_ldl(h, block=8):
    n = h.shape[0]
    chol = torch.linalg.cholesky(h)
    groups = n // block
    diagonal = torch.diagonal(chol.reshape(groups, block, groups, block), dim1=0, dim2=2).permute(2, 0, 1)
    d = diagonal @ diagonal.transpose(-1, -2)
    inverse = torch.linalg.inv(diagonal)
    l = torch.einsum("nkb,kbc->nkc", chol.reshape(n, groups, block), inverse).reshape(n, n)
    return l, d

class AxisResidualCodebook:
    """E8 base plus one of 16 signed coordinate corrections per 8-vector."""
    codesz = 8
    idx_dtype = torch.int64

    def __init__(self, base, amplitude):
        self.base = base
        self.amplitude = float(amplitude)

    def quantize(self, x, return_idx=True, **kwargs):
        init, idx = self.base.quantize(x)
        residual = x - init
        axis = residual.abs().argmax(-1)
        negative = residual.gather(1,axis[:,None]).squeeze(1) < 0
        values = init.clone()
        values.scatter_add_(1,axis[:,None],((1-2*negative.float())*self.amplitude)[:,None])
        codes = (idx.long()<<4) | (axis*2+negative.long())
        return (values,codes) if return_idx else values

def decode_codes(indices, base, amplitude=0.0):
    if amplitude:
        values=base.grid[(indices>>4).long()].clone()
        small=indices&15
        values.scatter_add_(-1,(small>>1)[...,None],((1-2*(small&1).float())*amplitude)[...,None])
        return values
    return base.grid[indices.long()]

def vector_quantize(weight, hessian, cb, ldlq, seed, damping=.01, scale_override=.9,
                    tune_iters=2, use_feedback=True, residual_bits=0):
    w = weight.float()
    m, n = w.shape
    h = hessian.to(device=w.device, dtype=torch.float32).clone()
    h /= h.diag().mean()
    h.diagonal().add_(damping)
    # Diagonal preconditioning moves activation outliers into stored weights;
    # the inverse transform needs this one FP16 value per input channel.
    balance = (h.diag().clamp_min(1e-8) / w.square().sum(0).clamp_min(1e-8)).pow(.25)
    balance = balance.half().float()
    h = h / balance[:, None] / balance[None, :]
    wscaled = w * balance[None, :]
    gen = torch.Generator(device=w.device).manual_seed(seed)
    su = (torch.randint(0, 2, (n,), generator=gen, device=w.device) * 2 - 1).float()
    sv = (torch.randint(0, 2, (m,), generator=gen, device=w.device) * 2 - 1).float()
    wr = rotate_last(rotate_last(wscaled * su).T * sv).T
    hr = rotate_last(rotate_last(h * su).T * su)
    hr = (hr + hr.T) * .5
    scale = (wr.square().mean().sqrt() / scale_override).float()
    wr /= scale
    amplitude=0.0
    quantizer=cb
    if residual_bits == 4:
        # Fit only to original weights; no held-out language tokens are used.
        samples=wr.reshape(-1,8)[::max(1,wr.numel()//8//65536)]
        initial=cb.quantize(samples,return_idx=False)
        amplitude=float((samples-initial).abs().amax(-1).mean())
        quantizer=AxisResidualCodebook(cb,amplitude)
    elif residual_bits:
        raise ValueError(residual_bits)
    if use_feedback:
        l, d = block_ldl(hr)
        q, idx = ldlq(wr, hr, l, d, quantizer,
                     SimpleNamespace(quip_tune_iters=tune_iters, resid_scale_override=-1), buf_cols=128)
    else:
        q = torch.empty_like(wr)
        idx = torch.empty((m, n//8), device=w.device, dtype=torch.int64)
        for start in range(0, m, 64):
            part, codes = quantizer.quantize(wr[start:start+64].reshape(-1, 8))
            q[start:start+64] = part.reshape(-1, n)
            idx[start:start+64] = codes.reshape(-1, n//8)
    assert idx.min() >= 0 and idx.max() < 2**(16+residual_bits)
    # Decode with only stored values and the shared codebook; check all codes.
    decoded = decode_codes(idx,cb,amplitude).reshape_as(q)
    if not torch.equal(q, decoded):
        raise RuntimeError("E8 index roundtrip differs from quantizer output")
    # Inverse left transform, then inverse right transform.
    left_restored = (rotate_last((decoded * scale).T, inverse=True) * sv).T
    restored = (rotate_last(left_restored, inverse=True) * su) / balance[None, :]
    restored = restored.half()
    delta = restored.float() - w
    original_h = hessian.to(w.device)
    proxy_num = ((delta @ original_h) * delta).sum()
    proxy_den = ((w @ original_h) * w).sum()
    info = {"shape": [m, n], "index_bits": 16+residual_bits, "values_per_index": 8,
            "relative_weight_error": float(delta.norm() / w.norm()),
            "relative_calibration_output_error": float((proxy_num / proxy_den).clamp_min(0).sqrt()),
            "feedback": use_feedback, "damping": damping, "scale_override": scale_override,
            "tune_iters": tune_iters, "seed": seed,
            "axis_residual_amplitude":amplitude,
            "payload_bytes": m*n*(16+residual_bits)//64 + 2*n + (m+n+7)//8 + 4}
    payload = {"indices": idx.to(torch.int32).cpu(), "balance": balance.half().cpu(),
               "input_sign": su.to(torch.int8).cpu(), "output_sign": sv.to(torch.int8).cpu(),
               "scale": scale.cpu(), "axis_residual_amplitude":amplitude}
    return restored, info, payload

def decode_e8(payload, cb, device="cuda"):
    idx = payload["indices"].to(device=device, dtype=torch.long)
    decoded = decode_codes(idx,cb,payload.get("axis_residual_amplitude",0.0)).reshape(idx.shape[0], -1)
    scale = payload["scale"].to(device)
    su = payload["input_sign"].to(device=device, dtype=torch.float32)
    sv = payload["output_sign"].to(device=device, dtype=torch.float32)
    balance = payload["balance"].to(device=device, dtype=torch.float32)
    left = (rotate_last((decoded * scale).T, inverse=True) * sv).T
    return ((rotate_last(left, inverse=True) * su) / balance[None, :]).half()

def write_e8(path, payload, info):
    header = json.dumps({"format": "mamba-e8-dct-hadamard-v1", "shape": info["shape"],
                         "index_order": "row-major", "rotation": "dct2-kron-hadamard; maximal power-of-two factor",
                         "axis_residual_amplitude":payload.get("axis_residual_amplitude",0.0)},
                        separators=(",", ":")).encode()
    with Path(path).open("wb") as f:
        f.write(b"ME8HD001" + struct.pack("<I", len(header)) + header)
        codes=payload["indices"].numpy()
        if payload.get("axis_residual_amplitude",0.0):
            f.write((codes>>4).astype("<u2").tobytes())
            small=(codes&15).astype(np.uint8).flatten()
            f.write((small[::2] | (small[1::2]<<4)).tobytes())
        else:
            f.write(codes.astype("<u2").tobytes())
        f.write(payload["balance"].numpy().astype("<f2").tobytes())
        for key in ("input_sign", "output_sign"):
            f.write(np.packbits(payload[key].numpy() < 0, bitorder="little").tobytes())
        f.write(payload["scale"].numpy().astype("<f4").tobytes())

def read_e8(path):
    with Path(path).open("rb") as f:
        assert f.read(8) == b"ME8HD001"
        length, = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(length))
        m, n = header["shape"]
        idx = torch.from_numpy(np.frombuffer(f.read(m*n//4), dtype="<u2").astype(np.int32)).reshape(m, n//8)
        amplitude=header.get("axis_residual_amplitude",0.0)
        if amplitude:
            raw=np.frombuffer(f.read(m*n//16),dtype=np.uint8)
            small=np.empty(m*n//8,dtype=np.int32)
            small[::2],small[1::2]=raw&15,raw>>4
            idx=(idx<<4)|torch.from_numpy(small).reshape_as(idx)
        balance = torch.from_numpy(np.frombuffer(f.read(n*2), dtype="<f2").copy())
        signs = []
        for size in (n, m):
            bits = np.unpackbits(np.frombuffer(f.read((size+7)//8), dtype=np.uint8), bitorder="little")[:size]
            signs.append(torch.from_numpy(1 - 2 * bits.astype(np.int8)))
        scale = torch.from_numpy(np.frombuffer(f.read(4), dtype="<f4").copy()).squeeze()
        assert not f.read(1), "unexpected trailing bytes"
    return {"indices": idx, "balance": balance, "input_sign": signs[0], "output_sign": signs[1], "scale": scale,
            "axis_residual_amplitude":amplitude}


def uniform_header(path):
    """Validate a group-128 raw file and return (header, code offset)."""
    path = Path(path)
    with path.open("rb") as f:
        if f.read(8) != b"MEQG0128":
            raise ValueError("Not a uniform weight file")
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError("Truncated uniform header")
        length, = struct.unpack("<I", raw)
        if length > 65536:
            raise ValueError("Oversized uniform header")
        header = json.loads(f.read(length))
    shape = header["shape"]
    if len(shape) != 2 or min(shape) <= 0 or shape[1] % 128:
        raise ValueError("Uniform files require a matrix with columns divisible by 128")
    if header["group"] != 128 or header["bits"] not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("Unsupported uniform scheme")
    count = math.prod(shape)
    expected = 12 + length + count * header["bits"] // 8 + count // 128 * 2
    if path.stat().st_size != expected:
        raise ValueError("Uniform file length mismatch")
    return header, 12 + length


def pack_uniform_codes(codes, bits):
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1, 8)
    if bits not in range(2, 9) or np.any(codes >= (1 << bits)):
        raise ValueError("Invalid uniform codes")
    accum = np.zeros(len(codes), dtype=np.uint64)
    for i in range(8):
        accum |= codes[:, i].astype(np.uint64) << (i * bits)
    packed = np.empty((len(codes), bits), dtype=np.uint8)
    for i in range(bits):
        packed[:, i] = (accum >> (i * 8)).astype(np.uint8)
    return packed.tobytes()


def unpack_uniform_codes(raw, bits):
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bits)
    accum = np.zeros(len(packed), dtype=np.uint64)
    for i in range(bits):
        accum |= packed[:, i].astype(np.uint64) << (i * 8)
    code = np.empty((len(packed), 8), dtype=np.uint8)
    for i in range(8):
        code[:, i] = ((accum >> (i * bits)) & ((1 << bits) - 1)).astype(np.uint8)
    return code.reshape(-1)


def write_uniform(path, weight, bits=5, device=None, rows_per_chunk=256, chunk_rows=None):
    """Write bounded row batches, retaining byte compatibility with MEQG0128.

    The input may be a CPU mmap tensor; only one row batch is converted to
    FP32. Codes and scales are written at fixed offsets without temporary
    full-vocabulary arrays. The finished file is atomically renamed.
    """
    if weight.ndim != 2 or weight.shape[1] % 128 or bits not in range(2, 9):
        raise ValueError("Expected a matrix with group-128 columns and 2..8 bits")
    chunk_rows = rows_per_chunk if chunk_rows is None else chunk_rows
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    header = json.dumps({"shape": list(weight.shape), "bits": bits,
                         "group": 128, "scale": "fp16"}).encode()
    start = 12 + len(header)
    count = weight.numel()
    code_bytes = count * bits // 8
    columns = weight.shape[1]
    limit = (1 << (bits - 1)) - 1
    with tmp.open("w+b") as f, torch.inference_mode():
        f.write(b"MEQG0128" + struct.pack("<I", len(header)) + header)
        f.truncate(start + code_bytes + count // 128 * 2)
        for row in range(0, weight.shape[0], chunk_rows):
            x = weight[row:row + chunk_rows].detach().to(device=device or weight.device, dtype=torch.float32).reshape(-1, 128)
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite weights")
            scale = (x.abs().amax(-1, keepdim=True) / limit).half()
            scale = torch.where(scale == 0, 1, scale)
            if not torch.isfinite(scale).all():
                raise ValueError("Uniform scale overflows FP16")
            codes = (torch.round(x / scale.float()).clamp(-limit, limit) + limit)
            codes = codes.to(torch.uint8).cpu().numpy().reshape(-1)
            f.seek(start + row * columns * bits // 8)
            f.write(pack_uniform_codes(codes, bits))
            f.seek(start + code_bytes + row * columns // 128 * 2)
            f.write(scale.cpu().numpy().astype("<f2").tobytes())
        f.flush()
        os.fsync(f.fileno())
    uniform_header(tmp)
    error_squared = norm_squared = 0.0
    with torch.inference_mode():
        for row, decoded in iter_uniform(tmp, device=device or weight.device, chunk_rows=chunk_rows):
            original = weight[row:row + len(decoded)].detach().to(device=decoded.device, dtype=torch.float32)
            groups = original.reshape(-1, 128)
            scale = (groups.abs().amax(-1, keepdim=True) / limit).half().float()
            scale = torch.where(scale == 0, 1, scale)
            expected = (torch.round(groups / scale).clamp(-limit, limit) * scale).half().reshape_as(decoded)
            if not torch.equal(expected, decoded):
                raise ValueError(f"Uniform disk roundtrip differs at row {row}")
            error_squared += float((decoded.float() - original).double().square().sum())
            norm_squared += float(original.double().square().sum())
    os.replace(tmp, path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            digest.update(data)
    return {"shape": list(weight.shape), "bits": bits, "group": 128,
            "rows_per_chunk": chunk_rows, "actual_file_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(), "disk_roundtrip_fp16_equal": True,
            "relative_weight_error": math.sqrt(error_squared / norm_squared) if norm_squared else 0.0}


def read_uniform_rows(path, start, end, device="cpu"):
    """Decode exactly [start,end) rows, with group boundaries preserved."""
    header, offset = uniform_header(path)
    rows, columns = header["shape"]
    if not (0 <= start <= end <= rows):
        raise ValueError("Invalid row interval")
    bits = header["bits"]
    count = (end - start) * columns
    if not count:
        return torch.empty((0, columns), dtype=torch.float16, device=device)
    with Path(path).open("rb") as f:
        f.seek(offset + start * columns * bits // 8)
        raw = f.read(count * bits // 8)
        f.seek(offset + rows * columns * bits // 8 + start * columns // 128 * 2)
        scales = np.frombuffer(f.read(count // 128 * 2), dtype="<f2").copy()
    codes = unpack_uniform_codes(raw, bits)
    values = torch.from_numpy(codes).to(device=device, dtype=torch.float32).reshape(-1, 128)
    values -= (1 << (bits - 1)) - 1
    scales = torch.from_numpy(scales).to(device=device, dtype=torch.float32)
    return (values * scales[:, None]).half().reshape(end - start, columns)


def iter_uniform(path, device="cpu", chunk_rows=512):
    """Yield (first_row, FP16 row tensor) without full-vocabulary expansion."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    header, _ = uniform_header(path)
    for first in range(0, header["shape"][0], chunk_rows):
        yield first, read_uniform_rows(path, first,
            min(first + chunk_rows, header["shape"][0]), device)


def read_uniform(path, device="cpu"):
    """Convenience full decode. Prefer iter_uniform for the 256K vocabulary."""
    header, _ = uniform_header(path)
    result = torch.empty(header["shape"], dtype=torch.float16, device=device)
    for first, values in iter_uniform(path, device):
        result[first:first + len(values)].copy_(values)
    return result
