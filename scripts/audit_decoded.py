#!/usr/bin/env python3
"""Cross-process audit of every FP16 tensor loaded from a completed E8/W5 package.

No original NVIDIA checkpoint is loaded. Projection hashes come from the
quantization manifest. W5 vocabulary rows are independently unpacked on CPU
using bit-plane extraction and compared with the loaded model. Remaining
parameters are compared against other_fp16.pt. This is an integrity check,
not a PPL or recall evaluation and not a compressed-residency benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.runtime import MODEL_CONFIG, load_quantized_model, sha256_file


def fp16_bytes(value):
    if value.dtype != torch.float16:
        raise ValueError(f"Expected FP16 tensor, found {value.dtype}")
    return value.detach().cpu().contiguous().numpy().astype("<f2", copy=False).tobytes(order="C")


def tensor_sha_fp16(value, rows_per_chunk=128):
    """Hash canonical row-major FP16 bytes without copying a whole vocabulary."""
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    digest = hashlib.sha256()
    if value.ndim == 0:
        digest.update(fp16_bytes(value))
    else:
        for first in range(0, len(value), rows_per_chunk):
            digest.update(fp16_bytes(value[first:first + rows_per_chunk]))
    return digest.hexdigest()


def independent_w5_rows(path, rows_per_chunk=128):
    """Decode W5 via individual little-endian bits, independent of codec.py.

    The production codec assembles eight codes into a uint64 accumulator.
    This audit expands the bit stream into groups of five bits instead, then
    performs FP32 multiplication with stored FP16 scales before FP16 rounding.
    """
    path = Path(path)
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    with path.open("rb") as stream:
        header_bytes = stream.read(12)
        if len(header_bytes) != 12:
            raise ValueError("Truncated W5 header")
        magic, length = struct.unpack("<8sI", header_bytes)
        if magic != b"MEQG0128" or not 0 < length < 65536:
            raise ValueError("Invalid W5 format")
        header = json.loads(stream.read(length))
        if header.get("bits") != 5 or header.get("group") != 128 or header.get("scale") != "fp16":
            raise ValueError("Audit supports the frozen group-128 W5 format only")
        shape = header["shape"]
        if len(shape) != 2 or min(shape) <= 0 or shape[1] % 128:
            raise ValueError("Invalid W5 row geometry")
        rows, columns = shape
        offset = 12 + length
        code_bytes = rows * columns * 5 // 8
        if path.stat().st_size != offset + code_bytes + rows * columns // 128 * 2:
            raise ValueError("W5 file size mismatch")
        powers = np.array([1, 2, 4, 8, 16], dtype=np.uint8)
        for first in range(0, rows, rows_per_chunk):
            count_rows = min(rows_per_chunk, rows - first)
            count = count_rows * columns
            stream.seek(offset + first * columns * 5 // 8)
            raw = stream.read(count * 5 // 8)
            stream.seek(offset + code_bytes + first * columns // 128 * 2)
            raw_scales = stream.read(count // 128 * 2)
            if len(raw) != count * 5 // 8 or len(raw_scales) != count // 128 * 2:
                raise ValueError("Truncated W5 row batch")
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little").reshape(count, 5)
            codes = (bits * powers).sum(axis=1, dtype=np.uint16)
            if np.any(codes > 30):
                raise ValueError("Unused W5 code 31 found")
            scales = np.frombuffer(raw_scales, dtype="<f2").astype(np.float32)
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("Nonpositive or nonfinite W5 scale")
            values = ((codes.astype(np.float32) - 15).reshape(-1, 128) * scales[:, None])
            yield first, values.astype("<f2").reshape(count_rows, columns)


@torch.inference_mode()
def audit_loaded_model(model, manifest, raw_dir, rows_per_chunk=128):
    """Audit an already-loaded model; the CLI additionally fixes 8B architecture."""
    directory = Path(raw_dir)
    parameters = dict(model.named_parameters())
    config = manifest["model_config"]
    layers = config["n_layer"]
    expected_labels = {f"layer{i}.{which}" for i in range(layers) for which in ("in_proj", "out_proj")}
    if set(manifest["matrices"]) != expected_labels:
        raise ValueError("Projection manifest coverage mismatch")
    if set(manifest["parameter_mapping"]) != set(parameters):
        raise ValueError("Parameter mapping does not cover exactly the loaded model")
    covered, records = set(), []
    for index in range(layers):
        for which in ("in_proj", "out_proj"):
            label = f"layer{index}.{which}"
            name = f"backbone.layers.{index}.mixer.{which}.weight"
            entry = manifest["matrices"][label]
            value = parameters[name]
            if entry["source_key"] != name or list(value.shape) != entry["shape"]:
                raise ValueError(f"Projection identity/shape mismatch: {label}")
            if manifest["parameter_mapping"][name] != f"{label}.e8":
                raise ValueError(f"Projection storage mapping mismatch: {label}")
            actual = tensor_sha_fp16(value, rows_per_chunk)
            if actual != entry["decoded_fp16_sha256"]:
                raise ValueError(f"Loaded projection differs from quantizer FP16 result: {label}; actual={actual}")
            records.append({"name": name, "family": "e8", "shape": list(value.shape),
                            "parameters": value.numel(), "decoded_fp16_sha256": actual,
                            "quantizer_hash_equal": True})
            covered.add(name)
        if (index + 1) % 8 == 0 or index + 1 == layers:
            print(f"[loaded E8 SHA] {2 * (index + 1)}/{2 * layers}", flush=True)
    embedding = parameters["backbone.embedding.weight"]
    head = parameters["lm_head.weight"]
    if embedding.data_ptr() == head.data_ptr():
        raise ValueError("Independent input embedding and output head were tied")
    for label, name in (("embedding", "backbone.embedding.weight"), ("lm_head", "lm_head.weight")):
        value = parameters[name]
        if value.dtype != torch.float16:
            raise ValueError(f"Vocabulary is not FP16: {name}")
        if manifest["parameter_mapping"][name] != label + ".uniform":
            raise ValueError(f"Vocabulary storage mapping mismatch: {label}")
        expected_entry = manifest["vocabularies"][label]
        if expected_entry["source_key"] != name or list(value.shape) != expected_entry["shape"]:
            raise ValueError(f"Vocabulary identity/shape mismatch: {name}")
        loaded_hash, independent_hash = hashlib.sha256(), hashlib.sha256()
        next_row = comparisons = 0
        for first, decoded in independent_w5_rows(directory / (label + ".uniform"), rows_per_chunk):
            if first != next_row or decoded.shape[1] != value.shape[1]:
                raise ValueError(f"W5 decoder row coverage mismatch: {name}")
            loaded = fp16_bytes(value[first:first + len(decoded)])
            expected = decoded.tobytes(order="C")
            if loaded != expected:
                raise ValueError(f"Independent W5 decode differs from loaded model: {name}, row {first}")
            loaded_hash.update(loaded)
            independent_hash.update(expected)
            next_row += len(decoded)
            comparisons += 1
        if next_row != value.shape[0] or loaded_hash.digest() != independent_hash.digest():
            raise ValueError(f"Incomplete independent vocabulary comparison: {name}")
        records.append({"name": name, "family": "w5", "shape": list(value.shape),
            "parameters": value.numel(), "rows_compared": next_row, "batches_compared": comparisons,
            "decoded_fp16_sha256": loaded_hash.hexdigest(),
            "independent_cpu_decoded_fp16_sha256": independent_hash.hexdigest(),
            "independent_cpu_decode_equal": True})
        covered.add(name)
        print(f"[loaded W5 independent decode] {name}: {next_row} rows", flush=True)
    other = torch.load(directory / "other_fp16.pt", map_location="cpu", weights_only=True)
    for name, expected in other.items():
        if name in covered or name not in parameters:
            raise ValueError(f"Overlapping or unknown FP16 tensor: {name}")
        if manifest["parameter_mapping"][name] != "other_fp16.pt":
            raise ValueError(f"FP16 storage mapping mismatch: {name}")
        value = parameters[name]
        if value.shape != expected.shape:
            raise ValueError(f"FP16 shape mismatch: {name}")
        actual_hash = tensor_sha_fp16(value, rows_per_chunk)
        expected_hash = tensor_sha_fp16(expected, rows_per_chunk)
        if actual_hash != expected_hash:
            raise ValueError(f"Loaded FP16 tensor differs from package: {name}")
        records.append({"name": name, "family": "fp16", "shape": list(value.shape),
                        "parameters": value.numel(), "decoded_fp16_sha256": actual_hash,
                        "stored_fp16_hash_equal": True})
        covered.add(name)
    if covered != set(parameters):
        raise ValueError(f"Unverified loaded parameters: {set(parameters) - covered}")
    parameter_count = sum(p.numel() for p in parameters.values())
    if parameter_count != manifest["total_parameter_count"]:
        raise ValueError("Total parameter count differs from package")
    return {"complete": True, "verified_tensors": len(records), "parameter_count": parameter_count,
            "e8_projections_verified": 2 * layers, "w5_vocabularies_verified": 2,
            "fp16_tensors_verified": len(other), "all_parameters_covered": True,
            "independent_vocabularies": True, "tensor_byte_encoding": "IEEE754 FP16 little-endian row-major",
            "tensors": records}


def self_test():
    """Small CPU test only; not a completed 8B model validation."""
    from torch import nn
    from mamba_e8w5.codec import write_uniform, read_uniform
    torch.set_num_threads(2)
    torch.manual_seed(44)
    with tempfile.TemporaryDirectory(prefix="e8w5-loaded-audit-fixture-") as temporary:
        directory = Path(temporary)
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.layers = nn.ModuleList([nn.Module()])
        model.backbone.layers[0].mixer = nn.Module()
        mixer = model.backbone.layers[0].mixer
        mixer.in_proj, mixer.out_proj = nn.Linear(128, 128, bias=False).half(), nn.Linear(128, 128, bias=False).half()
        model.backbone.embedding = nn.Embedding(32, 128).half()
        model.lm_head = nn.Linear(128, 32, bias=False).half()
        model.backbone.norm_f = nn.LayerNorm(128, elementwise_affine=True, bias=False).half()
        matrices, vocabularies, mapping = {}, {}, {}
        for which in ("in_proj", "out_proj"):
            name = f"backbone.layers.0.mixer.{which}.weight"
            value = dict(model.named_parameters())[name]
            matrices[f"layer0.{which}"] = {"source_key": name, "shape": list(value.shape),
                "decoded_fp16_sha256": tensor_sha_fp16(value)}
            mapping[name] = f"layer0.{which}.e8"
        for label, name in (("embedding", "backbone.embedding.weight"), ("lm_head", "lm_head.weight")):
            value = dict(model.named_parameters())[name]
            weight = torch.randn_like(value)
            weight[0].zero_()
            write_uniform(directory / (label + ".uniform"), weight, device="cpu", rows_per_chunk=7)
            with torch.no_grad():
                value.copy_(read_uniform(directory / (label + ".uniform")))
            vocabularies[label] = {"source_key": name, "shape": list(value.shape)}
            mapping[name] = label + ".uniform"
        other = {"backbone.norm_f.weight": model.backbone.norm_f.weight.detach().clone()}
        torch.save(other, directory / "other_fp16.pt")
        mapping["backbone.norm_f.weight"] = "other_fp16.pt"
        manifest = {"model_config": {"n_layer": 1}, "matrices": matrices, "vocabularies": vocabularies,
                    "parameter_mapping": mapping, "total_parameter_count": sum(p.numel() for p in model.parameters())}
        result = audit_loaded_model(model, manifest, directory, rows_per_chunk=11)
        with torch.no_grad():
            model.lm_head.weight[3, 7] += 1
        try:
            audit_loaded_model(model, manifest, directory, rows_per_chunk=11)
        except ValueError as error:
            if "Independent W5 decode differs" not in str(error):
                raise
        else:
            raise AssertionError("Corrupted loaded vocabulary was not detected")
        return {"complete": True, "fixture_only": True, "cpu_self_test": "passed",
                "fixture_verified_tensors": result["verified_tensors"],
                "independent_w5_bit_decoder_and_corruption_detection": True}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--quip-source", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Run only small CPU fixtures")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        if args.report:
            write_report(args.report, result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if not args.raw_dir or not args.report:
        parser.error("--raw-dir and --report are required outside --self-test")
    if args.rows_per_chunk <= 0:
        parser.error("--rows-per-chunk must be positive")
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    manifest_path = args.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or manifest["model_config"] != MODEL_CONFIG or len(manifest["matrices"]) != 112:
        raise ValueError("Expected a complete audited 56-layer Mamba-2 8B package")
    initial = {"complete": False, "raw_dir": str(args.raw_dir), "manifest_sha256": sha256_file(manifest_path),
               "device": args.device, "dtype": "float16", "rows_per_chunk": args.rows_per_chunk,
               "original_checkpoint_loaded": False, "audit_source_sha256": sha256_file(__file__),
               "runtime_source_sha256": sha256_file(ROOT / "mamba_e8w5/runtime.py"),
               "codec_source_sha256": sha256_file(ROOT / "mamba_e8w5/codec.py"),
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "note": "Exact loaded-weight integrity only. Full FP16 reference residency, not compressed inference residency or model-quality validation."}
    write_report(args.report, initial)
    cuda = str(args.device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    try:
        model = load_quantized_model(args.raw_dir, device=args.device, dtype=torch.float16,
                                     quip_source=args.quip_source)
        loaded_at = time.monotonic()
        result = audit_loaded_model(model, manifest, args.raw_dir, args.rows_per_chunk)
        package_receipt = getattr(model, "_package_receipt", None)
        if not package_receipt or package_receipt["manifest_sha256"] != initial["manifest_sha256"]:
            raise ValueError("Runtime package receipt does not match the audited manifest")
        if package_receipt["parameter_count"] != result["parameter_count"]:
            raise ValueError("Runtime package receipt parameter count mismatch")
        if sha256_file(manifest_path) != initial["manifest_sha256"]:
            raise ValueError("Package manifest changed during the audit")
        initial.update(result, package_receipt=package_receipt,
                       load_seconds=loaded_at-started, audit_seconds=time.monotonic()-loaded_at,
                       elapsed_seconds=time.monotonic()-started)
        if cuda:
            initial.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                           gpu_name=torch.cuda.get_device_name(args.device))
        write_report(args.report, initial)
        print(json.dumps({key: initial[key] for key in ("complete", "manifest_sha256", "parameter_count",
              "verified_tensors", "e8_projections_verified", "w5_vocabularies_verified", "fp16_tensors_verified",
              "load_seconds", "audit_seconds", "elapsed_seconds")}, indent=2), flush=True)
    except Exception as error:
        initial.update(complete=False, error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic()-started)
        write_report(args.report, initial)
        raise


if __name__ == "__main__":
    main()
