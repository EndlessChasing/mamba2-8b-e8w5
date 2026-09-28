#!/usr/bin/env python3
"""Predeclared six-matrix, train-only E8 2-versus-8-sweep screen.

Calls the frozen codec without modifications. Baseline decoded hashes must
match the completed parent package. No evaluation tokens, model forwards,
vocabulary changes, entropy coding, or full-model expansion are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.codec import (load_reference_primitives, vector_quantize,
                             write_e8, read_e8, decode_e8)
from mamba_e8w5.runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, load_source_state

LABELS = tuple(f"layer{layer}.{part}" for layer in (0, 18, 55)
               for part in ("in_proj", "out_proj"))
GATE = {"median_relative_mse_reduction_at_least": .01,
        "individual_relative_mse_reduction_at_least": .005,
        "minimum_individual_passes": 4,
        "maximum_individual_regression": .001}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def output_metrics(weight, decoded, hessian, rows=256):
    """Compute Tr(delta H delta.T), H=X.T X / T, in original coordinates.

    FP32 GEMM and FP64 reduction, without damping or balance. Division by the
    output-channel count gives the mean squared projection-output error over
    the exact calibration inputs represented by the Hessian.
    """
    numerator = torch.zeros((), device=weight.device, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for first in range(0, weight.shape[0], rows):
        reference = weight[first:first + rows].float()
        delta = decoded[first:first + rows].float() - reference
        numerator += ((delta @ hessian) * delta).sum(dtype=torch.float64)
        denominator += ((reference @ hessian) * reference).sum(dtype=torch.float64)
    num, den = float(numerator), float(denominator)
    if not (num >= 0 and den > 0):
        raise ValueError("Invalid calibration quadratic objective")
    return {"original_hessian_weighted_squared_error": num,
            "original_hessian_reference_squared_output": den,
            "mean_output_squared_error": num / weight.shape[0],
            "mean_reference_output_squared": den / weight.shape[0],
            "relative_output_mse": num / den}


def evaluate_gate(reductions):
    median = statistics.median(reductions)
    passes = sum(value >= GATE["individual_relative_mse_reduction_at_least"] for value in reductions)
    worst = min(reductions)
    return {"median_relative_mse_reduction": median,
            "individual_pass_count": passes,
            "worst_relative_mse_reduction": worst,
            "pass": (median >= GATE["median_relative_mse_reduction_at_least"]
                     and passes >= GATE["minimum_individual_passes"]
                     and worst >= -GATE["maximum_individual_regression"])}


def self_test():
    torch.manual_seed(318)
    x, w = torch.randn(71, 16), torch.randn(11, 16)
    q = (w * 7).round() / 7
    h = x.T @ x / len(x)
    measured = output_metrics(w, q, h, rows=3)
    expected = float(((x.double() @ (q - w).double().T).square()).mean())
    assert abs(measured["mean_output_squared_error"] / expected - 1) < 1e-6
    assert evaluate_gate([.01, .02, .03, .04, .005, -.0009])["pass"]
    assert not evaluate_gate([.01, .02, .03, .04, .005, -.0011])["pass"]
    assert not evaluate_gate([.001] * 6)["pass"]
    assert not evaluate_gate([.0001, .0001, .0001, .02, .03, .04])["pass"]
    print(json.dumps({"self_test": "pass", "explicit_calibration_output_mse": expected,
                      "hessian_output_mse": measured["mean_output_squared_error"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--hessian-dir", type=Path)
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("source_dir", "hessian_dir", "parent_dir", "out_dir")):
        parser.error("All four directories are required unless --self-test is used")
    if args.out_dir.exists():
        raise FileExistsError("Use a new screen directory; existing artifacts are preserved")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    parent_path, calibration_path = args.parent_dir / "manifest.json", args.hessian_dir / "manifest.json"
    parent, calibration = json.loads(parent_path.read_text()), json.loads(calibration_path.read_text())
    assert parent["complete"] and len(parent["matrices"]) == 112
    assert parent["model_config"] == calibration["model_config"] == MODEL_CONFIG
    assert calibration["complete"] and calibration["evaluation_data_used"] is False
    assert calibration["calibration_split"] == "train"
    assert parent["source_checkpoint_sha256"] == calibration["source_checkpoint_sha256"] == SOURCE_CHECKPOINT_SHA256
    assert parent["binding"]["hessian_manifest_sha256"] == sha(calibration_path)
    assert parent["binding"]["scale"] == .9 and parent["binding"]["damping"] == .01
    assert parent["binding"]["tune_iters"] == 2
    for name, digest in parent["binding"]["code_sha256"].items():
        assert sha(ROOT / "mamba_e8w5" / name) == digest, name
    for name, digest in parent["binding"]["quip_sha256"].items():
        assert sha(ROOT / "third_party/quip-sharp" / name) == digest, name
    binding = {"source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
               "tokenizer_sha256": parent["tokenizer_sha256"],
               "hessian_manifest_sha256": sha(calibration_path),
               "codec_source_sha256": sha(ROOT / "mamba_e8w5/codec.py"),
               "runtime_source_sha256": sha(ROOT / "mamba_e8w5/runtime.py"),
               "script_source_sha256": sha(__file__),
               "quip_sha256": parent["binding"]["quip_sha256"],
               "scale": .9, "damping": .01, "tune_iters": 8}
    report = {"format": "MAMBA2_E8W5_REFINEMENT_SCREEN_V1", "complete": False,
              "parent_manifest_sha256": sha(parent_path), "binding": binding,
              "predeclared_labels": list(LABELS), "sweeps": [2, 8], "expansion_gate": GATE,
              "calibration_split": "train", "evaluation_data_used": False,
              "metric": "Original undamped H=X.T X/T; projection-output squared error, FP32 GEMM with FP64 accumulation",
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
              "matrices": {}, "comparisons": {},
              "limitations": ["Six matrices are a bounded screen, not a full-model repair.",
                              "A calibration MSE gain does not establish PPL or recall improvement.",
                              "Raw E8 byte equality does not imply identical Huffman file size."]}
    overlay = {"format": "MAMBA2_E8W5_REFINEMENT_OVERLAY_V1", "complete": False,
               "scope": "six_matrix_screen", "screen_complete": False,
               "parent_manifest_sha256": sha(parent_path), "model_config": MODEL_CONFIG,
               "binding": binding, "matrices": {}, "files": {}}
    for sweeps in (2, 8):
        (args.out_dir / f"tune{sweeps}").mkdir(parents=True)
    write_json(args.out_dir / "report.json", report)
    write_json(args.out_dir / "tune8/manifest.json", overlay)
    print(json.dumps({"predeclared": LABELS, "gate": GATE, "binding": binding}), flush=True)
    source = load_source_state(args.source_dir)
    cb, ldlq = load_reference_primitives()
    cb = cb.cuda()
    with torch.inference_mode():
        for label in LABELS:
            layer_str, part = label.split(".")
            layer = int(layer_str[5:])
            index = 2 * layer + (part == "out_proj")
            source_key = f"backbone.layers.{layer}.mixer.{part}.weight"
            original = parent["matrices"][label]
            hp = args.hessian_dir / calibration["matrices"][label]["file"]
            hsha = sha(hp)
            assert hsha == original["hessian_sha256"] == calibration["matrices"][label]["sha256"]
            assert sha(args.parent_dir / f"{label}.e8") == original["sha256"]
            weight = source[source_key].to("cuda")
            hessian = torch.load(hp, map_location="cuda", weights_only=True)
            records = {}
            for sweeps in (2, 8):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.monotonic()
                restored, info, payload = vector_quantize(weight, hessian, cb, ldlq, 1000 + index,
                                                          damping=.01, scale_override=.9, tune_iters=sweeps)
                torch.cuda.synchronize()
                quant_seconds = time.monotonic() - t0
                path = args.out_dir / f"tune{sweeps}/{label}.e8"
                write_e8(path, payload, info)
                disk = decode_e8(read_e8(path), cb)
                assert torch.equal(restored, disk), label
                decoded_sha = tensor_sha(disk)
                baseline_match = None
                if sweeps == 2:
                    baseline_match = decoded_sha == original["decoded_fp16_sha256"]
                    assert baseline_match, f"Baseline 2-sweep FP16 SHA mismatch: {label}"
                    assert sha(path) == original["sha256"], f"Baseline raw SHA mismatch: {label}"
                assert path.stat().st_size == original["bytes"], f"Raw bitrate changed: {label}"
                metrics = output_metrics(weight, disk, hessian)
                assert abs(metrics["relative_output_mse"] / info["relative_calibration_output_error"] ** 2 - 1) < 2e-5
                torch.cuda.synchronize()
                entry = {**info, **metrics, "file": path.name, "source_key": source_key,
                         "hessian_sha256": hsha, "bytes": path.stat().st_size, "sha256": sha(path),
                         "decoded_fp16_sha256": decoded_sha, "disk_roundtrip_fp16_equal": True,
                         "baseline_decoded_and_raw_sha_match": baseline_match,
                         "quantization_seconds": quant_seconds,
                         "elapsed_seconds": time.monotonic() - t0,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
                records[str(sweeps)] = entry
                report["matrices"][label] = records
                if sweeps == 8:
                    overlay["matrices"][label] = entry
                    overlay["files"][path.name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                write_json(args.out_dir / "report.json", report)
                write_json(args.out_dir / "tune8/manifest.json", overlay)
                print(json.dumps({"matrix": label, "sweeps": sweeps, **entry}), flush=True)
                del restored, disk, payload
                torch.cuda.empty_cache()
            b, q = records["2"], records["8"]
            reduction = 1 - q["original_hessian_weighted_squared_error"] / b["original_hessian_weighted_squared_error"]
            report["comparisons"][label] = {"relative_output_mse_reduction": reduction,
                "quantization_time_ratio": q["quantization_seconds"] / b["quantization_seconds"],
                "raw_bytes_unchanged": q["bytes"] == b["bytes"]}
            write_json(args.out_dir / "report.json", report)
            del weight, hessian
            torch.cuda.empty_cache()
    reductions = [report["comparisons"][label]["relative_output_mse_reduction"] for label in LABELS]
    report["gate_result"] = evaluate_gate(reductions)
    report["timing"] = {str(sweeps): {
        "sum_quantization_seconds": sum(report["matrices"][label][str(sweeps)]["quantization_seconds"] for label in LABELS),
        "sum_with_readback_and_metrics_seconds": sum(report["matrices"][label][str(sweeps)]["elapsed_seconds"] for label in LABELS),
        "estimated_full112_quantization_seconds": 56 * sum(statistics.mean(
            report["matrices"][label][str(sweeps)]["quantization_seconds"]
            for label in LABELS if label.endswith(part)) for part in ("in_proj", "out_proj"))}
        for sweeps in (2, 8)}
    report["aggregate_weighted_mse_reduction"] = 1 - sum(report["matrices"][label]["8"]["original_hessian_weighted_squared_error"] for label in LABELS) / sum(report["matrices"][label]["2"]["original_hessian_weighted_squared_error"] for label in LABELS)
    report.update(complete=True, elapsed_seconds=time.monotonic() - started)
    overlay["screen_complete"] = True
    write_json(args.out_dir / "tune8/manifest.json", overlay)
    write_json(args.out_dir / "report.json", report)
    print(json.dumps({"complete": True, "gate_result": report["gate_result"], "timing": report["timing"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
