#!/usr/bin/env python3
"""Build a resumable 112-projection eight-sweep overlay after a passed screen.

The parent and pilot remain immutable. The frozen E8 codec is called directly;
no PPL data, new recipe, additional bit fields, or vocabulary changes are used.
Pending files and commit receipts are kept in a separate sibling work directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.codec import load_reference_primitives, vector_quantize, write_e8, read_e8, decode_e8
from mamba_e8w5.runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, load_source_state
from diagnose_e8_refinement import sha, tensor_sha, write_json, output_metrics, LABELS
from evaluate_refinement import RECIPE, verify_screen, checked_file, read_header

ALL_LABELS = tuple(f"layer{i}.{part}" for i in range(56) for part in ("in_proj", "out_proj"))


def identity(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def commit_file(pending, entry, report, output, work):
    """Receipt-before-rename transaction, recoverable without requantization."""
    label = entry["file"][:-3]
    assert label in ALL_LABELS
    destination = output / entry["file"]
    if destination.exists():
        raise FileExistsError(destination)
    receipt = {"binding_sha256": identity(report["binding"]), "entry": entry,
               "pending_basename": pending.name}
    receipt_path = work / f"{label}.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    write_json(receipt_path, receipt)
    os.replace(pending, destination)
    report["matrices"][label] = entry
    report["files"][entry["file"]] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    write_json(output / "manifest.json", report)


def recover(report, output, work):
    """Verify recorded files; adopt only fully hashed transaction receipts."""
    for label, entry in report["matrices"].items():
        checked_file(output, entry["file"], entry)
        assert report["files"][entry["file"]] == {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    restored = []
    for receipt_path in sorted(work.glob("layer*.json")):
        receipt = json.loads(receipt_path.read_text())
        assert receipt["binding_sha256"] == identity(report["binding"]), receipt_path
        entry = receipt["entry"]
        label = entry["file"][:-3]
        assert label in ALL_LABELS and receipt_path.name == f"{label}.json"
        assert Path(receipt["pending_basename"]).name == receipt["pending_basename"]
        if label in report["matrices"]:
            assert entry == report["matrices"][label]
            continue
        final = output / entry["file"]
        pending = work / receipt["pending_basename"]
        if final.exists():
            checked_file(output, entry["file"], entry)
        else:
            checked_file(work, pending.name, entry)
            os.replace(pending, final)
        report["matrices"][label] = entry
        report["files"][entry["file"]] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        restored.append(label)
    expected = set(report["files"]) | {"manifest.json"}
    assert {p.name for p in output.iterdir()} == expected, "Unrecorded overlay files; inspect before resuming"
    if restored:
        write_json(output / "manifest.json", report)
    return restored


def new_pending(work, label):
    descriptor, name = tempfile.mkstemp(prefix=f"{label}.", suffix=".e8.part", dir=work)
    os.close(descriptor)
    return Path(name)


def self_test():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        for stage in ("receipt_only", "renamed", "committed"):
            output, work = root / stage, root / (stage + "_work")
            output.mkdir(); work.mkdir()
            report = {"binding": {"test": True}, "matrices": {}, "files": {}}
            write_json(output / "manifest.json", report)
            pending = new_pending(work, "layer0.in_proj")
            pending.write_bytes(b"a valid tiny transaction")
            entry = {"file": "layer0.in_proj.e8", "bytes": pending.stat().st_size, "sha256": sha(pending)}
            if stage == "committed":
                commit_file(pending, entry, report, output, work)
            else:
                write_json(work / "layer0.in_proj.json", {"binding_sha256": identity(report["binding"]),
                           "entry": entry, "pending_basename": pending.name})
                if stage == "renamed":
                    os.replace(pending, output / entry["file"])
            recover(report, output, work)
            assert report["matrices"]["layer0.in_proj"] == entry
            assert not recover(report, output, work)
            (output / entry["file"]).write_bytes(b"corrupt")
            try:
                recover(report, output, work)
            except ValueError:
                pass
            else:
                raise AssertionError("Corruption was not rejected")
    print(json.dumps({"self_test": "pass", "resume_stages": 3, "corruption_rejected": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "hessian-dir", "parent-dir", "pilot-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/REFINEMENT_PROTOCOL.md")
    parser.add_argument("--preflight-only", action="store_true", help="CPU file checks and immutable pilot copies only")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("source_dir", "hessian_dir", "parent_dir", "pilot_dir", "out_dir")):
        parser.error("All five directory arguments are required")
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.out_dir.resolve()
    work = output.with_name(output.name + "_work")
    parent_path, calibration_path = args.parent_dir / "manifest.json", args.hessian_dir / "manifest.json"
    screen_path = args.pilot_dir / "report.json"
    parent, calibration = json.loads(parent_path.read_text()), json.loads(calibration_path.read_text())
    screen = json.loads(screen_path.read_text())
    assert parent["complete"] and set(parent["matrices"]) == set(ALL_LABELS)
    assert parent["model_config"] == calibration["model_config"] == MODEL_CONFIG
    assert parent["source_checkpoint_sha256"] == calibration["source_checkpoint_sha256"] == SOURCE_CHECKPOINT_SHA256
    assert calibration["complete"] and calibration["calibration_split"] == "train" and calibration["evaluation_data_used"] is False
    assert sha(calibration_path) == parent["binding"]["hessian_manifest_sha256"]
    assert parent["binding"]["scale"] == .9 and parent["binding"]["damping"] == .01 and parent["binding"]["tune_iters"] == 2
    for name, digest in parent["binding"]["code_sha256"].items():
        assert sha(ROOT / "mamba_e8w5" / name) == digest, name
    for name, digest in parent["binding"]["quip_sha256"].items():
        assert sha(ROOT / "third_party/quip-sharp" / name) == digest, name
    for name, entry in parent["files"].items():
        checked_file(args.parent_dir, name, entry)
    inherited = {name: entry for name, entry in parent["files"].items() if not name.endswith(".e8")}
    assert set(inherited) == {"embedding.uniform", "lm_head.uniform", "other_fp16.pt", "e8_codebook.bin", "config.json"}
    binding = {"source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
               "tokenizer_sha256": parent["tokenizer_sha256"],
               "hessian_manifest_sha256": sha(calibration_path),
               "codec_source_sha256": sha(ROOT / "mamba_e8w5/codec.py"),
               "runtime_source_sha256": sha(ROOT / "mamba_e8w5/runtime.py"),
               "quip_sha256": parent["binding"]["quip_sha256"],
               "refinement_protocol_sha256": sha(args.protocol),
               "screen_report_sha256": sha(screen_path),
               "screen_driver_sha256": sha(ROOT / "scripts/diagnose_e8_refinement.py"),
               "builder_source_sha256": sha(__file__)}
    assert binding["screen_driver_sha256"] == screen["binding"]["script_source_sha256"]
    candidate = {"format": "MAMBA2_E8W5_REFINEMENT_OVERLAY_V1", "complete": False,
                 "scope": "full112_train_only_eight_sweep_overlay", "parent_manifest_sha256": sha(parent_path),
                 "model_config": MODEL_CONFIG, "binding": binding, "recipe": RECIPE,
                 "matrices": {label: copy.deepcopy(screen["matrices"][label]["8"]) for label in LABELS},
                 "files": {}, "inherited_files": inherited}
    candidate["screen_receipt"] = verify_screen(screen_path, args.protocol, parent, candidate, calibration)
    for label in LABELS:
        entry = candidate["matrices"][label]
        path = checked_file(args.pilot_dir / "tune8", entry["file"], entry)
        assert read_header(path)["shape"] == entry["shape"]
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text())
        for field in ("format", "parent_manifest_sha256", "model_config", "binding", "recipe", "inherited_files", "screen_receipt"):
            assert report[field] == candidate[field], f"Resume provenance differs: {field}"
        assert work.is_dir(), "Missing transaction directory"
        recovered = recover(report, output, work)
        print(json.dumps({"resume_verified_matrices": len(report["matrices"]), "recovered_transactions": recovered}), flush=True)
    else:
        assert not output.exists() and not work.exists(), "Unrecorded output directory; inspect before continuing"
        output.mkdir(parents=True); work.mkdir()
        report = {**candidate, "matrices": {}, "files": {}, "sessions": []}
        write_json(manifest_path, report)
    for label in LABELS:
        if label in report["matrices"]:
            assert report["matrices"][label] == candidate["matrices"][label]
            continue
        entry = candidate["matrices"][label]
        pending = new_pending(work, label)
        shutil.copyfile(args.pilot_dir / "tune8" / entry["file"], pending)
        assert sha(pending) == entry["sha256"]
        commit_file(pending, entry, report, output, work)
    if report["complete"]:
        assert set(report["matrices"]) == set(ALL_LABELS)
        print(json.dumps({"complete": True, "all_files_reverified": True, "manifest_sha256": sha(manifest_path)}), flush=True)
        return
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "copied_pilot_files": 6,
                          "existing_verified_matrices": len(report["matrices"]),
                          "gpu_used": False, "binding": binding}), flush=True)
        return
    source = load_source_state(args.source_dir)
    cb, ldlq = load_reference_primitives()
    cb = cb.cuda()
    session = {"start_time_unix": time.time(), "matrices": [],
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "tf32_matmul": False}
    report["sessions"].append(session)
    write_json(manifest_path, report)
    with torch.inference_mode():
        for index, label in enumerate(ALL_LABELS):
            if label in report["matrices"]:
                continue
            layer, part = divmod(index, 2)
            family = ("in_proj", "out_proj")[part]
            source_key = f"backbone.layers.{layer}.mixer.{family}.weight"
            hp = args.hessian_dir / calibration["matrices"][label]["file"]
            hsha = sha(hp)
            assert hsha == parent["matrices"][label]["hessian_sha256"] == calibration["matrices"][label]["sha256"]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            weight = source[source_key].to("cuda")
            hessian = torch.load(hp, map_location="cuda", weights_only=True)
            tq = time.monotonic()
            restored, info, payload = vector_quantize(weight, hessian, cb, ldlq, 1000 + index,
                                                      damping=.01, scale_override=.9, tune_iters=8)
            torch.cuda.synchronize()
            quant_seconds = time.monotonic() - tq
            pending = new_pending(work, label)
            write_e8(pending, payload, info)
            disk = decode_e8(read_e8(pending), cb)
            assert torch.equal(restored, disk) and bool(torch.isfinite(disk).all()), label
            decoded_sha = tensor_sha(disk)
            metrics = output_metrics(weight, disk, hessian)
            assert abs(metrics["relative_output_mse"] / info["relative_calibration_output_error"] ** 2 - 1) < 2e-5
            assert pending.stat().st_size == parent["files"][f"{label}.e8"]["bytes"]
            torch.cuda.synchronize()
            entry = {**info, **metrics, "file": f"{label}.e8", "source_key": source_key,
                     "hessian_sha256": hsha, "bytes": pending.stat().st_size, "sha256": sha(pending),
                     "decoded_fp16_sha256": decoded_sha, "disk_roundtrip_fp16_equal": True,
                     "quantization_seconds": quant_seconds, "elapsed_seconds": time.monotonic() - t0,
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
            session["matrices"].append(label)
            commit_file(pending, entry, report, output, work)
            print(json.dumps({"matrix": label, "completed": len(report["matrices"]), "total": 112,
                              "relative_calibration_output_error": info["relative_calibration_output_error"],
                              "quantization_seconds": quant_seconds, "elapsed_seconds": entry["elapsed_seconds"],
                              "decoded_fp16_sha256": decoded_sha}), flush=True)
            del weight, hessian, restored, disk, payload
            torch.cuda.empty_cache()
    assert set(report["matrices"]) == set(ALL_LABELS)
    recover(report, output, work)
    assert len(report["files"]) == 112
    for label in LABELS:
        assert report["matrices"][label] == candidate["matrices"][label]
    reductions = {label: 1 - report["matrices"][label]["relative_output_mse"] / parent["matrices"][label]["relative_calibration_output_error"] ** 2
                  for label in ALL_LABELS}
    report["training_proxy_summary"] = {"relative_mse_reductions": reductions,
        "median_relative_mse_reduction": statistics.median(reductions.values()),
        "minimum_relative_mse_reduction": min(reductions.values()),
        "maximum_relative_mse_reduction": max(reductions.values()),
        "number_improved": sum(value > 0 for value in reductions.values()),
        "note": "Unscreened parent denominator is its original recorded FP32 RMS proxy squared; six pilot pairs use independent FP64-accumulated metrics in the screen report. No new acceptance threshold is applied to the other106 matrices."}
    report["parameter_mapping"] = {entry["source_key"]: entry["file"] for entry in report["matrices"].values()}
    projection_count = sum(entry["shape"][0] * entry["shape"][1] for entry in report["matrices"].values())
    report["replacement_parameter_count"] = projection_count
    report["resolved_parameter_count"] = parent["total_parameter_count"]
    report["resolved_parameter_tensor_count"] = len(parent["parameter_mapping"])
    assert projection_count == 6136266752 and report["resolved_parameter_count"] == 8236999680
    assert report["resolved_parameter_tensor_count"] == 507
    report["replacement_data_file_bytes"] = sum(entry["bytes"] for entry in report["files"].values())
    report["inherited_data_file_bytes"] = sum(entry["bytes"] for entry in inherited.values())
    report["logical_candidate_data_bytes"] = report["replacement_data_file_bytes"] + report["inherited_data_file_bytes"]
    assert report["logical_candidate_data_bytes"] == parent["data_file_bytes"]
    resolved = {name: {"origin": "parent", **entry} for name, entry in inherited.items()}
    resolved.update({name: {"origin": "overlay", **entry} for name, entry in report["files"].items()})
    report["resolved_files"] = resolved
    report["resolved_file_ledger_sha256"] = identity(resolved)
    session["elapsed_seconds_including_setup"] = time.monotonic() - started
    session["complete"] = True
    report["complete"] = True
    report["quality_evaluated"] = False
    report["publication_authorized"] = False
    report["storage_note"] = "Overlay is not standalone. Logical bytes include all inherited raw payloads; physical experiment storage also retains the entire immutable parent, pilot and transaction receipts. Tokenizer, software, licenses and manifests are extra."
    write_json(manifest_path, report)
    print(json.dumps({"complete": True, "manifest_sha256": sha(manifest_path),
                      "matrices": len(report["matrices"]), "logical_candidate_data_bytes": report["logical_candidate_data_bytes"],
                      "training_proxy_summary": report["training_proxy_summary"],
                      "elapsed_seconds": session["elapsed_seconds_including_setup"]}), flush=True)


if __name__ == "__main__":
    main()
