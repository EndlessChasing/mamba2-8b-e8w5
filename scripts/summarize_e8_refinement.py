#!/usr/bin/env python3
"""CPU-only reporting of a completed refinement; no new acceptance gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def inventory(directory):
    entries = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in experiment inventory: {path}")
        if path.is_file():
            entries[str(path.relative_to(directory))] = path.stat().st_size
    return {"files": len(entries), "bytes": sum(entries.values()), "file_bytes": entries}


def statistics_for(items):
    values = {label: entry["relative_squared_output_error_reduction"] for label, entry in items.items()}
    return {"matrices": len(values), "median_relative_reduction": statistics.median(values.values()),
            "mean_relative_reduction": statistics.mean(values.values()),
            "minimum_relative_reduction": min(values.values()),
            "maximum_relative_reduction": max(values.values()),
            "worst_matrix": min(values, key=values.get), "best_matrix": max(values, key=values.get),
            "number_improved": sum(value > 0 for value in values.values()),
            "number_worsened": sum(value < 0 for value in values.values()),
            "number_unchanged": sum(value == 0 for value in values.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent-dir", "overlay-dir", "pilot-dir", "integrity-report", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    parent_path, overlay_path = args.parent_dir / "manifest.json", args.overlay_dir / "manifest.json"
    pilot_path = args.pilot_dir / "report.json"
    parent, overlay, pilot, integrity = [json.loads(path.read_text()) for path in
                                       (parent_path, overlay_path, pilot_path, args.integrity_report)]
    assert parent["complete"] and overlay["complete"] and pilot["complete"] and integrity["complete"]
    receipt = integrity["overlay_receipt"]
    assert receipt["overlay_manifest_sha256"] == sha(overlay_path)
    assert receipt["parent_manifest_sha256"] == sha(parent_path)
    assert receipt["screen_receipt"]["screen_report_sha256"] == sha(pilot_path)
    assert integrity["verify_only"] and integrity["quality_evaluated"] is False
    rows = {}
    for label, entry in overlay["matrices"].items():
        if label in pilot["matrices"]:
            baseline_j = pilot["matrices"][label]["2"]["original_hessian_weighted_squared_error"]
            candidate_j = entry["original_hessian_weighted_squared_error"]
            reduction = 1 - candidate_j / baseline_j
            mode = "Both arms independently accumulated original-H J using FP32 GEMM and FP64 reductions"
        else:
            old_relative_mse = parent["matrices"][label]["relative_calibration_output_error"] ** 2
            reduction = 1 - entry["relative_output_mse"] / old_relative_mse
            baseline_j = old_relative_mse * entry["original_hessian_reference_squared_output"]
            candidate_j = entry["original_hessian_weighted_squared_error"]
            mode = "Parent recorded FP32 relative RMS proxy squared; candidate independent original-H relative MSE. Reconstructed parent J uses the same reference-output denominator; tiny accumulation precision differences remain."
        rows[label] = {"relative_squared_output_error_reduction": reduction,
                       "parent_original_hessian_squared_error": baseline_j,
                       "candidate_original_hessian_squared_error": candidate_j,
                       "comparison_precision": mode}
    assert len(rows) == 112
    stats = {"all112": statistics_for(rows)}
    for part in ("in_proj", "out_proj"):
        stats[part] = statistics_for({label: entry for label, entry in rows.items() if label.endswith(part)})
    disk = {"parent": inventory(args.parent_dir), "overlay": inventory(args.overlay_dir),
            "pilot": inventory(args.pilot_dir),
            "transaction_work": inventory(args.overlay_dir.with_name(args.overlay_dir.name + "_work"))}
    model_bytes = {"parent_raw_payload_bytes": sum(entry["bytes"] for entry in parent["files"].values()),
                   "replacement_projection_raw_bytes": sum(entry["bytes"] for entry in overlay["files"].values()),
                   "inherited_raw_payload_bytes": sum(entry["bytes"] for entry in overlay["inherited_files"].values()),
                   "logical_candidate_raw_payload_bytes": overlay["logical_candidate_data_bytes"],
                   "physical_parent_plus_overlay_directory_bytes": disk["parent"]["bytes"] + disk["overlay"]["bytes"],
                   "physical_parent_overlay_pilot_work_bytes": sum(entry["bytes"] for entry in disk.values()),
                   "physical_directory_inventory": disk,
                   "scope": "Logical raw payload excludes manifests, tokenizer, software, licenses and quality reports. Physical counts include all files in the four named experiment directories, including their manifests/receipts; source checkpoint and calibration are separate reproduction inputs. No refined Huffman archive has been built."}
    assert model_bytes["logical_candidate_raw_payload_bytes"] == model_bytes["parent_raw_payload_bytes"]
    report = {"complete": True, "format": "MAMBA2_E8W5_REFINEMENT_SUMMARY_V1",
              "parent_manifest_sha256": sha(parent_path), "overlay_manifest_sha256": sha(overlay_path),
              "screen_report_sha256": sha(pilot_path), "integrity_report_sha256": sha(args.integrity_report),
              "script_sha256": sha(Path(__file__)), "quality_evaluated": False,
              "original_hessian_squared_error": stats, "per_matrix": rows, "bytes": model_bytes,
              "timing": {"sum_all112_quantization_seconds_including_retained_pilots": sum(entry["quantization_seconds"] for entry in overlay["matrices"].values()),
                         "sum_all112_entry_elapsed_seconds_including_retained_pilots": sum(entry["elapsed_seconds"] for entry in overlay["matrices"].values()),
                         "sessions": overlay["sessions"],
                         "maximum_recorded_cuda_allocated_bytes": max(entry["peak_cuda_allocated_bytes"] for entry in overlay["matrices"].values())},
              "coverage": {"replacement_matrices": 112, "retained_pilot_files": 6,
                           "resolved_parameter_tensors": overlay["resolved_parameter_tensor_count"],
                           "resolved_parameters": overlay["resolved_parameter_count"]},
              "limitations": ["Training-proxy reporting only; there is no new acceptance threshold for these112 matrices.",
                              "The six-matrix predeclared gate was applied before full expansion; this summary does not change that gate.",
                              "Projection-output MSE improvement does not establish PPL, MK or source-relative quality.",
                              "Full-model decoded-hash verification belongs to the subsequent GPU evaluator; this receipt records CPU container/provenance verification."]}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "stats": stats, "model_bytes": {key: value for key, value in model_bytes.items() if key not in ("physical_directory_inventory", "scope")},
                      "report_sha256": sha(args.out)}), flush=True)


if __name__ == "__main__":
    main()
