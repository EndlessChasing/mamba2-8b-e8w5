#!/usr/bin/env python3
"""Controlled development-only PPL interventions on immutable source/package weights.

Evaluate all 2^3 combinations of E8 projections, W5 embedding, and W5 head.
Then restore projection families or native in_proj row slices in the full-Q
model. The working model swaps Parameter references; slice replacements use
new clones. Original and decoded quantized backups are never modified.
"""
import argparse
import itertools
import inspect
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.calibration import load_wikitext_tokens
from mamba_e8w5.evaluation import evaluate_ppl
from mamba_e8w5.runtime import (SentencePieceTokenizer, environment_receipt, gpu_memory_receipt,
                               load_quantized_model, load_source_model, sha256_file, token_digest)
from audit_decoded import tensor_sha_fp16


def write_report(path, report):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def check_protocol(baseline, quantized):
    for report in (baseline, quantized):
        if not report.get("complete") or report["suite"] != "dev" or report["execution"] != "prefill":
            raise ValueError("Expected complete development prefill reference reports")
        if report["ppl"]["target_tokens"] != 4096 or len(report["ppl"]["windows"]) != 4:
            raise ValueError("Expected exactly four 1024-target validation windows")
        for name in ("runtime", "evaluation"):
            if report[name+"_source_sha256"] != sha256_file(ROOT / "mamba_e8w5" / (name+".py")):
                raise ValueError(f"Frozen {name} source differs from reference report")
    for field in ("dataset", "source_checkpoint_sha256", "tokenizer_sha256"):
        if baseline[field] != quantized[field]:
            raise ValueError(f"Reference identities differ: {field}")
    for a, b in zip(baseline["ppl"]["windows"], quantized["ppl"]["windows"]):
        for field in ("start", "target_tokens", "token_sha256_int64le"):
            if a[field] != b[field]:
                raise ValueError(f"Reference windows differ: {field}")


def endpoint_gate(actual, reference, require_pass=True):
    errors = [abs(a["nll"]-b["nll"]) for a, b in zip(actual["windows"], reference["windows"])]
    maximum = max(errors)
    if maximum > 1e-3 and require_pass:
        raise RuntimeError(f"Reference endpoint did not reproduce: max window NLL difference {maximum}")
    return {"passed": maximum <= 1e-3, "max_window_nll_absolute_difference": maximum,
            "aggregate_nll_absolute_difference": abs(actual["nll"]-reference["nll"]),
            "tolerance_nll_per_window": 1e-3}


def deltas(metrics, baseline):
    return {"nll_delta": metrics["nll"]-baseline["nll"],
            "mean_nll_delta_nats_per_token": (metrics["nll"]-baseline["nll"])/metrics["target_tokens"],
            "ppl_relative_change": metrics["ppl"]/baseline["ppl"]-1,
            "per_window_nll_delta": [a["nll"]-b["nll"]
                                     for a, b in zip(metrics["windows"], baseline["windows"])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, default=Path("reports/baseline_dev_prefill.json"))
    parser.add_argument("--quantized-report", type=Path, default=Path("reports/e8w5_dev_prefill.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/ppl_components_dev.json"))
    parser.add_argument("--skip-row-rescues", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    baseline_ref = json.loads(args.baseline_report.read_text())
    quant_ref = json.loads(args.quantized_report.read_text())
    check_protocol(baseline_ref, quant_ref)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, "validation", baseline_ref["dataset"]["revision_argument"])
    if dataset != baseline_ref["dataset"]:
        raise ValueError("Loaded dataset identity differs from the frozen validation references")
    windows = []
    for entry in baseline_ref["ppl"]["windows"]:
        first, count = entry["start"], entry["target_tokens"]
        window = ids[first:first+count+1]
        if count != 1024 or token_digest(window.numpy()) != entry["token_sha256_int64le"]:
            raise ValueError("Validation token window changed")
        windows.append((first, window))
    source = load_source_model(args.source_dir)
    quantized = load_quantized_model(args.raw_dir)
    if quantized._package_receipt["manifest_sha256"] != quant_ref["package_receipt"]["manifest_sha256"]:
        raise ValueError("Quantized package differs from the reference quality measurement")
    manifest = json.loads((args.raw_dir/'manifest.json').read_text())
    decoded_hashes = {}
    for label, entry in manifest['matrices'].items():
        actual = tensor_sha_fp16(quantized.get_parameter(entry['source_key']))
        if actual != entry['decoded_fp16_sha256']:
            raise ValueError(f'Decoded E8 FP16 hash mismatch: {label}')
        decoded_hashes[label] = actual
    direct_quantized = evaluate_ppl(quantized, windows, execution='prefill')
    old_quantized_comparison = endpoint_gate(direct_quantized, quant_ref['ppl'], require_pass=False)
    print(json.dumps({'stage': 'direct_quantized_anchor', 'ppl': direct_quantized['ppl'],
                      'decoded_E8_hashes_verified': len(decoded_hashes),
                      'prior_report_comparison': old_quantized_comparison}), flush=True)
    original = dict(source.named_parameters())
    coded = dict(quantized.named_parameters())
    if original.keys() != coded.keys():
        raise ValueError("Parameter inventories differ")
    projections = sorted(name for name in original if name.endswith(("mixer.in_proj.weight", "mixer.out_proj.weight")))
    groups = {"E8_projections": projections, "W5_embedding": ["backbone.embedding.weight"],
              "W5_lm_head": ["lm_head.weight"]}
    changed = set().union(*[set(names) for names in groups.values()])
    if len(projections) != 112:
        raise ValueError("Expected 112 projection matrices")
    unchanged = sorted(set(original)-changed)
    for name in unchanged:
        if not torch.equal(original[name], coded[name]):
            raise ValueError(f"Nonintervened parameter differs: {name}")
    bindings = {name: (source.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1])
                for name in changed}

    def assign(name, value):
        module, leaf = bindings[name]
        setattr(module, leaf, value)

    def combination(mask):
        for bit, names in zip(mask, groups.values()):
            for name in names:
                assign(name, coded[name] if bit == "1" else original[name])
        for name in unchanged:
            if source.get_parameter(name).data_ptr() != original[name].data_ptr():
                raise RuntimeError("Unexpected nonintervened parameter replacement")

    report = {"complete": False, "factorial_complete": False, "scope": "Exact frozen 4x1024 validation PPL; diagnostic only",
              "script_sha256": sha256_file(__file__), "runtime_source_sha256": sha256_file(ROOT/"mamba_e8w5/runtime.py"),
              "evaluation_source_sha256": sha256_file(ROOT/"mamba_e8w5/evaluation.py"),
              "reference_reports": {"baseline_sha256": sha256_file(args.baseline_report),
                                    "quantized_sha256": sha256_file(args.quantized_report)},
              "same_process_direct_quantized_anchor": direct_quantized,
              "prior_quantized_endpoint_comparison": old_quantized_comparison,
              "decoded_E8_fp16_hashes": decoded_hashes,
              "endpoint_probe_sha256": sha256_file(ROOT/'reports/ppl_endpoint_probe.json'),
              "prior_failed_factorial_receipt_sha256": sha256_file(ROOT/'reports/ppl_components_dev_failed_endpoint.json'),
              "endpoint_reproducibility_note": "The historical Q endpoint failed the original strict perwindow 0.001-NLL gate. A separate fresh-process probe found direct Q and swapped Q bitwise hidden equality, exact repeated NLLs, no buffer/scalar/stride differences, and matching sampled decoded E8 hashes. This run verifies all 112 decoded E8 hashes and anchors 111 against the direct quantized object within this process. The historical numerical discrepancy is retained explicitly; its underlying kernel cause is not established.",
              "source_receipt": source._package_receipt, "quantized_receipt": quantized._package_receipt,
              "dataset": dataset, "environment": environment_receipt(),
              "unchanged_parameter_tensors_verified_equal": len(unchanged),
              "factors": {key: {"tensors": len(names), "parameters": sum(original[name].numel() for name in names)}
                          for key, names in groups.items()},
              "bit_order": list(groups), "factorial": {}, "row_rescues": {},
              "interpretation": "PPL is exponential in mean NLL. Component PPL changes are not additive; conditional NLL effects depend on other components. Development results are diagnostic, not held-out retesting or release approval."}
    write_report(args.report, report)
    try:
        # Anchor both endpoints before any attribution or rescue intervention.
        order = ["000", "111", "100", "010", "001", "110", "101", "011"]
        for mask in order:
            combination(mask)
            result = evaluate_ppl(source, windows, execution="prefill")
            entry = {"quantized_factors": [key for key, bit in zip(groups, mask) if bit == "1"],
                     "ppl": result, "delta_from_source": deltas(result, baseline_ref["ppl"])}
            if mask in ("000", "111"):
                entry["endpoint_gate"] = endpoint_gate(result, baseline_ref["ppl"] if mask == "000" else direct_quantized)
                entry["endpoint_gate"]["reference"] = 'historical source report' if mask == '000' else 'same-process direct quantized model'
            report["factorial"][mask] = entry
            write_report(args.report, report)
            print(json.dumps({"stage": "factorial", "mask": mask, "ppl": result["ppl"],
                              "mean_nll_delta": entry["delta_from_source"]["mean_nll_delta_nats_per_token"]}), flush=True)
        conditional = {}
        for factor, label in enumerate(groups):
            effects = []
            others = [i for i in range(3) if i != factor]
            for setting in itertools.product("01", repeat=2):
                low = ["0"]*3
                for index, bit in zip(others, setting):
                    low[index] = bit
                high = low.copy(); high[factor] = "1"
                first, second = "".join(low), "".join(high)
                a, b = report["factorial"][first]["ppl"], report["factorial"][second]["ppl"]
                effects.append({"from": first, "to": second, "nll_delta": b["nll"]-a["nll"],
                                "mean_nll_delta_nats_per_token": (b["nll"]-a["nll"])/4096})
            conditional[label] = effects
        report["conditional_nll_effects"] = conditional
        report["factorial_complete"] = True
        write_report(args.report, report)
        print("[factorial complete] Saved all eight combinations and conditional NLL effects", flush=True)
        if not args.skip_row_rescues:
            mixer = source.backbone.layers[0].mixer
            native_path = inspect.getfile(type(mixer))
            report['native_mamba2_source'] = {'path': native_path, 'sha256': sha256_file(native_path),
                'row_order': 'forward splits empty z0/x0, z, xBC, dt; xBC splits x, B, C'}
            inner, state, ngroups, heads = mixer.d_inner, mixer.d_state, mixer.ngroups, mixer.nheads
            if (inner, state, ngroups, heads, mixer.d_ssm) != (8192, 128, 8, 128, 8192):
                raise ValueError("Native in_proj geometry differs from audited 8B layout")
            slices = {"z": (0, inner), "x": (inner, 2*inner),
                      "B": (2*inner, 2*inner+ngroups*state),
                      "C": (2*inner+ngroups*state, 2*inner+2*ngroups*state),
                      "dt": (2*inner+2*ngroups*state, 2*inner+2*ngroups*state+heads)}
            report["native_in_proj_row_slices"] = {key: list(value) for key, value in slices.items()}
            rescues = [("original_all_in_proj", "in_proj", None), ("original_all_out_proj", "out_proj", None),
                       ("original_dt_rows", "in_proj", slices["dt"]),
                       ("original_BC_rows", "in_proj", (slices["B"][0], slices["C"][1])),
                       ("original_B_rows", "in_proj", slices["B"]),
                       ("original_C_rows", "in_proj", slices["C"]),
                       ("original_z_rows", "in_proj", slices["z"]),
                       ("original_x_rows", "in_proj", slices["x"])]
            for label, family, interval in rescues:
                combination("111")
                names = [name for name in projections if name.endswith(f"mixer.{family}.weight")]
                error_sum, source_sum, parameters, layer_stats = 0., 0., 0, []
                for name in names:
                    old, encoded = original[name], coded[name]
                    first, end = interval if interval is not None else (0, old.shape[0])
                    a, b = old[first:end].float(), encoded[first:end].float()
                    num = float((b-a).square().sum()); den = float(a.square().sum())
                    error_sum += num; source_sum += den; parameters += a.numel()
                    layer_stats.append({"parameter": name, "rows": [first, end], "parameters": a.numel(),
                                        "source_squared_norm": den, "error_squared_norm": num,
                                        "relative_weight_error": (num/max(den, 1e-30))**.5})
                    if interval is None:
                        assign(name, old)
                    else:
                        replacement = encoded.detach().clone()
                        replacement[first:end].copy_(old[first:end])
                        assign(name, nn.Parameter(replacement, requires_grad=False))
                        del replacement
                    del a, b
                result = evaluate_ppl(source, windows, execution="prefill")
                report["row_rescues"][label] = {
                    "ppl": result, "delta_from_source": deltas(result, baseline_ref["ppl"]),
                    "delta_from_full_quantized": deltas(result, report["factorial"]["111"]["ppl"]),
                    "restored_parameters": parameters, "fp16_restored_weight_payload_bytes": parameters*2,
                    "fp16_slice_patch_bytes_if_full_E8_base_retained": parameters*2 if interval is not None else None,
                    "source_vs_quantized_relative_weight_error": (error_sum/max(source_sum, 1e-30))**.5,
                    "per_layer_weight_error": layer_stats,
                    "storage_note": "Diagnostic FP16 restoration only; no deployable package generated. For row patches the full E8 base must still be retained because the left rotation mixes native rows. Metadata and a patch execution kernel are not included."}
                combination("111")  # Release temporary row clones before the next intervention.
                write_report(args.report, report)
                print(json.dumps({"stage": "rescue", "name": label, "ppl": result["ppl"],
                                  "restored_parameters": parameters,
                                  "delta_nll_from_fullQ": report["row_rescues"][label]["delta_from_full_quantized"]["nll_delta"]}), flush=True)
    finally:
        for name in changed:
            assign(name, original[name])
    if any(source.get_parameter(name) is not parameter for name, parameter in original.items()):
        raise RuntimeError("Working model was not restored to its original parameter references")
    report["original_parameter_references_restored"] = True
    report["complete"] = True
    report["elapsed_seconds"] = time.perf_counter()-started
    report["gpu_memory"] = gpu_memory_receipt()
    write_report(args.report, report)
    print(json.dumps({"complete": True, "factorial_cases": len(report["factorial"]),
                      "rescue_cases": len(report["row_rescues"]), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
