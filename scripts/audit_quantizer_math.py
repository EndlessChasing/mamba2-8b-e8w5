#!/usr/bin/env python3
"""Independent CPU checks of the frozen E8 math and selected Hessian diagonals.

Does not edit weights, tune on evaluation data, use a GPU, or claim a PPL cause.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.codec import (rotation_factors, rotate_last, block_ldl,
                             load_reference_primitives)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def rel(actual, expected):
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))


def objective(delta, hessian):
    return float(((delta.double() @ hessian.double()) * delta.double()).sum())


def math_checks():
    torch.manual_seed(317)
    results = {"device": "cpu", "gpu_used": False}
    transforms = []
    for m, n in ((160, 128), (144, 384)):
        weight = torch.randn(m, n)
        observations = torch.randn(n * 2, n)
        h = observations.T @ observations / len(observations)
        h /= h.diag().mean()
        h.diagonal().add_(.01)
        balance = (h.diag().clamp_min(1e-8) / weight.square().sum(0).clamp_min(1e-8)).pow(.25).half().float()
        su, sv = torch.randint(0, 2, (n,)) * 2 - 1, torch.randint(0, 2, (m,)) * 2 - 1
        cn, hn = rotation_factors(n, "cpu")
        cm, hm = rotation_factors(m, "cpu")
        qn, qm = torch.kron(cn.T.contiguous(), hn), torch.kron(cm.T.contiguous(), hm)
        wr = rotate_last(rotate_last(weight * balance * su).T * sv).T
        explicit_wr = qm.T @ (sv[:, None] * weight * balance * su) @ qn
        balanced_h = h / balance[:, None] / balance[None, :]
        hr = rotate_last(rotate_last(balanced_h * su).T * su)
        explicit_hr = qn.T @ (balanced_h * su[:, None] * su[None, :]) @ qn
        restored = (rotate_last((rotate_last(wr.T, inverse=True) * sv).T, inverse=True) * su) / balance
        delta = torch.randn_like(weight) * .01
        transformed_delta = rotate_last(rotate_last(delta * balance * su).T * sv).T
        original_loss, transformed_loss = objective(delta, h), objective(transformed_delta, hr)
        l, d = block_ldl(hr)
        ldl = l @ torch.block_diag(*list(d.unbind())) @ l.T
        row = {"shape": [m, n], "weight_rotation_vs_explicit_kron_relative_error": rel(wr, explicit_wr),
               "hessian_rotation_vs_explicit_kron_relative_error": rel(hr, explicit_hr),
               "no_quantization_inverse_relative_error": rel(restored, weight),
               "weighted_objective_original": original_loss, "weighted_objective_transformed": transformed_loss,
               "weighted_objective_relative_difference": abs(original_loss - transformed_loss) / original_loss,
               "block_ldl_reconstruction_relative_error": rel(ldl, hr)}
        errors = [v for k, v in row.items() if k.endswith("relative_error") or k.endswith("relative_difference")]
        row["pass"] = max(errors) < 3e-6
        transforms.append(row)
    results["algebraic_transform_oracles"] = transforms
    c, _ = rotation_factors(18560, "cpu")
    sample = torch.randn(4, 18560)
    results["actual_in_projection_output_axis"] = {"dimension": 18560, "dct_factor": list(c.shape),
        "dct_orthogonality_relative_error": rel(c @ c.T, torch.eye(c.shape[0])),
        "roundtrip_relative_error": rel(rotate_last(rotate_last(sample), inverse=True), sample)}
    codebook, buffered = load_reference_primitives()
    source = ROOT / "third_party/quip-sharp/lib/algo/quip.py"
    tree = ast.parse(source.read_text())
    definition = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "LDLQ"]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=definition, type_ignores=[]), str(source), "exec"), scope)
    unbuffered = scope["LDLQ"]
    n = 256
    data = torch.randn(2 * n, n)
    h = data.T @ data / len(data) + torch.eye(n) * .01
    wr = torch.randn(24, n) * .9
    l, d = block_ldl(h)
    ldlq_results = []
    for sweeps in (0, 2):
        args = SimpleNamespace(quip_tune_iters=sweeps, resid_scale_override=-1)
        direct, direct_indices = unbuffered(wr, h, l, d, codebook, args)
        candidate, candidate_indices = buffered(wr, h, l, d, codebook, args, buf_cols=128)
        ldlq_results.append({"tuning_sweeps": sweeps,
            "values_equal": bool(torch.equal(direct, candidate)),
            "indices_equal": bool(torch.equal(direct_indices, candidate_indices)),
            "different_indices": int((direct_indices != candidate_indices).sum()),
            "buffered_weighted_loss": objective(wr - candidate, h),
            "unbuffered_weighted_loss": objective(wr - direct, h)})
    results["buffered_vs_unbuffered_upstream_ldlq"] = ldlq_results
    nearest_checks = []
    for scale in (.1, .9, 1.5, 3.0):
        vectors = torch.randn(32, 8) * scale
        chosen, _ = codebook.quantize(vectors)
        # Independent exhaustive 65,536-vector Euclidean search, in FP64.
        grid = codebook.grid.double()
        distances = (vectors.double().square().sum(1)[:, None] + grid.square().sum(1)[None, :]
                     - 2 * vectors.double() @ grid.T)
        minimum = distances.min(1).values
        chosen_error = (vectors.double() - chosen.double()).square().sum(1)
        gap = chosen_error - minimum
        nearest_checks.append({"input_scale": scale, "vectors": len(vectors),
            "maximum_excess_squared_error_vs_exhaustive": float(gap.max()),
            "all_nearest_to_tolerance": bool(torch.all(gap.abs() < 1e-8))})
    results["e8_codebook_vs_exhaustive_nearest"] = nearest_checks
    results["all_synthetic_math_checks_pass"] = (all(x["pass"] for x in transforms)
        and results["actual_in_projection_output_axis"]["roundtrip_relative_error"] < 3e-6
        and all(x["values_equal"] and x["indices_equal"] for x in ldlq_results)
        and all(x["all_nearest_to_tolerance"] for x in nearest_checks))
    results["limitations"] = ["Synthetic CPU algebra/oracle checks do not measure 8B language quality.",
        "Matching upstream code does not establish global optimality of LDLQ or optimal damping/scale.",
        "No GPU evaluation, layer restoration, or original-source-model tensor loading was performed."]
    return results


def hessian_diagonals(directory, labels):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    result = []
    for label in labels:
        entry = manifest["matrices"][label]
        path = directory / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Hessian hash mismatch: " + label)
        h = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        diagonal = h.diag().double()
        normalized = diagonal / diagonal.mean()
        damping = []
        for value in (.0001, .001, .003, .01, .03):
            below = normalized < value
            damping.append({"relative_damping": value, "channels_below_damping": int(below.sum()),
                "fraction_channels_below_damping": float(below.double().mean()),
                "fraction_original_trace_below_damping": float(normalized[below].sum() / normalized.sum()),
                "added_trace_fraction": value,
                "median_damping_to_original_diagonal_ratio": float((value / normalized.clamp_min(1e-300)).median())})
        result.append({"label": label, "sha256": entry["sha256"], "shape": list(h.shape),
            "diagonal_min": float(diagonal.min()), "diagonal_max": float(diagonal.max()),
            "diagonal_mean": float(diagonal.mean()), "zero_diagonal_channels": int((diagonal == 0).sum()),
            "normalized_diagonal_quantiles": {str(q): float(torch.quantile(normalized, q))
                                              for q in (0., .01, .1, .25, .5, .75, .9, .99, 1.)},
            "damping_diagnostics": damping,
            "scope": "Read-only diagonal diagnostic; this ratio is not the matrix condition number."})
        del h, diagonal
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hessian-dir", type=Path)
    parser.add_argument("--hessian-labels", nargs="+", default=["layer18.out_proj", "layer43.out_proj", "layer55.out_proj"])
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    report = {"complete": False, "script_sha256": sha(__file__),
              "frozen_codec_sha256": sha(ROOT / "mamba_e8w5/codec.py")}
    report.update(math_checks())
    if args.hessian_dir:
        report["selected_actual_hessian_diagonals"] = hessian_diagonals(args.hessian_dir, args.hessian_labels)
    report.update(complete=True, elapsed_seconds=time.monotonic() - started)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
