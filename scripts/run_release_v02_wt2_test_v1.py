#!/usr/bin/env python3
"""External frozen paired WT2 test for the unchanged public E8/W5 v0.2.0 artifact."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path

FORMAT = "E8W5_RELEASE_V02_WT2_TEST_V1"
RAW_SHA = "7dd96a44d3de6e634b949e2fb94bbc004c0aecc8404d3848d13f83b396616a1f"
TOKEN_SHA = "5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608"
TEXT_SHA = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
TOKENIZER_SHA = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PACKAGES = {"torch": "2.11.0+cu128", "mamba-ssm": "2.3.2.post1", "triton": "3.6.0",
            "numpy": "1.26.4", "datasets": "4.8.5", "sentencepiece": "0.2.1"}
ENV = ("MAMBA_DETERMINISTIC", "TRITON_CACHE_AUTOTUNING", "TRITON_AUTOTUNE_BLOCK_SIZE_M",
       "TRITON_AUTOTUNE_BLOCK_SIZE_N", "TRITON_AUTOTUNE_BLOCK_SIZE_K", "TRITON_AUTOTUNE_BLOCK_SIZE_DSTATE",
       "TRITON_CACHE_DIR", "TRITON_INTERPRET", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
       "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_file(root, name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name and not name.startswith("/"), "Unsafe source path")
    path = root
    for part in name.split("/"):
        require(part not in ("", ".", ".."), "Unsafe source path component")
        path = path / part
        require(not path.is_symlink(), "Source symlink forbidden")
    require(path.is_file(), "Source file absent: " + name)
    return path


def numerical(torch):
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cuda_matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "environment_allowlist": {name: os.environ.get(name) for name in ENV},
            "cuda_autocast_enabled": torch.is_autocast_enabled("cuda")}


def external_sources():
    rows = {}
    for name, module in sorted(sys.modules.items()):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            file = getattr(module, "__file__", None)
            if file and Path(file).is_file():
                rows[name] = {"path": str(Path(file).resolve()), "sha256": sha(file)}
    return rows


def check_score(result, windows):
    require(result["execution"] == "prefill" and result["cache_bytes"] is None and result["logits_chunk_tokens"] == 64,
            "Published prefill scoring mode changed")
    require(len(result["windows"]) == 147, "Incomplete window population")
    count = 0
    nll = 0.0
    for row, expected in zip(result["windows"], windows):
        require(row["start"] == expected["start"] and row["input_tokens"] == expected["target_tokens"] and
                row["target_tokens"] == expected["target_tokens"] and row["token_sha256_int64le"] == expected["token_sha256_int64le"],
                "Changed window population")
        require(math.isfinite(row["nll"]) and row["ppl"] == math.exp(row["nll"] / row["target_tokens"]), "Invalid window arithmetic")
        count += row["target_tokens"]
        nll += row["nll"]
    require(count == 300963 and result["target_tokens"] == count and nll == result["nll"] and result["ppl"] == math.exp(nll / count),
            "Aggregate target coverage or PPL arithmetic differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--software-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, software, protocol, out = [p.resolve() for p in (args.raw_dir, args.software_dir, args.protocol, args.out_dir)]
    require(not args.raw_dir.is_symlink() and raw.is_dir() and not args.software_dir.is_symlink() and software.is_dir(), "Regular raw/source directories required")
    require(not out.exists(), "Use a fresh output directory")
    require(not out.is_relative_to(raw) and not out.is_relative_to(software), "Output cannot modify the frozen package/source tree")
    require(sha(protocol) == args.expected_protocol_sha256 and FORMAT in protocol.read_text(), "Frozen protocol hash/identifier differs")
    runner = Path(__file__).resolve()
    runner_sha, protocol_sha = sha(runner), sha(protocol)
    manifest_path = safe_file(raw, "manifest.json")
    require(sha(manifest_path) == RAW_SHA, "Published raw manifest identity differs")
    manifest = json.loads(manifest_path.read_text())
    source_hashes = {name: sha(safe_file(software, name)) for name in manifest["binding"]["code_sha256"]}
    require(source_hashes == manifest["binding"]["code_sha256"], "Archived software binding differs")
    require(platform.python_version() == "3.10.12", "Published Python version differs")
    actual_packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(actual_packages == PACKAGES, "Published numerical package versions differ")
    require(all(os.environ.get(name) is None for name in ENV), "Published numerical environment allowlist differs")
    require(not any(name == "mamba_e8w5" or name.startswith("mamba_e8w5.") for name in sys.modules), "Model package was already imported")
    sys.path.insert(0, str(software))
    import torch
    from mamba_e8w5 import release_runtime as release, resurface_native as native, evaluation, runtime
    for module in (release, native, evaluation, runtime):
        require(Path(module.__file__).resolve().is_relative_to(software), "Imported model code outside archive")
    require(torch.cuda.is_available(), "CUDA GPU required")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False, warn_only=False)
    before = numerical(torch)
    out.mkdir(parents=True)
    started = time.time()
    report = {"format": FORMAT, "complete": False, "stage": "full_test", "fitting_performed": False,
              "candidate_selection_performed": False, "publication_performed": False,
              "source_tag": "v0.2.0-resurface", "source_tag_commit": "e89c764e33a7fcd72095e1b9b9ea47f22d3ff7bb",
              "runner_sha256": runner_sha, "protocol_sha256": protocol_sha, "raw_manifest_sha256": RAW_SHA,
              "raw_dir": str(raw), "software_dir": str(software), "software_sha256": source_hashes,
              "numerical_before": before, "environment": runtime.environment_receipt(),
              "test_history": "Project test previously used by original e8w5_v1; this frozen v0.2.0 retrospective test is a different published artifact.",
              "quality_threshold": None, "arms": {}}
    path = out / "comparison.json"
    write(path, report)
    model = None
    try:
        tokenizer = release.load_tokenizer(raw)
        require(tokenizer.sha256 == TOKENIZER_SHA, "Tokenizer differs")
        ids, dataset = evaluation.load_wikitext_tokens(tokenizer, "test", REVISION)
        require(dataset["split"] == "test" and dataset["revision_argument"] == REVISION and
                dataset["text_sha256"] == TEXT_SHA and dataset["token_stream_sha256_int64le"] == TOKEN_SHA and
                dataset["total_tokens"] == 300964 and not dataset["automatic_special_tokens"], "Pinned full test corpus differs")
        tokens_path = out / "tokens.int64le"
        tokens_path.write_bytes(ids.numpy().astype("<i8", copy=False).tobytes())
        require(sha(tokens_path) == TOKEN_SHA, "Raw token export differs")
        windows = evaluation.ppl_windows(ids, 2048)
        population = [{"start": start, "target_tokens": len(window)-1,
                       "token_sha256_int64le": runtime.token_digest(window.numpy())} for start, window in windows]
        require(len(windows) == 147 and population[-1]["target_tokens"] == 1955 and
                sum(row["target_tokens"] for row in population) == 300963 and
                [row["start"] for row in population] == [i*2048 for i in range(147)], "Incomplete test population")
        report.update(dataset=dataset, population=population, raw_tokens={"file": tokens_path.name, "bytes": tokens_path.stat().st_size, "sha256": TOKEN_SHA})
        write(path, report)
        torch.cuda.reset_peak_memory_stats()
        model = release.load_model(raw, device="cuda", expected_manifest_sha256=RAW_SHA)
        report["package_receipt"] = model._package_receipt
        expected = manifest["base507_fp16_sha256"]
        adapter_expected = manifest["adapter"]["tensor_sha256"]
        require(model._package_receipt["actual507_fp16_sha256"] == expected and
                model._package_receipt["actual224_adapter_fp16_sha256"] == adapter_expected, "Loaded exact tensors differ")
        identities = {name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()}
        initial_bank = model._release_resurface_adapter
        require(not initial_bank.closed and initial_bank.gate_mode == "soft", "Mandatory loaded adapter not active")
        report["load_memory"] = runtime.gpu_memory_receipt()
        release.close_model(model)
        require(initial_bank.closed and not initial_bank._handles and model not in native._OWNERS and
                not hasattr(model, "_release_resurface_adapter"), "Adapter removal incomplete")
        del initial_bank
        for arm in ("no_resurface", "resurface"):
            if arm == "resurface":
                bank = native.install_fp16(model, raw / "adapter_fp16.pt", expected_binding=manifest["adapter"]["binding"],
                                           expected_base_hashes=expected, production=True)
                model._release_resurface_adapter = bank
                require(not bank.closed and bank.gate_mode == "soft" and len(bank.masters) == 224, "Fresh published adapter not active")
                hashes = release._validate_tensors(bank.masters, release.adapter_shapes(), adapter_expected)
            else:
                hashes = {}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            score = evaluation.evaluate_ppl(model, windows, execution="prefill", logits_chunk=64)
            torch.cuda.synchronize()
            memory = runtime.gpu_memory_receipt()
            check_score(score, population)
            actual = release._validate_tensors(dict(model.named_parameters()), release.base_shapes(), expected)
            require({name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()} == identities,
                    "Base identity/version changed")
            require(all(not p.requires_grad and p.grad is None for p in model.parameters()), "Base gradients changed")
            audit = {"base507_before_sha256": expected, "base507_after_sha256": actual,
                     "base_identity_version_gradients_unchanged": True, "adapter_tensors": len(hashes),
                     "adapter224_before_sha256": hashes, "adapter224_after_sha256": {}, "active_adapter": arm == "resurface"}
            if arm == "resurface":
                require(not bank.closed and bank.gate_mode == "soft", "Adapter unexpectedly closed")
                audit["native_release_audit"] = release.audit_loaded_model(model, check_values=False)
                audit["adapter224_after_sha256"] = audit["native_release_audit"]["adapter224_fp16_sha256"]
                require(audit["adapter224_after_sha256"] == hashes, "Adapter changed during scoring")
            else:
                require(model not in native._OWNERS and not hasattr(model, "_release_resurface_adapter"), "Off arm installed adapter")
            report["arms"][arm] = {"ppl": score, "audit": audit, "gpu_memory": memory}
            write(path, report)
        release.close_model(model)
        require(bank.closed and not bank._handles and model not in native._OWNERS and not hasattr(model, "_release_resurface_adapter"), "Final adapter removal incomplete")
        probe = evaluation.evaluate_ppl(model, windows[:1], execution="prefill", logits_chunk=64)
        restored = probe["windows"][0] == report["arms"]["no_resurface"]["ppl"]["windows"][0]
        require(restored, "Adapter removal did not restore exact first-window score")
        require({name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()} == identities and
                all(not p.requires_grad and p.grad is None for p in model.parameters()), "Base identity/gradients changed during restoration probe")
        report["restoration"] = {"probe_window_index": 0, "excluded_from_aggregate": True,
                                 "adapter_removed": True, "native_owner_absent": True,
                                 "first_window_exact": restored, "probe": probe}
        off, on = [report["arms"][arm]["ppl"] for arm in ("no_resurface", "resurface")]
        report["comparison"] = {"control_ppl": off["ppl"], "candidate_ppl": on["ppl"],
                                "ppl_relative_change": on["ppl"]/off["ppl"]-1,
                                "same_windows_and_targets": True,
                                "per_window_nll_delta": [b["nll"]-a["nll"] for a,b in zip(off["windows"],on["windows"])],
                                "improved_windows": sum(b["nll"] < a["nll"] for a,b in zip(off["windows"],on["windows"]))}
        report["numerical_after"] = numerical(torch)
        require(report["numerical_after"] == before, "Numerical settings changed during test")
        report["external_loaded_source_sha256"] = external_sources()
        require(sha(runner) == runner_sha and sha(protocol) == protocol_sha and sha(manifest_path) == RAW_SHA and
                {name: sha(safe_file(software,name)) for name in source_hashes} == source_hashes, "Input bindings changed during test")
        report["frozen_input_recheck_passed"] = True
        report["complete"] = True
        report["elapsed_seconds"] = time.time() - started
        write(path, report)
        print(json.dumps({"complete": True, "control_ppl": off["ppl"], "candidate_ppl": on["ppl"],
                          "improved_windows": report["comparison"]["improved_windows"], "report": str(path)}), flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["elapsed_seconds"] = time.time() - started
        write(path, report)
        raise
    finally:
        if model is not None:
            release.close_model(model)


if __name__ == "__main__":
    main()
