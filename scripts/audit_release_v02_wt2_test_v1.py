#!/usr/bin/env python3
"""Independent stdlib arithmetic/binding audit of the frozen E8/W5 v0.2 test."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
from pathlib import Path

FORMAT = "E8W5_RELEASE_V02_WT2_TEST_V1"
RAW_SHA = "7dd96a44d3de6e634b949e2fb94bbc004c0aecc8404d3848d13f83b396616a1f"
TOKEN_SHA = "5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608"
TEXT_SHA = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
TOKENIZER_SHA = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-10)


def read(path):
    def unique(pairs):
        d = {}
        for k,v in pairs:
            require(k not in d, "Duplicate JSON key")
            d[k] = v
        return d
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--raw-manifest", type=Path, required=True)
    p.add_argument("--software-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    require(not a.out.exists(), "Use a fresh audit receipt")
    result = {"format": "E8W5_RELEASE_V02_WT2_TEST_CPU_AUDIT_V1", "passed": False,
              "comparison_sha256": sha(a.comparison), "raw_manifest_sha256": sha(a.raw_manifest),
              "protocol_sha256": sha(a.protocol), "runner_sha256": sha(a.runner),
              "auditor_sha256": sha(__file__), "quality_threshold": None,
              "scope": "Stdlib complete-population arithmetic and immutable source/reported tensor-ledger audit; no GPU, model load or large-weight rehash."}
    try:
        d, m = read(a.comparison), read(a.raw_manifest)
        require(d["format"] == FORMAT and d["complete"] is True and d["stage"] == "full_test", "Incomplete/wrong test report")
        require(d["raw_manifest_sha256"] == result["raw_manifest_sha256"] == RAW_SHA, "Published manifest changed")
        require(d["runner_sha256"] == result["runner_sha256"] and d["protocol_sha256"] == result["protocol_sha256"] and
                FORMAT in a.protocol.read_text(), "Runner/protocol binding changed")
        require(d["fitting_performed"] is False and d["candidate_selection_performed"] is False and
                d["publication_performed"] is False and d["quality_threshold"] is None, "Evaluation scope changed")
        require(d["source_tag"] == "v0.2.0-resurface" and d["source_tag_commit"] == "e89c764e33a7fcd72095e1b9b9ea47f22d3ff7bb", "Wrong published version")
        source = m["binding"]["code_sha256"]
        require(d["software_sha256"] == source, "Software ledger differs")
        for name, expected in source.items():
            path = a.software_dir / name
            require(not path.is_symlink() and path.is_file() and sha(path) == expected, "Actual archived source differs: " + name)
        ds = d["dataset"]
        require(ds["dataset"] == "Salesforce/wikitext" and ds["configuration"] == "wikitext-2-raw-v1" and
                ds["split"] == "test" and ds["revision_argument"] == REVISION and ds["document_join"] == "two newline characters" and
                ds["automatic_special_tokens"] is False and ds["tokenizer_sha256"] == TOKENIZER_SHA and
                ds["text_sha256"] == TEXT_SHA and ds["token_stream_sha256_int64le"] == TOKEN_SHA and ds["total_tokens"] == 300964,
                "Pinned test dataset/tokenizer differs")
        t = d["raw_tokens"]
        require(t["file"] == "tokens.int64le" and t["bytes"] == 300964*8 and t["sha256"] == TOKEN_SHA, "Raw-token inventory differs")
        token_path = a.comparison.parent / "tokens.int64le"
        require(token_path.stat().st_size == t["bytes"] and sha(token_path) == TOKEN_SHA, "Actual raw tokens differ")
        raw = token_path.read_bytes()
        population = [{"start": start, "target_tokens": min(2048, 300963-start),
                       "token_sha256_int64le": hashlib.sha256(raw[start*8:min(start+2049,300964)*8]).hexdigest()}
                      for start in range(0, 300963, 2048)]
        require(d["population"] == population and len(population) == 147 and population[-1]["target_tokens"] == 1955,
                "Window coverage differs")
        base, adapter = m["base507_fp16_sha256"], m["adapter"]["tensor_sha256"]
        require(len(base) == 507 and len(adapter) == 224, "Full model/adapter ledgers absent")
        package = d["package_receipt"]
        require(package["actual507_fp16_sha256"] == base and package["actual224_adapter_fp16_sha256"] == adapter and
                package["base_parameter_count"] == 8236999680 and package["adapter_parameter_count"] == 1154104 and
                package["decoded_base_weight_bytes"] == 16473999360 and package["decoded_adapter_weight_bytes"] == 2308208 and
                package["source_checkpoint_opened"] is False and package["training_artifacts_opened"] is False and
                package["verified_payload_files"] == 119 and package["gate_mode"] == "soft", "Actual published loaded model receipt differs")
        require(set(d["arms"]) == {"no_resurface", "resurface"}, "Unexpected/missing arms")
        for arm in ("no_resurface", "resurface"):
            row = d["arms"][arm]
            scores, audit = row["ppl"], row["audit"]
            require(scores["execution"] == "prefill" and scores["cache_bytes"] is None and scores["cache_dtype"] == "SSD scan internal precision" and
                    scores["logits_chunk_tokens"] == 64 and len(scores["windows"]) == 147, "Wrong scoring path/population")
            nll = 0.0
            for observed, expected in zip(scores["windows"], population):
                require(observed["start"] == expected["start"] and observed["target_tokens"] == observed["input_tokens"] == expected["target_tokens"] and
                        observed["token_sha256_int64le"] == expected["token_sha256_int64le"], "Different paired window")
                require(close(observed["ppl"], math.exp(observed["nll"] / observed["target_tokens"])), "Window PPL arithmetic differs")
                nll += observed["nll"]
            require(scores["target_tokens"] == 300963 and close(scores["nll"], nll) and close(scores["ppl"],math.exp(nll/300963)), "Aggregate arithmetic differs")
            require(audit["base507_before_sha256"] == base == audit["base507_after_sha256"] and audit["base_identity_version_gradients_unchanged"] is True,
                    "Frozen base tensor audit failed")
            active = arm == "resurface"
            require(audit["active_adapter"] is active and audit["adapter_tensors"] == (224 if active else 0), "Adapter activity differs")
            require(audit["adapter224_before_sha256"] == audit["adapter224_after_sha256"] == (adapter if active else {}), "Adapter tensor ledger differs")
            if active:
                native = audit["native_release_audit"]
                require(native["soft_adapter_enabled"] is True and native["adapter224_fp16_sha256"] == adapter and
                        native["base"]["identity_version_gradients_unchanged"] is True, "Native adapter audit failed")
            memory = row["gpu_memory"]
            require(all(type(memory[k]) is int and memory[k] > 0 for k in ("peak_allocated_bytes","peak_reserved_bytes","current_allocated_bytes")), "Missing measured arm memory")
        off,on = [d["arms"][arm]["ppl"] for arm in ("no_resurface","resurface")]
        c = d["comparison"]
        delta = [b["nll"]-a["nll"] for a,b in zip(off["windows"],on["windows"])]
        require(c["control_ppl"] == off["ppl"] and c["candidate_ppl"] == on["ppl"] and
                close(c["ppl_relative_change"],on["ppl"]/off["ppl"]-1) and c["same_windows_and_targets"] is True and
                c["per_window_nll_delta"] == delta and c["improved_windows"] == sum(x<0 for x in delta), "Paired comparison arithmetic differs")
        r = d["restoration"]
        require(r["probe_window_index"] == 0 and r["excluded_from_aggregate"] is True and r["adapter_removed"] is True and
                r["native_owner_absent"] is True and r["first_window_exact"] is True and
                r["probe"]["windows"] == off["windows"][:1], "Exact adapter-removal restoration failed")
        require(d["numerical_before"] == d["numerical_after"] and d["frozen_input_recheck_passed"] is True, "Frozen backend/input settings changed")
        settings = d["numerical_before"]
        require(settings["float32_matmul_precision"] == "highest" and
                settings["cuda_matmul_allow_fp16_reduced_precision_reduction"] is True and
                all(settings[k] is False for k in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic",
                                                   "deterministic_algorithms_enabled", "deterministic_algorithms_warn_only", "cuda_autocast_enabled")) and
                all(v is None for v in settings["environment_allowlist"].values()), "Published numerical settings differ")
        require(d["environment"]["packages"] == {"torch":"2.11.0+cu128","mamba-ssm":"2.3.2.post1","numpy":"1.26.4","triton":"3.6.0","datasets":"4.8.5","sentencepiece":"0.2.1"}, "Measured package versions differ")
        result.update(passed=True, complete=True, windows=147, targets=300963, last_window_targets=1955,
                      control_ppl=off["ppl"], candidate_ppl=on["ppl"], ppl_relative_change=c["ppl_relative_change"],
                      improved_windows=c["improved_windows"], base507_ledger_exact=True, adapter224_ledger_exact=True,
                      adapter_removal_restoration_exact=True, population_complete=True)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
