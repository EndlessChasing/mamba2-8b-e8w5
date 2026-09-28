"""Opt-in CUDA graph driver for the unchanged tokenwise PPL/MK protocol.

The numerical graph is bound to a successful bitwise parity receipt. Reports
retain runtime/evaluation source identities and additionally record graph and
driver hashes. The default evaluation module and its native execution stay intact.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import evaluation
from .calibration import WIKITEXT_REVISION, load_wikitext_tokens
from .runtime import (SentencePieceTokenizer, cache_bytes, environment_receipt,
                      gpu_memory_receipt, load_quantized_model, load_source_model,
                      sha256_file, token_digest)
from .tokenwise_graph import GraphTokenStepper


def read_parity_receipt(path):
    path = Path(path)
    receipt = json.loads(path.read_text())
    if not receipt.get("complete") or not receipt.get("parity", {}).get("passed"):
        raise ValueError("Graph driver requires a completed, passing bitwise parity receipt")
    graph_hash = sha256_file(Path(__file__).with_name("tokenwise_graph.py"))
    runtime_hash = sha256_file(Path(__file__).with_name("runtime.py"))
    if receipt["graph_source_sha256"] != graph_hash or receipt["runtime_source_sha256"] != runtime_hash:
        raise ValueError("Parity receipt does not match the graph/runtime source being executed")
    return {"sha256": sha256_file(path), "receipt": receipt}


class GraphEvaluationAdapter:
    """Reuse one captured stepper across PPL windows and independent MK prompts."""
    def __init__(self, model, max_seqlen=4096):
        self.model = model
        self.stepper = GraphTokenStepper(model, max_seqlen=max_seqlen)

    def backbone_tokenwise(self, model, ids, state_dtype=torch.float16):
        if model is not self.model or state_dtype != torch.float16:
            raise ValueError("Graph adapter model/state dtype differs from captured execution")
        return self.stepper.sequence(ids, reset=True)

    @torch.inference_mode()
    def generate_greedy(self, model, tokenizer, prompt, max_new_tokens=12, execution="tokenwise"):
        if model is not self.model or execution != "tokenwise":
            raise ValueError("Graph adapter implements only this model's tokenwise protocol")
        device = next(model.parameters()).device
        ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None]
        if ids.shape[1] == 0 or ids.shape[1]+max_new_tokens > self.stepper.max_seqlen:
            raise ValueError("Prompt/generation length is outside the configured graph sequence limit")
        self.stepper.reset()
        for position in range(ids.shape[1]):
            hidden = self.stepper.step(ids[:, position:position+1])
        generated = []
        # Match evaluation.generate_greedy, including the final state update.
        for _ in range(max_new_tokens):
            logits = model.lm_head(hidden[:, -1:])
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite graph generation logits")
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
            hidden = self.stepper.step(token)
        return tokenizer.decode(generated), generated, ids.shape[1], cache_bytes(self.stepper.cache)


@torch.inference_mode()
def check_greedy_parity(model, tokenizer, adapter):
    case = evaluation.synthetic_mk_cases("validation", samples_per_cell=2)[0]
    torch.cuda.synchronize()
    start = time.perf_counter()
    plain = evaluation.generate_greedy(model, tokenizer, case["prompt"], execution="tokenwise")
    torch.cuda.synchronize()
    plain_seconds = time.perf_counter()-start
    start = time.perf_counter()
    graph = adapter.generate_greedy(model, tokenizer, case["prompt"], execution="tokenwise")
    torch.cuda.synchronize()
    graph_seconds = time.perf_counter()-start
    return {"passed": plain == graph, "case_id": case["id"],
            "prompt_token_sha256_int64le": token_digest(tokenizer.encode(case["prompt"])),
            "prompt_tokens": plain[2], "plain_generated_ids": plain[1], "graph_generated_ids": graph[1],
            "plain_output": plain[0], "graph_output": graph[0],
            "plain_cache_bytes": plain[3], "graph_cache_bytes": graph[3],
            "plain_seconds": plain_seconds, "graph_seconds": graph_seconds,
            "timing_limit": "Single ordered run on a shared GPU; no dedicated throughput claim.",
            "scope": "One validation prompt; parity of native-vs-graph outputs, not a recall-quality result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", "--model-dir", dest="source_dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, default=Path("reports/tokenwise_graph_parity_v1.json"))
    parser.add_argument("--greedy-parity-report", type=Path,
                        help="Bind a prior passing driver greedy-parity receipt when evaluating")
    parser.add_argument("--check-greedy-parity", action="store_true",
                        help="Only compare native and graph greedy outputs on one public dev prompt")
    parser.add_argument("--suite", choices=("dev", "full"), default="dev")
    parser.add_argument("--seqlen", type=int)
    parser.add_argument("--nwin", type=int)
    parser.add_argument("--mk-samples-per-cell", type=int)
    parser.add_argument("--skip-mk", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--dataset-revision", default=WIKITEXT_REVISION)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.skip_mk and args.skip_ppl and not args.check_greedy_parity:
        parser.error("Select at least one evaluation task")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    parity = read_parity_receipt(args.parity_report)
    driver_hash = sha256_file(__file__)
    greedy_receipt = None
    if not args.check_greedy_parity:
        if args.greedy_parity_report is None:
            parser.error("--greedy-parity-report is required for graph evaluation")
        prior = json.loads(args.greedy_parity_report.read_text())
        if not prior.get("complete") or not prior.get("greedy_parity", {}).get("passed"):
            raise ValueError("Greedy parity report did not pass")
        if prior["implementation"]["driver_source_sha256"] != driver_hash:
            raise ValueError("Greedy parity report refers to different graph driver source")
        if prior["implementation"]["graph_source_sha256"] != parity["receipt"]["graph_source_sha256"]:
            raise ValueError("Greedy parity graph implementation differs")
        greedy_receipt = {"sha256": sha256_file(args.greedy_parity_report), "receipt": prior}
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    model = load_quantized_model(args.raw_dir) if args.raw_dir else load_source_model(args.source_dir)
    adapter = GraphEvaluationAdapter(model)
    implementation = {"name": "CUDA graph replay of native tokenwise steps",
                      "graph_source_sha256": parity["receipt"]["graph_source_sha256"],
                      "driver_source_sha256": driver_hash, "bitwise_parity": parity,
                      "greedy_parity": greedy_receipt, "capture_seconds": adapter.stepper.capture_seconds,
                      "parity_scope": "Source-model implementation checks; not a proof for every possible input or quantized weight value."}
    report = {"complete": False, "suite": args.suite, "execution": "tokenwise",
              "source_checkpoint_sha256": model._package_receipt["source_checkpoint_sha256"],
              "weight_format": "E8/W5 decoded to FP16" if args.raw_dir else "original weights cast to FP16",
              "raw_package": str(args.raw_dir) if args.raw_dir else None,
              "tokenizer_sha256": tokenizer.sha256, "environment": environment_receipt(),
              "runtime_source_sha256": sha256_file(Path(__file__).with_name("runtime.py")),
              "evaluation_source_sha256": sha256_file(Path(__file__).with_name("evaluation.py")),
              "package_receipt": model._package_receipt, "implementation": implementation}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.check_greedy_parity:
        report["greedy_parity"] = check_greedy_parity(model, tokenizer, adapter)
        report["complete"] = True
        report["gpu_memory"] = gpu_memory_receipt()
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report["greedy_parity"], indent=2, allow_nan=False), flush=True)
        if not report["greedy_parity"]["passed"]:
            raise SystemExit("Greedy parity failed; keep native tokenwise evaluation")
        return
    split = "validation" if args.suite == "dev" else "test"
    seqlen = args.seqlen or (1024 if args.suite == "dev" else 2048)
    nwin = args.nwin if args.nwin is not None else (4 if args.suite == "dev" else None)
    samples = args.mk_samples_per_cell or (2 if args.suite == "dev" else 8)
    if seqlen > adapter.stepper.max_seqlen:
        parser.error("Sequence exceeds model's audited 4096-token context")
    native_backbone = evaluation.backbone_tokenwise
    native_greedy = evaluation.generate_greedy
    evaluation.backbone_tokenwise = adapter.backbone_tokenwise
    evaluation.generate_greedy = adapter.generate_greedy
    try:
        if not args.skip_ppl:
            ids, dataset = load_wikitext_tokens(tokenizer, split, args.dataset_revision)
            windows = evaluation.ppl_windows(ids, seqlen, nwin)
            report["dataset"] = dataset
            report["ppl"] = evaluation.evaluate_ppl(model, windows, "tokenwise")
            report["ppl"]["coverage"] = "full split" if nwin is None else "fixed nonoverlapping subset"
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if not args.skip_mk:
            report["mk"] = evaluation.evaluate_mk(model, tokenizer, split, samples, "tokenwise")
    finally:
        evaluation.backbone_tokenwise = native_backbone
        evaluation.generate_greedy = native_greedy
    report["complete"] = True
    report["gpu_memory"] = gpu_memory_receipt()
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "ppl": report.get("ppl", {}).get("ppl"),
                      "mk": report.get("mk", {}).get("summary")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
