"""Optional CUDA graph implementation of the existing tokenwise FP16 protocol.

The first token executes native prefill normally. Every later token replays a
captured native single-token step, including all FP16 state writes. Warmup and
capture states are cleared before user tokens. No model arithmetic is replaced.
This implementation is opt-in and must pass its own bitwise parity receipt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .runtime import (SentencePieceTokenizer, backbone_tokenwise, cache_bytes,
                      environment_receipt, gpu_memory_receipt, load_source_model,
                      make_cache, sha256_file, token_digest)


class GraphTokenStepper:
    """One native Mamba sequence with reusable fixed-address FP16 state buffers."""
    @torch.inference_mode()
    def __init__(self, model, max_seqlen=4096, warmup_steps=3):
        self.model = model
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("CUDA graph stepper requires a CUDA model")
        if max_seqlen < 2 or warmup_steps < 1:
            raise ValueError("Invalid graph sequence or warmup length")
        self.max_seqlen = int(max_seqlen)
        self.cache = make_cache(model, self.max_seqlen, batch_size=1, state_dtype=torch.float16)
        for pair in self.cache.key_value_memory_dict.values():
            if any(tensor.dtype != torch.float16 for tensor in pair):
                raise ValueError("Graph protocol requires FP16 convolution and SSM cache")
        self.token = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        self.position = 0
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        with torch.cuda.stream(self.capture_stream):
            # Compile/warm native first-token prefill and steady-state kernels.
            self.cache.seqlen_offset = 0
            self.model.backbone(self.token, inference_params=self.cache)
            self.cache.seqlen_offset = 1
            for _ in range(warmup_steps):
                self.model.backbone(self.token, inference_params=self.cache)
        self.capture_stream.synchronize()
        # Python's offset branch is fixed at >0 throughout this graph.
        self.cache.seqlen_offset = 1
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.output = self.model.backbone(self.token, inference_params=self.cache)
        torch.cuda.current_stream(self.device).wait_stream(self.capture_stream)
        self.reset()
        torch.cuda.synchronize(self.device)
        self.capture_seconds = time.perf_counter()-started

    @torch.inference_mode()
    def reset(self):
        for pair in self.cache.key_value_memory_dict.values():
            for tensor in pair:
                tensor.zero_()
        self.position = 0
        self.cache.seqlen_offset = 0

    @torch.inference_mode()
    def step(self, token):
        if self.position >= self.max_seqlen:
            raise ValueError("Sequence exceeds the graph stepper's configured maximum")
        if isinstance(token, int):
            self.token.fill_(token)
        else:
            if token.numel() != 1:
                raise ValueError("Graph stepper accepts exactly one token at a time")
            self.token.copy_(token.reshape(1, 1))
        self.cache.seqlen_offset = self.position
        if self.position == 0:
            result = self.model.backbone(self.token, inference_params=self.cache)
        else:
            self.graph.replay()
            result = self.output
        self.position += 1
        # Captured output storage is reused; callers may retain every hidden state.
        return result.clone()

    @torch.inference_mode()
    def sequence(self, ids, reset=True):
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
            raise ValueError("Expected nonempty batch-one input IDs")
        if reset:
            self.reset()
        output = [self.step(ids[:, position:position+1]) for position in range(ids.shape[1])]
        return torch.cat(output, dim=1), self.cache


def _timed(function, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, time.perf_counter()-started


@torch.inference_mode()
def parity_and_timing(model, ids):
    """Compare all hidden outputs and each layer's two caches after real tokens."""
    device = next(model.parameters()).device
    (plain_hidden, plain_cache), plain_seconds = _timed(
        lambda: backbone_tokenwise(model, ids, torch.float16), device)
    stepper = GraphTokenStepper(model, max_seqlen=max(4096, ids.shape[1]+1))
    cleared_after_capture = all(not bool(torch.count_nonzero(tensor))
                                for pair in stepper.cache.key_value_memory_dict.values() for tensor in pair)
    if not cleared_after_capture:
        raise RuntimeError("Warmup/capture state was not completely reset")
    (graph_hidden, graph_cache), graph_seconds = _timed(lambda: stepper.sequence(ids), device)
    cache_rows = []
    for index in sorted(plain_cache.key_value_memory_dict):
        for name, first, second in zip(("conv", "ssm"), plain_cache.key_value_memory_dict[index],
                                       graph_cache.key_value_memory_dict[index]):
            cache_rows.append({"layer": index, "cache": name, "dtype": str(second.dtype),
                               "shape": list(second.shape), "bitwise_equal": torch.equal(first, second),
                               "max_absolute_error": float((first.float()-second.float()).abs().max())})
    hidden_equal = torch.equal(plain_hidden, graph_hidden)
    all_finite = bool(torch.isfinite(plain_hidden).all() and torch.isfinite(graph_hidden).all())
    report = {"tokens": ids.shape[1], "hidden_bitwise_equal": hidden_equal,
              "all_caches_bitwise_equal": all(row["bitwise_equal"] for row in cache_rows),
              "all_hidden_finite": all_finite, "capture_state_reset_verified": cleared_after_capture,
              "hidden_max_absolute_error": float((plain_hidden.float()-graph_hidden.float()).abs().max()),
              "hidden_relative_error": float((plain_hidden.float()-graph_hidden.float()).norm()
                                               / plain_hidden.float().norm().clamp_min(1e-30)),
              "cache_tensors": cache_rows, "cache_bytes": cache_bytes(graph_cache),
              "plain_seconds": plain_seconds, "capture_seconds": stepper.capture_seconds,
              "graph_seconds": graph_seconds, "steady_sequence_speedup": plain_seconds/graph_seconds,
              "plain_tokens_per_second": ids.shape[1]/plain_seconds,
              "graph_tokens_per_second": ids.shape[1]/graph_seconds,
              "timing_limit": "Single ordered measurement on a shared GPU; includes normal first-token prefill and output copies, excludes graph construction.",
              "state_rule": "First token uses native prefill; remaining native steps write FP16 conv/SSM caches after every token."}
    # Reusing the same stepper must not leak state from a previous prompt.
    repeated, _ = stepper.sequence(ids)
    report["second_sequence_hidden_bitwise_equal"] = torch.equal(graph_hidden, repeated)
    report["second_sequence_caches_bitwise_equal"] = all(
        torch.equal(a, b) for index, first in plain_cache.key_value_memory_dict.items()
        for a, b in zip(first, stepper.cache.key_value_memory_dict[index]))
    report["passed"] = all(report[field] for field in (
        "hidden_bitwise_equal", "all_caches_bitwise_equal", "all_hidden_finite", "capture_state_reset_verified",
        "second_sequence_hidden_bitwise_equal", "second_sequence_caches_bitwise_equal"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    if not 2 <= args.tokens <= 4096:
        parser.error("--tokens must be between 2 and 4096")
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    model = load_source_model(args.source_dir)
    sentence = "The spacecraft crossed the quiet sky. The engineer checked the instrument readings and recorded the result in a notebook. "
    ids = tokenizer.encode(sentence*512)[:args.tokens]
    report = {"complete": False, "implementation": "native tokenwise CUDA graph",
              "runtime_source_sha256": sha256_file(Path(__file__).with_name("runtime.py")),
              "graph_source_sha256": sha256_file(__file__), "tokenizer_sha256": tokenizer.sha256,
              "package_receipt": model._package_receipt, "environment": environment_receipt(),
              "token_sha256_int64le": token_digest(ids)}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    report["parity"] = parity_and_timing(model, torch.tensor([ids], device="cuda"))
    report["gpu_memory"] = gpu_memory_receipt()
    report["complete"] = True
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report["parity"].items() if key != "cache_tensors"},
                     indent=2, allow_nan=False), flush=True)
    if not report["parity"]["passed"]:
        raise SystemExit("CUDA graph did not pass bitwise parity; keep native tokenwise execution")


if __name__ == "__main__":
    main()
