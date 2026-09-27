"""Collect disjoint WikiText TRAIN covariances for all Mamba projection inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .runtime import (MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, SentencePieceTokenizer,
                      environment_receipt, gpu_memory_receipt, load_source_model, sha256_file, token_digest)

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def window_starts(token_count, seqlen, nwin, offset=0):
    """Evenly spread, nonoverlapping windows; never silently reuse short data."""
    usable = token_count - offset
    if seqlen <= 0 or nwin <= 0 or usable < seqlen * nwin:
        raise ValueError("Insufficient tokens for the requested nonoverlapping windows")
    if nwin == 1:
        return [offset]
    span = usable - seqlen
    starts = [offset + (index * span) // (nwin-1) for index in range(nwin)]
    if any(b-a < seqlen for a, b in zip(starts, starts[1:])):
        raise ValueError("Calibration windows overlap")
    return starts


def load_wikitext_tokens(tokenizer, split, revision=WIKITEXT_REVISION):
    from datasets import load_dataset
    kwargs = {"revision": revision} if revision else {}
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, **kwargs)
    text = "\n\n".join(dataset["text"])
    ids = tokenizer.encode(text)
    metadata = {
        "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
        "split": split, "revision_argument": revision, "dataset_fingerprint": dataset._fingerprint,
        "document_join": "two newline characters", "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_stream_sha256_int64le": token_digest(ids), "total_tokens": len(ids),
        "tokenizer_sha256": tokenizer.sha256, "automatic_special_tokens": False,
    }
    return torch.tensor(ids, dtype=torch.long), metadata


@torch.inference_mode()
def collect_hessians(model, windows, out_dir):
    """FP32 X.T@X sums on GPU; save one normalized Hessian at a time."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    accum, counts, hooks = {}, {}, []
    started = time.time()
    device = next(model.parameters()).device
    for block in model.backbone.layers:
        block.mixer.use_mem_eff_path = False

    def hook_for(name):
        def hook(_module, args):
            values = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            if name not in accum:
                accum[name] = torch.zeros((values.shape[-1], values.shape[-1]),
                                          device=values.device, dtype=torch.float32)
                counts[name] = 0
            accum[name].addmm_(values.T, values)
            counts[name] += values.shape[0]
        return hook

    for index, layer in enumerate(model.backbone.layers):
        for which in ("in_proj", "out_proj"):
            module = getattr(layer.mixer, which)
            hooks.append(module.register_forward_pre_hook(hook_for(f"layer{index}.{which}")))
    try:
        for index, window in enumerate(windows):
            model.backbone(window[None].to(device))  # Never allocate 256K-vocabulary logits.
            if (index+1) % 4 == 0 or index == 0:
                print(f"[calibration] {index+1}/{len(windows)} windows, "
                      f"{time.time()-started:.1f}s", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if len(accum) != 2 * len(model.backbone.layers):
        raise RuntimeError("Projection hooks were bypassed or did not cover every layer")
    expected_tokens = sum(len(window) for window in windows)
    if set(counts.values()) != {expected_tokens}:
        raise RuntimeError(f"Inconsistent calibration hook counts: {counts}")
    files = {}
    for name in list(accum):
        hessian = accum.pop(name).div_(counts[name]).cpu()
        if not torch.isfinite(hessian).all():
            raise RuntimeError(f"Nonfinite calibration Hessian: {name}")
        path = directory / f"{name}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(hessian, temporary)
        temporary.replace(path)
        files[name] = {"file": path.name, "bytes": path.stat().st_size,
                       "sha256": sha256_file(path), "shape": list(hessian.shape),
                       "diagonal_min": float(hessian.diag().min()),
                       "diagonal_max": float(hessian.diag().max()), "tokens": counts[name]}
        del hessian
    return {"matrices": files, "counts": counts, "elapsed_seconds": time.time()-started,
            "accumulator_dtype": "float32", "formula": "X.T @ X / token_count",
            "fused_projection_path": False, "calibration_tokens": expected_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", "--model-dir", dest="source_dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nwin", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dataset-revision", default=WIKITEXT_REVISION)
    parser.add_argument("--checkpoint-sha256", default=SOURCE_CHECKPOINT_SHA256)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Calibration manifest already exists; use a new output directory")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, "train", args.dataset_revision)
    starts = window_starts(len(ids), args.seqlen, args.nwin)
    windows = [ids[start:start+args.seqlen] for start in starts]
    window_path = args.out_dir / "calibration_tokens.pt"
    torch.save(torch.stack(windows), window_path)
    initial = {"complete": False, "model_config": MODEL_CONFIG, "dataset": dataset,
               "source_checkpoint_sha256": args.checkpoint_sha256,
               "nwin": args.nwin, "seqlen": args.seqlen, "starts": starts,
               "selected_tokens_sha256_int64le": token_digest(torch.stack(windows).flatten().numpy()),
               "token_windows_file": window_path.name, "token_windows_file_sha256": sha256_file(window_path),
               "calibration_split": "train", "evaluation_data_used": False,
               "environment": environment_receipt(),
               "runtime_source_sha256": sha256_file(Path(__file__).with_name("runtime.py")),
               "calibration_source_sha256": sha256_file(__file__)}
    manifest_path.write_text(json.dumps(initial, indent=2)+"\n")
    model = load_source_model(args.source_dir, expected_sha256=args.checkpoint_sha256)
    results = collect_hessians(model, windows, args.out_dir)
    initial.update(results)
    initial["complete"] = True
    initial["hessian_files_bytes"] = sum(x["bytes"] for x in results["matrices"].values())
    initial["gpu_memory"] = gpu_memory_receipt()
    manifest_path.write_text(json.dumps(initial, indent=2)+"\n")
    print(json.dumps({"complete": True, "matrices": len(results["matrices"]),
                      "tokens": results["calibration_tokens"],
                      "elapsed_seconds": results["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
