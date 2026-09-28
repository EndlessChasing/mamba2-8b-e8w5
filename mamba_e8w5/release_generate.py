"""Generate with the restored v0.2 model and its mandatory soft Resurface adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import release_runtime as release
from . import runtime
from .evaluation import generate_greedy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--prompt')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repeat', action='store_true', help='Require exact generated IDs on a fresh-cache repeat')
    parser.add_argument('--verify-only', action='store_true', help='CPU package/data/software verification; no native model')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error('Report already exists; choose a fresh path')
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    if not args.verify_only and args.prompt is None:
        parser.error('--prompt is required for generation')
    started = time.monotonic()
    torch.set_num_threads(8)
    report = {'format': 'MAMBA2_AXIS_RESURFACE_GENERATION_V1', 'complete': False}
    model = None
    try:
        if args.verify_only:
            _, receipt = release.verify_package(args.model_dir, args.expected_manifest_sha256)
            report.update(package=receipt, mode='verify-only', cuda_initialized=torch.cuda.is_initialized(),
                          decoded_weights_verified=False)
        else:
            if torch.device(args.device).type != 'cuda':
                parser.error('Native Mamba inference requires a CUDA device; --verify-only is CPU-only')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            torch.cuda.reset_peak_memory_stats(args.device)
            tokenizer = release.load_tokenizer(args.model_dir)
            tokens = tokenizer.encode(args.prompt)
            if not tokens or len(tokens) + args.max_new_tokens > 4096:
                raise ValueError('Prompt must be nonempty and prompt plus generation must not exceed4096 tokens')
            model = release.load_model(args.model_dir, args.device, args.expected_manifest_sha256)
            cache = runtime.make_cache(model, len(tokens) + args.max_new_tokens, state_dtype=torch.float16)
            tensors = [v for pair in cache.key_value_memory_dict.values() for v in pair]
            if len(tensors) != 112 or any(v.dtype != torch.float16 for v in tensors):
                raise ValueError('Native cache does not cover112 FP16 tensors')
            cache_receipt = {'tensor_count': len(tensors), 'dtype': 'float16',
                             'bytes': runtime.cache_bytes(cache), 'fresh_for_each_generation': True}
            del cache, tensors
            output, generated, prompt_count, cache_bytes = generate_greedy(
                model, tokenizer, args.prompt, args.max_new_tokens, 'prefill')
            repeated = None
            if args.repeat:
                replay = generate_greedy(model, tokenizer, args.prompt, args.max_new_tokens, 'prefill')
                if replay != (output, generated, prompt_count, cache_bytes):
                    raise RuntimeError('Fresh-cache generation repeat differs')
                repeated = True
            if cache_bytes != cache_receipt['bytes']:
                raise ValueError('Generation cache byte accounting differs')
            report.update(mode='generation', prompt=args.prompt, completion=output, generated_ids=generated,
                prompt_tokens=prompt_count, prompt_token_sha256_int64le=runtime.token_digest(tokens),
                cache=cache_receipt, fresh_cache_repeat_identical=repeated,
                package=model._package_receipt, final_model_audit=release.audit_loaded_model(model),
                environment=runtime.environment_receipt(), gpu_memory=runtime.gpu_memory_receipt(),
                execution='native prefill then recurrent FP16 cache; greedy full-vocabulary argmax',
                soft_adapter_enabled_for_all_tokens=True)
        report['complete'] = True
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        if model is not None:
            release.close_model(model)
        report['elapsed_seconds'] = time.monotonic() - started
        report['script_sha256'] = runtime.sha256_file(__file__)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x') as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
