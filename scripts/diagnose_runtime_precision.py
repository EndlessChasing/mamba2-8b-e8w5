#!/usr/bin/env python3
"""Compare source BF16 and FP16 on the frozen development PPL windows.

This is a numerical control in the public mamba_ssm runtime, not a validation
against NVIDIA's original Megatron execution. It does not tune any weights.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mamba_e8w5.calibration import WIKITEXT_REVISION, load_wikitext_tokens
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from mamba_e8w5.runtime import (MODEL_CONFIG, SentencePieceTokenizer,
    environment_receipt, load_source_model, sha256_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', WIKITEXT_REVISION)
    windows = ppl_windows(ids, 1024, 4)
    report = {'complete': False, 'purpose': 'Original source BF16 versus FP16 numerical control',
              'dataset': dataset, 'model_config': MODEL_CONFIG,
              'runtime_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'mamba_e8w5/runtime.py'),
              'script_sha256': sha256_file(__file__), 'environment': environment_receipt(),
              'variants': {}, 'limitations': ['Public mamba_ssm runtime; original Megatron runtime not tested.',
                  'Four development validation windows, not full test.',
                  'Prefill SSD scan; this is not a per-token BF16 cache experiment.']}
    model = load_source_model(args.source_dir, dtype=torch.bfloat16)
    report['source_receipt'] = model._package_receipt
    with torch.inference_mode():
        for label, dtype in [('bf16', torch.bfloat16), ('fp16', torch.float16)]:
            model.to(dtype=dtype)
            assert all(p.dtype == dtype for p in model.parameters())
            print(f'Evaluating original source: {label}', flush=True)
            report['variants'][label] = evaluate_ppl(model, windows, 'prefill', 64)
    baseline = json.loads(args.baseline_report.read_text())
    actual = report['variants']['fp16']
    assert dataset == baseline['dataset']
    assert [w['token_sha256_int64le'] for w in actual['windows']] == [w['token_sha256_int64le'] for w in baseline['ppl']['windows']]
    difference = actual['ppl'] - baseline['ppl']['ppl']
    assert abs(difference) < 1e-5, ('FP16 endpoint failed reproduction', difference)
    fp16 = actual['nll'] / actual['target_tokens']
    bf16 = report['variants']['bf16']['nll'] / report['variants']['bf16']['target_tokens']
    report['comparison'] = {'fp16_endpoint_ppl_difference': difference,
        'fp16_minus_bf16_nll_per_token': fp16-bf16,
        'fp16_vs_bf16_ppl_percent': 100 * math.expm1(fp16-bf16)}
    report['complete'] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['comparison']), flush=True)


if __name__ == '__main__':
    main()
