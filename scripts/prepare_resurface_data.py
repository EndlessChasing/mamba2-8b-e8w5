#!/usr/bin/env python3
"""Explicit CPU preparation of one protocol-bound Resurface split."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba_e8w5 import resurface_data as data
from mamba_e8w5.runtime import SentencePieceTokenizer

PROTOCOL_SHA = '469063147d892282a2bdad8558afca8466d6f365f1efe14ecbba7417e9857dd6'
HELPER_SHA = 'e813a382c25e99529eaddedb78aecd122b3c875903831f485c1cdf054ef7760c'
RUNTIME_SHA = 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'dev', 'confirm'), required=True)
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'training_data/resurface_readapted_v1')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report = args.report.resolve()
    if args.report.parent != ROOT/'reports' or args.report.exists():
        raise ValueError('Fresh report under reports required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU preparation requires CUDA hidden and uninitialized')
    torch.set_num_threads(8)
    protocol = ROOT/'docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md'
    pins = {str(protocol): PROTOCOL_SHA, str(Path(data.__file__)): HELPER_SHA,
            str(ROOT/'mamba_e8w5/runtime.py'): RUNTIME_SHA,
            str(Path(__file__).resolve()): data.sha_file(__file__)}
    report = {'format': 'MAMBA2_RESURFACE_DATA_PREPARATION_V1', 'complete': False,
              'split': args.split, 'code_sha256': pins, 'gpu_used': False}
    started = time.monotonic()
    try:
        for path, digest in pins.items():
            if data.sha_file(path) != digest:
                raise ValueError('Preparation input changed: '+path)
        tokenizer = SentencePieceTokenizer(ROOT/'models/source')
        manifest = data.prepare_split(args.dataset_root, args.split, tokenizer,
            protocol_path=protocol, protocol_sha256=PROTOCOL_SHA,
            tokenizer_sha256=data.TOKENIZER_SHA)
        manifest_path = args.dataset_root/args.split/'manifest.json'
        manifest_sha = data.sha_file(manifest_path)
        if args.split == 'train':
            loaded, rows, tokens = data.load_training(args.dataset_root, manifest_sha, tokenizer)
        else:
            loaded, rows, tokens = data.load_evaluation(args.dataset_root, args.split, manifest_sha, tokenizer)
        if loaded != manifest or loaded['protocol_sha256'] != PROTOCOL_SHA:
            raise ValueError('Manifest protocol/readback differs')
        for path, digest in pins.items():
            if data.sha_file(path) != digest:
                raise ValueError('Preparation input changed: '+path)
        if torch.cuda.is_initialized():
            raise ValueError('CUDA unexpectedly initialized')
        report.update(complete=True, manifest=manifest, manifest_path=str(manifest_path),
            manifest_sha256=manifest_sha, manifest_bytes=manifest_path.stat().st_size,
            exact_raw_and_tokenized_roundtrip=True, cuda_initialized=False)
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('complete', 'split', 'manifest_sha256', 'manifest_bytes')}), flush=True)


if __name__ == '__main__':
    main()
