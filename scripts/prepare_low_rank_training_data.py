#!/usr/bin/env python3
"""Prepare256 fresh TRAIN windows; CPU only, no model loading or Hessians."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.runtime import (
    SentencePieceTokenizer, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256,
    sha256_file, token_digest,
)

SEQLEN = 2048
NWIN = 256
EPOCHS = 4
SEED = 20260928
TRAIN_TOKENS = 2533678
BOUND = {
    'calibration/v1/manifest.json':
        '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab',
    'training_data/small_compensation_v1/manifest.json':
        '88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709',
    'reports/projection_crossmoment_pilot_v1.json':
        '034a161df40fa4dac5be99b25545f9087da6695accc414a852fafe0e47228f87',
    'training_data/prototype_compensation_v1/manifest.json':
        'a485424d1b7894eb1d5a6361ab4c25fdb4f6e5bedc56d7a15a7e4add2c454bcd',
}
CODE_BOUND = {
    'mamba_e8w5/runtime.py':
        'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/calibration.py':
        'b5a7356a2ab769538940aed218653a092a7055aa26a9ec6f62d3783b9ac78ee5',
}
EXPECTED_COUNTS = {'original_calibration': 32, 'all_small_training': 256,
    'crossmoment_fit': 32, 'crossmoment_heldout': 16, 'prototype_training': 32}


def previous_intervals(documents):
    cal, small, pilot, prototype = documents
    excluded = {'original_calibration': cal['starts'], 'all_small_training': small['starts'],
        'crossmoment_fit': [w['start'] for w in pilot['fit_windows']],
        'crossmoment_heldout': [w['start'] for w in pilot['heldout_windows']],
        'prototype_training': prototype['starts']}
    if any(d.get('complete') is not True for d in documents):
        raise ValueError('Prior provenance is incomplete')
    for name, starts in excluded.items():
        if (not isinstance(starts, list) or len(starts) != EXPECTED_COUNTS[name]
                or len(set(starts)) != len(starts)
                or any(type(s) is not int or s < 0 or s + SEQLEN > TRAIN_TOKENS for s in starts)):
            raise ValueError(f'Invalid prior intervals: {name}')
    return excluded


def select_starts(excluded):
    """Fixed half-open interval exclusion and spread selection; no data scoring."""
    if {key: len(value) for key, value in excluded.items()} != EXPECTED_COUNTS:
        raise ValueError('Prior interval family counts differ')
    intervals = [start for values in excluded.values() for start in values]
    grid = list(range(0, TRAIN_TOKENS - SEQLEN + 1, SEQLEN))
    eligible = [s for s in grid if all(s + SEQLEN <= old or old + SEQLEN <= s for old in intervals)]
    if len(grid) != 1237 or len(eligible) != 839:
        raise ValueError('Declared eligible-grid counts differ')
    starts = [eligible[i * 838 // 255] for i in range(NWIN)]
    if (len(set(starts)) != NWIN or starts[0] != 8192 or starts[-1] != 2523136
            or min(b - a for a, b in zip(starts, starts[1:])) != 6144):
        raise ValueError('Declared selected grid geometry differs')
    overlaps = {name: sum(max(0, min(s + SEQLEN, old + SEQLEN) - max(s, old))
        for s in starts for old in values) for name, values in excluded.items()}
    internal_overlap = sum(max(0, a + SEQLEN - b) for a, b in zip(starts, starts[1:]))
    if internal_overlap or any(overlaps.values()):
        raise ValueError('Selected stored token intervals overlap')
    return starts, {'grid_blocks': len(grid), 'excluded_grid_blocks': len(grid) - len(eligible),
        'eligible_blocks': len(eligible), 'selected_blocks': NWIN,
        'internal_overlap_tokens': internal_overlap,
        'all_previous_fitting_overlap_tokens': sum(overlaps.values()),
        'previous_family_overlap_tokens': overlaps, 'minimum_start_gap': 6144}


def training_schedule():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    return [window for _ in range(EPOCHS)
        for window in torch.randperm(NWIN, generator=generator, device='cpu').tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'models/source')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'training_data/low_rank_compensation_v1')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/LOW_RANK_COMPENSATION_PROTOCOL.md')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("Run CPU preparation with CUDA_VISIBLE_DEVICES=''")
    if torch.cuda.is_initialized():
        raise ValueError('CUDA was initialized before CPU data preparation')
    torch.set_num_threads(2)
    if args.out_dir.exists():
        raise FileExistsError('Prepared data directory must be new; preserve prior receipts')
    bound = {**BOUND, **CODE_BOUND}
    for name, digest in bound.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f'Pinned provenance differs: {name}')
    protocol_sha = sha256_file(args.protocol)
    source_sha = sha256_file(__file__)
    documents = [json.loads((ROOT / name).read_text()) for name in BOUND]
    excluded = previous_intervals(documents)
    starts, overlap = select_starts(excluded)

    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'train', WIKITEXT_REVISION)
    if (ids.device.type != 'cpu' or ids.dtype != torch.int64 or tuple(ids.shape) != (TRAIN_TOKENS,)
            or any(dataset != d['dataset'] for d in documents)
            or dataset['split'] != 'train' or dataset['tokenizer_sha256'] != TOKENIZER_SHA256
            or documents[0]['source_checkpoint_sha256'] != SOURCE_CHECKPOINT_SHA256):
        raise ValueError('Pinned TRAIN token stream or source identity differs')
    windows = torch.stack([ids[start:start + SEQLEN] for start in starts])
    if tuple(windows.shape) != (NWIN, SEQLEN):
        raise ValueError('Prepared token shape differs')
    schedule = training_schedule()
    if len(schedule) != 1024 or any(sorted(schedule[i:i + NWIN]) != list(range(NWIN))
            for i in range(0, len(schedule), NWIN)):
        raise ValueError('Seeded schedule is not four permutations')

    args.out_dir.mkdir(parents=True, exist_ok=False)
    token_path = args.out_dir / 'training_tokens.pt'
    with token_path.open('xb') as stream:
        torch.save(windows, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(token_path, map_location='cpu', weights_only=True)
    if (not isinstance(restored, torch.Tensor) or restored.dtype != torch.int64
            or restored.device.type != 'cpu' or restored.shape != windows.shape
            or not torch.equal(windows, restored)):
        raise ValueError('Stored training windows differ after readback')
    if torch.cuda.is_initialized():
        raise ValueError('CPU data preparation unexpectedly initialized CUDA')
    for name, digest in bound.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f'Provenance changed during preparation: {name}')
    if sha256_file(args.protocol) != protocol_sha or sha256_file(__file__) != source_sha:
        raise ValueError('Preparation code or protocol changed during execution')

    manifest = {'format': 'MAMBA2_LOW_RANK_TRAIN_WINDOWS_V1', 'complete': True,
        'split': 'train', 'evaluation_data_used': False, 'dataset': dataset,
        'source_checkpoint_sha256': SOURCE_CHECKPOINT_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
        'nwin': NWIN, 'seqlen': SEQLEN, 'starts': starts,
        'selection': '2048-token grid excluding all prior intervals; eligible[floor(i*838/255)], i=0..255',
        'excluded_starts': excluded, 'excluded_manifests_sha256': BOUND, 'overlap': overlap,
        'unique_stored_window_tokens': NWIN * SEQLEN,
        'unique_input_positions_per_pass': NWIN * (SEQLEN - 1),
        'targets_per_pass': NWIN * (SEQLEN - 1),
        'four_pass_target_exposures': EPOCHS * NWIN * (SEQLEN - 1),
        'schedule_seed': SEED, 'schedule': schedule,
        'schedule_generator': 'four successive CPU torch.randperm(256), separate from factor initialization',
        'token_windows_file': token_path.name, 'token_windows_file_bytes': token_path.stat().st_size,
        'token_windows_file_sha256': sha256_file(token_path),
        'selected_tokens_sha256_int64le': token_digest(windows.flatten().numpy()),
        'windows': [{'start': start, 'stored_tokens': SEQLEN, 'targets': SEQLEN - 1,
            'token_sha256_int64le': token_digest(window.numpy())} for start, window in zip(starts, windows)],
        'preparation_source_sha256': source_sha,
        'calibration_loader_source_sha256': CODE_BOUND['mamba_e8w5/calibration.py'],
        'runtime_source_sha256': CODE_BOUND['mamba_e8w5/runtime.py'],
        'protocol_sha256': protocol_sha,
        'environment': {'python': sys.version, 'torch': torch.__version__,
            'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_initialized': False},
        'disk_roundtrip_equal': True}
    temporary = args.out_dir / 'manifest.json.partial'
    with temporary.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    manifest_path = args.out_dir / 'manifest.json'
    os.link(temporary, manifest_path)
    temporary.unlink()
    print(json.dumps({'complete': True, 'manifest_sha256': sha256_file(manifest_path),
        'token_windows_file_sha256': manifest['token_windows_file_sha256'],
        'selected_tokens_sha256_int64le': manifest['selected_tokens_sha256_int64le'],
        'overlap': overlap, 'targets_per_pass': manifest['targets_per_pass'],
        'four_pass_target_exposures': manifest['four_pass_target_exposures'],
        'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
