#!/usr/bin/env python3
"""Pack and fully read back static resolved weights; no live software snapshot."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.huffman import pack_directory, verify_container, sha256_file
from scripts.package_release import validate_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    raw, output, report_path = (p.resolve() for p in (args.raw_dir, args.output, args.report))
    for path in (output, output.with_name(output.name+'.partial'), report_path):
        if path.exists():
            raise FileExistsError(path)
        if raw == path or raw in path.parents:
            raise ValueError('Output/report must be outside the immutable resolved raw directory')
    manifest_path = raw/'manifest.json'
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise ValueError('Resolved manifest differs from requested candidate')
    manifest = validate_raw(raw)
    if manifest.get('format') != 'MAMBA2_E8W5_ALL_SMALL_RESOLVED_V1':
        raise ValueError('This wrapper supports only the evaluated all-small resolved format')
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    space = os.statvfs(output.parent)
    available = space.f_bavail*space.f_frsize
    if available < manifest['raw_package_bytes']+(256 << 20):
        raise RuntimeError('Insufficient free disk space for this container')
    started = time.perf_counter()
    pack_directory(raw, output)
    receipt = verify_container(output, trusted_directory=raw)
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise RuntimeError('Resolved manifest changed during packing')
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only container task unexpectedly initialized CUDA')
    receipt.update(raw_manifest_sha256=args.expected_manifest_sha256,
        composition_script_sha256=manifest['binding']['composition_script_sha256'],
        quality_report_sha256=manifest['quality']['report_sha256'],
        quality_arm=manifest['quality']['arm'],
        scope='Weights container including its raw manifest. Excludes tokenizer, software, licenses and outer distribution metadata.',
        elapsed_seconds=time.perf_counter()-started, script_sha256=sha256_file(__file__),
        huffman_source_sha256=sha256_file(ROOT/'mamba_e8w5/huffman.py'),
        cuda_initialized=False, quality_remeasured=False, complete_release_built=False,
        available_disk_bytes_before=available)
    with report_path.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete': True, 'container_bytes': receipt['actual_file_bytes'],
                      'report_sha256': sha256_file(report_path)}), flush=True)


if __name__ == '__main__':
    main()
