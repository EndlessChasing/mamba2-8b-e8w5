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
from mamba_e8w5.huffman import Reader, pack_directory, verify_container, sha256_file
from scripts.package_release import validate_raw


def verify_pinned_identity(raw, container, expected_manifest_sha256, readback_receipt):
    """Bind verified container bytes to the pinned ledger, not mutable raw files."""
    raw, container = Path(raw), Path(container)
    manifest_path = raw/'manifest.json'
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError('Pinned resolved manifest changed')
    manifest = validate_raw(raw)
    expected = {name: (entry['bytes'], entry['sha256']) for name, entry in manifest['files'].items()}
    expected['manifest.json'] = (manifest_path.stat().st_size, expected_manifest_sha256)
    reader = Reader(container)
    actual = {name: (entry['original_bytes'], entry['sha256']) for name, entry in reader.files.items()}
    if actual != expected or reader.manifest['source_manifest_sha256'] != expected_manifest_sha256:
        raise ValueError('Container member identities differ from the pinned candidate ledger')
    if (readback_receipt.get('complete') is not True
            or readback_receipt['files_verified'] != len(expected)
            or container.stat().st_size != readback_receipt['actual_file_bytes']
            or sha256_file(container) != readback_receipt['sha256']
            or sha256_file(manifest_path) != expected_manifest_sha256):
        raise ValueError('Prior full readback does not describe these exact container bytes')
    return {'complete': True, 'members_matched_to_pinned_manifest': len(expected),
        'raw_file_ledger_revalidated': True, 'raw_manifest_sha256': expected_manifest_sha256,
        'container_sha256': readback_receipt['sha256'],
        'previous_full_readback_files': readback_receipt['files_verified'],
        'scope': 'Identity audit against pinned manifest and completed full readback; no new quality score.'}


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
    identity = verify_pinned_identity(raw, output, args.expected_manifest_sha256, receipt)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only container task unexpectedly initialized CUDA')
    receipt.update(raw_manifest_sha256=args.expected_manifest_sha256,
        composition_script_sha256=manifest['binding']['composition_script_sha256'],
        quality_report_sha256=manifest['quality']['report_sha256'],
        quality_arm=manifest['quality']['arm'],
        scope='Weights container including its raw manifest. Excludes tokenizer, software, licenses and outer distribution metadata.',
        elapsed_seconds=time.perf_counter()-started, script_sha256=sha256_file(__file__),
        pinned_candidate_identity=identity,
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
