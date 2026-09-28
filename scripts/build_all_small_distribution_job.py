#!/usr/bin/env python3
"""Build and restore a local all-small distribution from a static source export.

This is CPU-only packaging. It does not publish, load an original checkpoint,
perform generation/PPL, or wait on/change the independent prototype training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA = 'bc554db936b13cbbeaeef5267d96b8d6ba7c183bf740c7e64e5f26ac7c478af8'
WEIGHT_SHA = '46be4e12d18ed36e33adf002962f0b7bcafe8478b249b628eca853930d43efcd'
WEIGHT_BYTES = 2660171471


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-dir', 'tokenizer', 'output', 'restored-dir', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('raw_dir', 'tokenizer', 'output', 'restored_dir', 'report'):
        setattr(args, name, getattr(args, name).resolve())
    if not (ROOT/'SNAPSHOT_PROVENANCE.json').is_file():
        raise RuntimeError('Run this job from an audited static source export')
    if sha(args.raw_dir/'manifest.json') != RAW_SHA:
        raise RuntimeError('Expected the audited all-small resolved package')
    for path in (args.output, args.output.with_name(args.output.name+'.building'), args.restored_dir, args.report):
        if path.exists():
            raise FileExistsError(path)
    if len({args.output, args.restored_dir, args.report}) != 3:
        raise ValueError('Job output paths must be distinct')
    for path in (args.output, args.restored_dir, args.report):
        if ROOT == path or ROOT in path.parents or args.raw_dir == path or args.raw_dir in path.parents:
            raise ValueError('Outputs must not modify source snapshot or raw input')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log_paths = {stage: args.report.with_name(args.report.stem+'.'+stage+'.log') for stage in ('build', 'restore')}
    if any(path.exists() for path in log_paths.values()):
        raise FileExistsError('Job log already exists')
    report = {'format': 'MAMBA2_ALL_SMALL_DISTRIBUTION_JOB_V1', 'complete': False,
        'pid': os.getpid(), 'started_at_unix': time.time(), 'script_sha256': sha(__file__),
        'source_snapshot': str(ROOT), 'snapshot_provenance_sha256': sha(ROOT/'SNAPSHOT_PROVENANCE.json'),
        'raw_manifest_sha256': RAW_SHA, 'stages': [], 'publication_performed': False,
        'gpu_used': False, 'quality_remeasured': False, 'archived_gpu_inference_verified': False}

    def save():
        temporary = args.report.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.report)

    def run(stage, command):
        row = {'stage': stage, 'command': command, 'started_at_unix': time.time(), 'log': str(log_paths[stage])}
        report['stage'] = stage
        report['stages'].append(row)
        save()
        with log_paths[stage].open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                     'PYTHONDONTWRITEBYTECODE': '1'})
            row['pid'] = process.pid
            save()
            row['exit_code'] = process.wait()
        row['ended_at_unix'] = time.time()
        save()
        if row['exit_code']:
            raise RuntimeError(f'{stage} failed; inspect {log_paths[stage]}')
        text = log_paths[stage].read_text()
        marker = text.rfind('\n{')
        result = json.loads(text[marker+1:] if marker >= 0 else text)
        if result.get('complete') is not True:
            raise RuntimeError(f'{stage} did not return a complete receipt')
        row['receipt'] = result
        save()
        return result

    with args.report.open('x') as stream:
        stream.write('{}\n')
    save()
    try:
        tool = str(ROOT/'scripts/package_release.py')
        build = [sys.executable, '-u', tool, 'build', '--raw-dir', str(args.raw_dir),
            '--tokenizer', str(args.tokenizer), '--output', str(args.output), '--split-bytes', '1500000000',
            '--quality-report', str(ROOT/'reports/small_compensation_v1_eval.json')]
        for name in ('SNAPSHOT_PROVENANCE.json', 'reports/source_model.json', 'reports/source_download.json',
                     'reports/all_small_resolved_v1_composition.json', 'reports/all_small_resolved_v1_huffman_identity.json'):
            build.extend(['--provenance-report', str(ROOT/name)])
        built = run('build', build)
        if built['container_bytes'] != WEIGHT_BYTES or built['container_sha256'] != WEIGHT_SHA:
            raise RuntimeError('Distribution does not contain the already audited weights container')
        restored = run('restore', [sys.executable, '-u', tool, 'restore', '--release-dir', str(args.output),
            '--output', str(args.restored_dir), '--expected-manifest-sha256', built['manifest_sha256']])
        if (restored['raw_manifest_sha256'] != RAW_SHA or restored['restored_raw_files'] != 118
                or restored['container_sha256'] != WEIGHT_SHA):
            raise RuntimeError('Restored distribution differs from the evaluated candidate')
        report.update(complete=True, stage='complete', release_manifest_sha256=built['manifest_sha256'],
            total_release_bytes=built['total_release_bytes'], container_bytes=built['container_bytes'],
            release_dir=str(args.output), restored_dir=str(args.restored_dir))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['ended_at_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
