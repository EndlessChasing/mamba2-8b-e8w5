#!/usr/bin/env python3
"""Read-only low-rank job status; optional wait is bounded to60 seconds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
JOB_FORMAT = 'MAMBA2_LOW_RANK_JOB_V1'
MAX_REPORT_BYTES = 256 * 1024 * 1024


def read_process(pid, *, proc_root=Path('/proc')):
    """Capture Linux process identity without treating PID existence as identity."""
    if not proc_root.is_dir():
        return {'pid': pid, 'present': None, 'reason': 'proc_unavailable'}
    if type(pid) is not int or pid <= 0:
        return {'pid': pid, 'present': False, 'reason': 'invalid_or_missing_pid'}
    try:
        directory = proc_root / str(pid)
        # /proc stat comm may contain spaces/parentheses; split after its last ')'.
        before = (directory / 'stat').read_text()
        fields = before[before.rfind(')') + 2:].split()
        state, start_ticks = fields[0], int(fields[19])
        argv = [v.decode(errors='replace') for v in (directory / 'cmdline').read_bytes().split(b'\0') if v]
        boot_id = (proc_root / 'sys/kernel/random/boot_id').read_text().strip()
        after = (directory / 'stat').read_text()
        after_fields = after[after.rfind(')') + 2:].split()
        if int(after_fields[19]) != start_ticks:
            return {'pid': pid, 'present': True, 'reason': 'process_changed_during_read'}
        return {'pid': pid, 'present': True, 'state': after_fields[0],
            'start_ticks': start_ticks, 'boot_id': boot_id, 'argv': argv,
            'live': after_fields[0] not in ('Z', 'X', 'x')}
    except (OSError, ValueError, IndexError) as error:
        return {'pid': pid, 'present': False, 'reason': type(error).__name__}


def verified_process(record, *, proc_root=Path('/proc')):
    observed = read_process(record.get('pid'), proc_root=proc_root)
    keys = ('pid', 'start_ticks', 'boot_id', 'argv')
    comparable = (type(record.get('start_ticks')) is int
        and isinstance(record.get('boot_id'), str) and bool(record['boot_id'])
        and isinstance(record.get('argv'), list) and bool(record['argv']))
    matches = bool(comparable and observed.get('present')
        and all(observed.get(k) == record.get(k) for k in keys))
    return {'pid': record.get('pid'), 'present': observed.get('present', False),
        'state': observed.get('state'), 'identity_matches': matches,
        'matching_live_process': matches and observed.get('live', False),
        'start_ticks': observed.get('start_ticks'),
        'reason': observed.get('reason') or (None if matches else 'missing_or_mismatched_identity')}


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError('Receipt exceeds bounded reader limit')
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('Receipt must be a JSON object')
        return value, None
    except (OSError, ValueError) as error:
        return None, f'{type(error).__name__}: {error}'


def tail_text(path, maximum_bytes=65536):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - maximum_bytes))
            return stream.read(maximum_bytes).decode(errors='replace')
    except OSError:
        return ''


def evaluation_progress(evaluation, log_text):
    evaluation = evaluation or {}
    results = evaluation.get('results', {})
    markers = list(re.finditer(r'\[[^\]\r\n]*evaluation\] Scoring (\S+)', log_text))
    arm = markers[-1].group(1) if markers else None
    current_text = log_text[markers[-1].end():] if markers else ''
    windows = list(re.finditer(r'\[PPL ([^\]]+)\] (\d+)/(\d+), ppl=([^,\s]+)', current_text))
    completed = len(results.get(arm, {}).get('windows', [])) if arm else 0
    progress = {'complete': evaluation.get('complete', False), 'current_or_last_started_arm': arm,
        'completed_arm_ppl': {name: value.get('ppl') for name, value in results.items()},
        'current_arm_completed_windows': completed, 'current_arm_total_windows': 130,
        'progress_source': 'completed receipt' if completed else 'log; may lag',
        'comparison': evaluation.get('comparison')}
    if windows and not completed:
        last = windows[-1]
        progress.update(current_arm_completed_windows=int(last.group(2)),
            current_arm_total_windows=int(last.group(3)), execution=last.group(1),
            latest_logged_partial_ppl=float(last.group(4)))
    return progress


def snapshot(job_path, *, proc_root=Path('/proc')):
    job, error = read_json(job_path)
    result = {'observed_at_unix': time.time(), 'job_file': str(job_path)}
    if job is None:
        return {**result, 'status': 'unreadable' if error else 'not_started', 'error': error}
    if job.get('format') != JOB_FORMAT:
        return {**result, 'status': 'unreadable', 'error': 'Unexpected job format'}
    outputs = job.get('outputs', {})
    runner = verified_process(job.get('runner_identity', {}), proc_root=proc_root)
    stages = [{**{key: row.get(key) for key in ('stage', 'pid', 'exit_code', 'started_at_unix', 'ended_at_unix')},
        'process': verified_process(row.get('process_identity', {}), proc_root=proc_root)}
        for row in job.get('stages', [])]
    live_child = any(row['process']['matching_live_process'] for row in stages)
    result.update(status=('complete' if job.get('complete') is True else 'failed' if job.get('error')
        else 'process_verification_unavailable' if runner.get('reason') == 'proc_unavailable'
        else 'running' if runner['matching_live_process'] or live_child else 'not_running'),
        job_complete=job.get('complete', False), stage=job.get('stage'),
        job_error=job.get('error'), runner=runner, stages=stages)
    training, training_error = read_json(outputs.get('training_report', '/nonexistent'))
    if training is not None:
        history = training.get('history', [])
        last = history[-1] if history else {}
        result['training'] = {'complete': training.get('complete', False),
            'error': training.get('error'), 'successful_updates': training.get('successful_updates', 0),
            'fixed_successful_updates': 1024, 'attempts': training.get('attempts', 0),
            'overflows': training.get('overflow_retries', 0),
            'latest_attempt': {key: last.get(key) for key in ('attempt', 'successful_updates_after',
                'loss', 'ce', 'teacher_to_student_kl', 'overflow', 'loss_scale_after', 'attempt_seconds')}}
    elif training_error:
        result['training_read_error'] = training_error
    evaluation, evaluation_error = read_json(outputs.get('evaluation_report', '/nonexistent'))
    log = tail_text(outputs.get('evaluation_log', '/nonexistent'))
    if evaluation is not None or log:
        result['evaluation'] = evaluation_progress(evaluation, log)
    if evaluation_error:
        result['evaluation_read_error'] = evaluation_error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=ROOT / 'reports/low_rank_compensation_v1_job.json')
    parser.add_argument('--milestone', type=int, help='Return early when this completed-update count is reached')
    parser.add_argument('--wait-seconds', type=float, default=0, help='Bounded wait,0..60 seconds; never restarts a job')
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 60 or (args.milestone is not None and not 1 <= args.milestone <= 1024):
        parser.error('wait-seconds must be0..60 and milestone1..1024')
    deadline = time.monotonic() + args.wait_seconds
    while True:
        result = snapshot(args.job)
        reached = (args.milestone is not None
            and result.get('training', {}).get('successful_updates', 0) >= args.milestone)
        terminal = result.get('status') in ('complete', 'failed', 'not_running', 'unreadable')
        if reached or terminal or time.monotonic() >= deadline:
            result['milestone_reached'] = reached if args.milestone is not None else None
            print(json.dumps(result, indent=2, allow_nan=False))
            return
        time.sleep(min(5, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
