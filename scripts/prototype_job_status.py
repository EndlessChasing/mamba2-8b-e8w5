#!/usr/bin/env python3
"""Read job receipts and verify actual Linux processes; never start/restart work."""
import argparse
import json
from pathlib import Path
import statistics
import time


def process(pid, expected_script):
    directory = Path('/proc')/str(pid)
    try:
        command = (directory/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        status = (directory/'status').read_text()
        state = next(line for line in status.splitlines() if line.startswith('State:'))
    except FileNotFoundError:
        return {'pid': pid, 'present': False, 'matching_live_process': False}
    matches = expected_script in command
    return {'pid': pid, 'present': True, 'command_matches': matches, 'state': state,
            'matching_live_process': matches and not any(x in state for x in ('Z (', 'X (')),
            'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=Path('reports/prototype_compensation_v1_job.json'))
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    training = json.loads(Path(job['outputs']['training_report']).read_text())
    successes = [row for row in training.get('history', []) if row.get('overflow') is False]
    seconds = [row['attempt_seconds'] for row in successes]
    result = {'observed_at_unix': time.time(), 'job_complete': job['complete'],
        'stage': job.get('stage'), 'job_error': job.get('error'),
        'runner': process(job['pid'], 'scripts/run_prototype_compensation_job.py'),
        'stages': [{**{key: row.get(key) for key in ('stage', 'exit_code', 'started_at_unix', 'ended_at_unix')},
                    'process': process(row['pid'], row['command'][2])} for row in job['stages']],
        'training': {'complete': training['complete'], 'error': training.get('error'),
            'successful_updates': training.get('successful_updates', 0),
            'attempts': training.get('attempts', 0), 'overflows': training.get('overflow_retries', 0),
            'zero_decodes_verified': training.get('zero_decoded_projection_audit', {}).get('verified'),
            'last_success_seconds': seconds[-1] if seconds else None,
            'median_success_seconds': statistics.median(seconds) if seconds else None,
            'latest_success': {key: successes[-1].get(key) for key in
                ('attempt', 'successful_updates_after', 'ce', 'teacher_to_student_kl', 'loss',
                 'loss_scale_after', 'gradient_norm_before_clip')} if successes else None}}
    evaluation_path = Path(job['outputs']['evaluation_report'])
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        result['evaluation'] = {'complete': evaluation['complete'], 'comparison': evaluation.get('comparison')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
