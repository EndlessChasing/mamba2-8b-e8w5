#!/usr/bin/env python3
"""Fixed1024 low-rank training then independent validation; no resume/publication.

Final trainer/evaluator/passed-smoke hashes are pinned. --check-only creates
no files and launches no process. Actual launch remains root-coordinated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_low_rank_job import JOB_FORMAT, read_process

PROTOCOL_SHA = 'b732c24692ae5dc91c5c67c1a0e0a0c435be3402959bfe5db874c67246836cf2'
PINS = {
    'scripts/train_low_rank_compensation.py': 'fdcb974e7abcd9379a6da900cd5801839e03973275010185f59235d67d938273',
    'scripts/evaluate_low_rank_compensation.py': 'f0d1285276d1b4ea3bb132b98cb412a9718289b715d70ee1fa961c08bc3b0bd7',
    'reports/low_rank_compensation_v1_smoke.json': 'dacedf6401f082d172da31a5737c11d038998c939988999099f89b9620de0132',
    'docs/LOW_RANK_COMPENSATION_PROTOCOL.md': PROTOCOL_SHA,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def outputs_for(root):
    return {'work': root / 'artifacts/low_rank_compensation_v1_train_work',
        'overlay': root / 'artifacts/low_rank_compensation_v1',
        'training_report': root / 'reports/low_rank_compensation_v1_train.json',
        'evaluation_report': root / 'reports/low_rank_compensation_v1_eval.json',
        'training_log': root / 'reports/low_rank_compensation_v1_train.log',
        'evaluation_log': root / 'reports/low_rank_compensation_v1_eval.log'}


def verify_pins(root=ROOT, pins=None):
    pins = PINS if pins is None else pins
    # Validate every pin first, before opening inputs or reserving any output.
    if any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in pins.values()):
        raise ValueError('Launch blocked: replace every PENDING final-code/passed-smoke SHA256 pin')
    for name, digest in pins.items():
        if sha(root / name) != digest:
            raise ValueError(f'Pinned executable/protocol/smoke differs: {name}')


def check_smoke(root=ROOT):
    smoke = json.loads((root / 'reports/low_rank_compensation_v1_smoke.json').read_text())
    binding = smoke.get('binding', {})
    if (smoke.get('complete') is not True or smoke.get('passed') is not True
            or smoke.get('mode') != 'smoke' or smoke.get('trial_state_discarded') is not True
            or binding.get('protocol_sha256') != PROTOCOL_SHA
            or binding.get('code_sha256', {}).get('scripts/train_low_rank_compensation.py')
                != PINS['scripts/train_low_rank_compensation.py']):
        raise ValueError('Pinned receipt is not a passed smoke for the final trainer/protocol')


def final_checkpoint(training, work):
    if (training.get('complete') is not True or training.get('mode') != 'train'
            or type(training.get('successful_updates')) is not int or training['successful_updates'] != 1024
            or type(training.get('final_step')) is not int or training['final_step'] != 1024
            or type(training.get('overflow_retries')) is not int or not 0 <= training['overflow_retries'] <= 8
            or type(training.get('attempts')) is not int or training['attempts'] != 1024 + training['overflow_retries']
            or type(training.get('successful_target_exposures')) is not int
            or training['successful_target_exposures'] != 2096128):
        raise RuntimeError('Training did not complete the fixed1024-update recipe')
    receipt = training['final_checkpoint']
    checkpoint = Path(receipt['file']).resolve()
    if work.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise RuntimeError('Final checkpoint must exist inside the fixed training work directory')
    if sha(checkpoint) != receipt['sha256']:
        raise RuntimeError('Final checkpoint differs from its training receipt')
    return checkpoint


def check_evaluation(evaluation):
    if (evaluation.get('complete') is not True or evaluation.get('mode') != 'three_arm_full_validation'
            or evaluation.get('protocol', {}).get('target_tokens') != 264764
            or evaluation.get('protocol', {}).get('windows') != 130
            or set(evaluation.get('results', {})) != {'source_fp16','best_small_e8w5','low_rank_e8w5'}):
        raise RuntimeError('Independent three-arm full validation is incomplete')
    for value in evaluation['results'].values():
        if (value.get('target_tokens') != 264764 or len(value.get('windows', [])) != 130
                or not math.isfinite(value.get('ppl', float('nan')))):
            raise RuntimeError('Independent validation arm is incomplete/nonfinite')
    if not isinstance(evaluation.get('comparison'), dict):
        raise RuntimeError('Independent validation comparison is missing')


class RequestedStop(RuntimeError):
    pass


def stop_requested(signum, _frame):
    raise RequestedStop(f'Job stop requested by signal{signum}; no resume/replay')


def run_stage(stage, command, log, report, save, *, root=ROOT):
    """Sequential child with durable exit identity; failures never advance a stage."""
    verify_pins(root)
    entry = {'stage': stage, 'command': command, 'started_at_unix': time.time()}
    report['stages'].append(entry)
    report['stage'] = stage
    save()
    process = None
    try:
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            entry.update(pid=process.pid, process_identity=read_process(process.pid))
            save()
            entry['exit_code'] = process.wait()
    except BaseException:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    entry['forced_kill_after_termination_timeout'] = True
            entry['exit_code'] = process.returncode
        entry['interrupted_or_launch_failed'] = True
        raise
    finally:
        entry['ended_at_unix'] = time.time()
        save()
    if entry['exit_code'] != 0:
        raise RuntimeError(f'{stage} exited{entry["exit_code"]}; preserve and inspect {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate readiness without creating files or launching processes')
    args = parser.parse_args()
    outputs = outputs_for(ROOT)
    report_path = ROOT / 'reports/low_rank_compensation_v1_job.json'
    for path in (report_path, report_path.with_suffix('.json.partial'), *outputs.values()):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'No resume/overwrite: {path}')
    verify_pins()
    check_smoke()
    runner = read_process(os.getpid())
    if not runner.get('present') or not runner.get('live') or not runner.get('argv'):
        raise RuntimeError('Linux /proc identity is required before launching this job')
    if args.check_only:
        print(json.dumps({'ready': True, 'launch_performed': False, 'pins': PINS}))
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x') as stream:
        stream.write('{}\n')
    report = {'format': JOB_FORMAT, 'complete': False, 'pid': os.getpid(),
        'runner_identity': runner, 'started_at_unix': time.time(),
        'script_sha256': sha(__file__), 'checker_sha256': sha(ROOT / 'scripts/check_low_rank_job.py'),
        'bound_inputs': dict(PINS), 'outputs': {key: str(value) for key, value in outputs.items()},
        'fixed_successful_updates': 1024, 'stages': [], 'stage': 'starting',
        'publication_performed': False, 'resume': False,
        'scope': 'Fixed TRAIN recipe followed by independent full validation; no MK/test/publishing.'}

    def save():
        temporary = report_path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(report_path)

    save()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    try:
        run_stage('training', [sys.executable, '-u', str(ROOT / 'scripts/train_low_rank_compensation.py'),
            '--mode', 'train', '--protocol-sha256', PROTOCOL_SHA,
            '--work-dir', str(outputs['work']), '--out-dir', str(outputs['overlay']),
            '--report', str(outputs['training_report']),
            '--smoke-report', str(ROOT / 'reports/low_rank_compensation_v1_smoke.json')], outputs['training_log'], report, save)
        training = json.loads(outputs['training_report'].read_text())
        checkpoint = final_checkpoint(training, outputs['work'])
        report.update(training_report_sha256=sha(outputs['training_report']),
            final_checkpoint={'file': str(checkpoint), 'sha256': sha(checkpoint)})
        run_stage('independent_full_validation', [sys.executable, '-u', str(ROOT / 'scripts/evaluate_low_rank_compensation.py'),
            '--overlay-dir', str(outputs['overlay']), '--training-report', str(outputs['training_report']),
            '--training-checkpoint', str(checkpoint),
            '--smoke-report', str(ROOT / 'reports/low_rank_compensation_v1_smoke.json'),
            '--report', str(outputs['evaluation_report'])], outputs['evaluation_log'], report, save)
        evaluation = json.loads(outputs['evaluation_report'].read_text())
        check_evaluation(evaluation)
        verify_pins()
        report.update(complete=True, stage='complete', comparison=evaluation['comparison'],
            evaluation_report_sha256=sha(outputs['evaluation_report']))
    except BaseException as error:
        report.update(stage='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['ended_at_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
