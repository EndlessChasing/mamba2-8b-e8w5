#!/usr/bin/env python3
"""Supervise exactly one selected teacher-KL stage; never resume or publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_low_rank_job import read_process

CHECKER_SHA = '46cca2177963e8cb673fb404860c4cf29f5823b98263d6b62d5d61947d17a7cb'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_completion(stage, result):
    if result.get('complete') is not True:
        raise RuntimeError('Child exited without a completed receipt')
    if stage == 'smoke':
        if result.get('passed') is not True or result.get('trial_state_discarded') is not True:
            raise RuntimeError('Smoke did not pass and discard its trial state')
    elif stage == 'train':
        if (result.get('successful_updates') != 448 or result.get('final_step') != 448
                or result.get('successful_target_exposures') != 917056
                or type(result.get('overflow_retries')) is not int
                or not 0 <= result['overflow_retries'] <= 8
                or result.get('attempts') != 448 + result['overflow_retries']):
            raise RuntimeError('Fixed one-pass accounting differs')
    else:
        gate = result.get('reserved_gate', {})
        full = result.get('full_validation', {})
        if any(type(gate.get(key)) is not bool for key in ('ppl_met', 'kl_met', 'passed')):
            raise RuntimeError('Missing reserved-set decision')
        if gate['passed'] != (gate['ppl_met'] and gate['kl_met']):
            raise RuntimeError('Inconsistent reserved-set decision')
        if full.get('performed') is not gate['passed']:
            raise RuntimeError('Full validation must follow the reserved-set gate exactly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'train', 'evaluate'), required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    protocol = ROOT / 'docs/TEACHER_KL_COMPENSATION_PROTOCOL.md'
    if sha(protocol) != args.protocol_sha256 or sha(ROOT / 'scripts/check_low_rank_job.py') != CHECKER_SHA:
        raise ValueError('Protocol or process-identity primitive changed')
    name = 'teacher_kl_compensation_v1'
    suffix = 'eval' if args.stage == 'evaluate' else args.stage
    report = ROOT / f'reports/{name}_{suffix}.json'
    journal = ROOT / f'reports/{name}_{suffix}_process.json'
    log = ROOT / f'reports/{name}_{suffix}.log'
    for path in (report, journal, log, journal.with_suffix('.json.partial')):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'No overwrite or automatic retry: {path}')
    if args.stage == 'evaluate':
        script = ROOT / 'scripts/evaluate_teacher_kl_compensation.py'
        train = ROOT / f'reports/{name}_train.json'
        trained = json.loads(train.read_text())
        validate_completion('train', trained)
        checkpoint = Path(trained['final_checkpoint']['file']).resolve()
        if ROOT / f'artifacts/{name}_train_work' not in checkpoint.parents:
            raise ValueError('Checkpoint is outside the fixed training work directory')
        if sha(checkpoint) != trained['final_checkpoint']['sha256']:
            raise ValueError('Final checkpoint differs')
        command = [sys.executable, '-u', str(script), '--overlay-dir', str(ROOT / f'artifacts/{name}'),
                   '--training-report', str(train), '--training-checkpoint', str(checkpoint),
                   '--smoke-report', str(ROOT / f'reports/{name}_smoke.json'), '--report', str(report)]
    else:
        script = ROOT / 'scripts/train_teacher_kl_compensation.py'
        output = name if args.stage == 'train' else name + '_smoke_unused'
        command = [sys.executable, '-u', str(script), '--mode', args.stage,
                   '--work-dir', str(ROOT / f'artifacts/{name}_{args.stage}_work'),
                   '--out-dir', str(ROOT / f'artifacts/{output}'), '--report', str(report),
                   '--protocol-sha256', args.protocol_sha256]
        if args.stage == 'train':
            command += ['--smoke-report', str(ROOT / f'reports/{name}_smoke.json')]
    identity = read_process(os.getpid())
    if not identity.get('live'):
        raise RuntimeError('Linux process identity unavailable')
    value = {'format': 'MAMBA2_TEACHER_KL_STAGE_PROCESS_V1', 'complete': False,
             'stage': args.stage, 'started_at_unix': time.time(), 'runner_identity': identity,
             'command': command, 'script_sha256': sha(script), 'supervisor_sha256': sha(__file__),
             'protocol_sha256': args.protocol_sha256, 'publication_performed': False,
             'automatic_next_stage': False, 'log': str(log), 'child_report': str(report)}
    with journal.open('x') as stream:
        json.dump(value, stream)

    def save():
        temporary = journal.with_suffix('.json.partial')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(journal)

    def stop(signum, _frame):
        raise InterruptedError(f'Stop requested by signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process = None
    try:
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            value['child_identity'] = read_process(process.pid)
            save()
            print(json.dumps({'stage': args.stage, 'child_identity': value['child_identity']}), flush=True)
            value['exit_code'] = process.wait()
        if value['exit_code'] != 0:
            raise RuntimeError(f'Child exited {value["exit_code"]}; preserve failed receipt')
        result = json.loads(report.read_text())
        validate_completion(args.stage, result)
        if sha(script) != value['script_sha256'] or sha(protocol) != args.protocol_sha256:
            raise RuntimeError('Stage executable or protocol changed during execution')
        value.update(complete=True, child_report_sha256=sha(report),
                     reserved_gate=result.get('reserved_gate'), comparison=result.get('comparison'))
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        value.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                     exit_code=process.returncode if process is not None else None)
        raise
    finally:
        value['ended_at_unix'] = time.time()
        save()
    print(json.dumps({'complete': value['complete'], 'stage': args.stage,
                      'journal': str(journal), 'child_report_sha256': value['child_report_sha256']}), flush=True)


if __name__ == '__main__':
    main()
