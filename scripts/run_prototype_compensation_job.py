#!/usr/bin/env python3
"""Run the declared fixed training, then independent validation, without publishing."""
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
PINS = {
    'scripts/train_prototype_compensation.py': 'bc6f59da1671d0e8973984f8ce24d176c364a128086f17b57b34c1f4022e1bc0',
    'scripts/evaluate_prototype_compensation.py': '59a10619c16721056dc8afe128f3c44eef5d05015488e10b7f063b4cd3394ebd',
    'reports/prototype_compensation_v1_smoke.json': '05f41c5607b90da8b9cc68924c854655c12fe8ce1b76450c6ca5974dead85a6f',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    outputs = {
        'work': ROOT/'artifacts/prototype_compensation_v1_train_work',
        'overlay': ROOT/'artifacts/prototype_compensation_v1',
        'training_report': ROOT/'reports/prototype_compensation_v1_train.json',
        'evaluation_report': ROOT/'reports/prototype_compensation_v1_eval.json',
        'training_log': ROOT/'reports/prototype_compensation_v1_train.log',
        'evaluation_log': ROOT/'reports/prototype_compensation_v1_eval.log',
    }
    for path in [report_path, *outputs.values()]:
        if path.exists():
            raise FileExistsError(path)
    for name, digest in PINS.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Bound executable/smoke differs: {name}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the job receipt before any subprocess can start.
    with report_path.open('x') as stream:
        stream.write('{}\n')
    report = {'format': 'MAMBA2_PROTOTYPE_JOB_V1', 'complete': False,
              'pid': os.getpid(), 'started_at_unix': time.time(),
              'script_sha256': sha(__file__), 'bound_inputs': PINS,
              'outputs': {key: str(path) for key, path in outputs.items()},
              'stages': [], 'publication_performed': False,
              'fixed_successful_updates': 128,
              'scope': 'Fixed TRAIN recipe followed by independent full validation; no MK or test scoring.'}

    def save():
        temporary = report_path.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(report_path)

    def run(stage, command, log):
        entry = {'stage': stage, 'command': command, 'started_at_unix': time.time()}
        report['stages'].append(entry)
        report['stage'] = stage
        save()
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            save()
            entry['exit_code'] = process.wait()
        entry['ended_at_unix'] = time.time()
        save()
        if entry['exit_code']:
            raise RuntimeError(f'{stage} failed with exit code {entry["exit_code"]}; inspect {log}')

    save()
    try:
        run('training', [sys.executable, '-u', str(ROOT/'scripts/train_prototype_compensation.py'),
            '--mode', 'train', '--protocol-sha256',
            'e50295425dcd2f5a09b5fea4254b9c6f6f38744bc73583f93621ac1ce942cb9d',
            '--work-dir', str(outputs['work']), '--out-dir', str(outputs['overlay']),
            '--report', str(outputs['training_report']),
            '--smoke-report', str(ROOT/'reports/prototype_compensation_v1_smoke.json')],
            outputs['training_log'])
        train = json.loads(outputs['training_report'].read_text())
        if train.get('complete') is not True or train.get('successful_updates') != 128:
            raise RuntimeError('Training did not report the declared 128 completed updates')
        checkpoint = Path(train['final_checkpoint']['file'])
        if sha(checkpoint) != train['final_checkpoint']['sha256']:
            raise RuntimeError('Final training checkpoint differs from receipt')
        for name, digest in PINS.items():
            if sha(ROOT/name) != digest:
                raise RuntimeError(f'Bound executable/smoke changed: {name}')
        run('independent_full_validation', [sys.executable, '-u',
            str(ROOT/'scripts/evaluate_prototype_compensation.py'),
            '--overlay-dir', str(outputs['overlay']),
            '--training-report', str(outputs['training_report']),
            '--training-checkpoint', str(checkpoint),
            '--smoke-report', str(ROOT/'reports/prototype_compensation_v1_smoke.json'),
            '--report', str(outputs['evaluation_report'])], outputs['evaluation_log'])
        evaluated = json.loads(outputs['evaluation_report'].read_text())
        if evaluated.get('complete') is not True:
            raise RuntimeError('Independent evaluation is incomplete')
        report.update(complete=True, stage='complete', comparison=evaluated['comparison'],
                      evaluation_report_sha256=sha(outputs['evaluation_report']))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['ended_at_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
