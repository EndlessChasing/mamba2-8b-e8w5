#!/usr/bin/env python3
"""Read-only CPU accounting audit of the first periodic prototype checkpoint.

No trainer imports, model loading, checkpoint selection, export, or PPL scoring.
The exact single-read live report and completed journal prefix are retained.
Run only after checkpoint32 is listed in the trainer's atomic report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SHA = '05f41c5607b90da8b9cc68924c854655c12fe8ce1b76450c6ca5974dead85a6f'
LABELS = {f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            result.update(data)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def same_json(left, right):
    # JSON preserves bool/int distinctions which ordinary dict equality loses.
    return json_bytes(left) == json_bytes(right)


def regular(path):
    require(not path.is_symlink() and path.is_file(), f'Expected regular file: {path}')
    return path


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {'file': str(path), 'bytes': len(data), 'sha256': digest(data)}


def schedule():
    rng = torch.Generator(device='cpu').manual_seed(20260927)
    return [{'epoch': epoch, 'position_in_epoch': position, 'window': window}
            for epoch in range(4)
            for position, window in enumerate(torch.randperm(32, generator=rng).tolist())]


def history_counts(history, plan):
    require(isinstance(history, list) and len(history) <= 136, 'Invalid history length')
    successes = overflows = 0
    for attempt, row in enumerate(history, 1):
        require(type(row['attempt']) is int and row['attempt'] == attempt, 'Attempt sequence differs')
        require(type(row['successful_updates_after']) is int, 'Success counter is not an integer')
        require(type(row['overflow']) is bool, 'Overflow flag is not a boolean')
        require(type(row['targets']) is int and row['targets'] == 2047, 'Wrong target count')
        require(successes < 128 and same_json(row['schedule_entry'], plan[successes]), 'Schedule/retry differs')
        overflows += int(row['overflow'])
        successes += int(not row['overflow'])
        require(row['successful_updates_after'] == successes, 'Success sequence differs')
    require(overflows <= 8, 'Overflow budget exceeded')
    return {'successful_updates': successes, 'attempts': len(history), 'overflow_retries': overflows,
            'successful_target_exposures': successes*2047,
            'attempted_target_exposures': len(history)*2047}


def journal_prefix(data, history, pid):
    """Consume exactly N start/complete pairs; later/inflight bytes are ignored."""
    lines = data.splitlines(keepends=True)
    require(len(lines) >= 2*len(history), 'Journal has not committed the checkpoint prefix')
    prefix = lines[:2*len(history)]
    successes = 0
    for index, row in enumerate(history):
        pair = prefix[2*index:2*index+2]
        require(all(line.endswith(b'\n') for line in pair), 'Incomplete journal prefix line')
        start, complete = [json.loads(line) for line in pair]
        for event in (start, complete):
            require(type(event['pid']) is int and event['pid'] == pid, 'Journal PID differs')
            require(type(event['time_unix']) in (int, float) and math.isfinite(event['time_unix']), 'Invalid journal time')
        trim = lambda event: {k:v for k,v in event.items() if k not in ('time_unix', 'pid')}
        wanted = {'event':'attempt_start', 'attempt':index+1,
                  'successful_updates_before':successes, 'schedule_entry':row['schedule_entry']}
        require(same_json(trim(start), wanted), 'Journal attempt_start differs')
        require(same_json(trim(complete), {'event':'attempt_complete', **row}), 'Journal attempt_complete differs')
        successes += int(not row['overflow'])
    return b''.join(prefix)


def audit(args, output, snapshot, journal_snapshot):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    # The trainer uses atomic replace; this one open observes one complete version.
    live_bytes = regular(args.train_report).read_bytes()
    output['live_report_snapshot'] = write_new(snapshot, live_bytes)
    live = json.loads(live_bytes)
    require(live.get('format') == 'MAMBA2_PROTOTYPE_COMPENSATION_TRAIN_V1'
            and live.get('mode') == 'train' and live.get('evaluation_data_used') is False
            and live.get('split') == 'train', 'Unexpected training report')
    smoke_bytes = regular(args.smoke_report).read_bytes()
    require(digest(smoke_bytes) == SMOKE_SHA, 'Passed smoke file differs from fixed experiment')
    smoke = json.loads(smoke_bytes)
    require(smoke.get('complete') is True and smoke.get('passed') is True
            and smoke.get('mode') == 'smoke' and smoke.get('trial_state_discarded') is True,
            'Expected passed and discarded smoke')
    require(same_json(live['binding'], smoke['binding']) and live['smoke_report_sha256'] == SMOKE_SHA,
            'Live binding/smoke identity differs')
    selected = [entry for entry in live.get('checkpoints', [])
                if re.fullmatch(r'checkpoint_attempt\d{3}_step032_periodic\.pt', Path(entry['file']).name)]
    require(len(selected) == 1, 'Exactly one completed checkpoint32 receipt must already be listed')
    receipt = selected[0]
    path = Path(receipt['file'])
    require(path.is_absolute() and path.parent.resolve() == args.work_dir, 'Checkpoint is outside requested work directory')
    regular(path)
    require(type(receipt['bytes']) is int and path.stat().st_size == receipt['bytes']
            and sha(path) == receipt['sha256'], 'Checkpoint file differs from report receipt')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    require(sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes'], 'Checkpoint changed while being read')
    require(checkpoint.get('format') == 'MAMBA2_PROTOTYPE_TRAINING_CHECKPOINT_V1'
            and checkpoint.get('reason') == 'periodic' and checkpoint.get('resumable') is False,
            'Not the nonresumable periodic checkpoint')
    require(same_json(checkpoint['binding'], smoke['binding'])
            and checkpoint['smoke_report_sha256'] == SMOKE_SHA, 'Checkpoint binding differs')
    for name in ('parent_manifest_sha256', 'small_manifest_sha256'):
        require(checkpoint[name] == checkpoint['binding'][name], f'Checkpoint alias differs: {name}')
    for section in ('frozen_code_sha256', 'code_sha256'):
        for name, expected in checkpoint['binding'][section].items():
            require(sha(regular(ROOT/name)) == expected, f'Current bound code differs: {name}')
    plan = schedule()
    require(same_json(checkpoint['schedule'], plan) and same_json(live['schedule'], plan), 'Fixed schedule differs')
    counts = history_counts(checkpoint['history'], plan)
    require(counts['successful_updates'] == 32, 'Checkpoint is not step32')
    for name in ('successful_updates', 'attempts', 'overflow_retries'):
        require(type(checkpoint[name]) is int and checkpoint[name] == counts[name], f'Counter differs: {name}')
    require(path.name == f"checkpoint_attempt{counts['attempts']:03d}_step032_periodic.pt", 'Checkpoint filename/accounting differs')
    live_counts = history_counts(live['history'], plan)
    for name, value in live_counts.items():
        require(type(live[name]) is int and live[name] == value, f'Live report counter differs: {name}')
    require(same_json(live['history'][:counts['attempts']], checkpoint['history']), 'Live history prefix differs')
    journal = regular(args.work_dir/'attempts.jsonl')
    prefix = journal_prefix(journal.read_bytes(), checkpoint['history'], live['pid'])
    output['journal_prefix_snapshot'] = write_new(journal_snapshot, prefix)
    ledger = checkpoint['frozen_model_sha256']
    require(isinstance(ledger, dict) and len(ledger) == 507
            and ledger == live['baseline_gpu_sha256'] == smoke['baseline_gpu_sha256']
            == smoke['frozen_parameter_audit']['after_sha256'], 'Startup frozen ledger differs')
    masters = checkpoint['masters']
    require(isinstance(masters, dict) and set(masters) == LABELS, 'Master table inventory differs')
    table_records = {}
    for name, value in masters.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.float32 and tuple(value.shape) == (256,8)
                and bool(torch.isfinite(value).all()) and bool(torch.isfinite(value.half()).all()),
                f'Invalid master or FP16 overflow: {name}')
        table_records[name] = {'fp32_sha256':digest(value.contiguous().numpy().tobytes()),
            'rounded_fp16_sha256':digest(value.half().contiguous().numpy().tobytes()),
            'nonzero_fp32':int(torch.count_nonzero(value)), 'nonzero_fp16':int(torch.count_nonzero(value.half()))}
    optimizer = checkpoint['optimizer']
    groups, states = optimizer['param_groups'], optimizer['state']
    require(len(groups) == 1, 'Unexpected optimizer group count')
    ids = groups[0]['params']
    require(len(ids) == 112 and all(type(i) is int for i in ids)
            and len(set(ids)) == 112 and set(states) == set(ids), 'Optimizer coverage differs')
    for name, expected in {'lr':.001, 'betas':(.9,.999), 'eps':1e-8, 'weight_decay':0., 'amsgrad':False}.items():
        require(groups[0][name] == expected, f'Optimizer recipe differs: {name}')
    for index in ids:
        state = states[index]
        require(set(state) == {'step','exp_avg','exp_avg_sq'}, 'Unexpected Adam state')
        step = state['step']
        require(isinstance(step, torch.Tensor) and step.device.type == 'cpu'
                and step.numel() == 1 and float(step) == 32., 'Optimizer step differs')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = state[name]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    and value.dtype == torch.float32 and tuple(value.shape) == (256,8)
                    and bool(torch.isfinite(value).all()), 'Invalid Adam moment')
        require(bool((state['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
    scale = checkpoint['scaler']
    require(scale['scale'] == checkpoint['history'][-1]['loss_scale_after']
            and scale['scale'] == 1024.*(.5**counts['overflow_retries'])
            and scale['growth_factor'] == 2. and scale['backoff_factor'] == .5
            and scale['growth_interval'] == 2000, 'Loss scaler recipe/state differs')
    if counts['overflow_retries'] == 0:
        require(type(scale['_growth_tracker']) is int and scale['_growth_tracker'] == 32, 'Loss scaler success count differs')
    require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    output.update(complete=True, checkpoint=receipt, counters=counts,
        live_snapshot_counters=live_counts, binding=checkpoint['binding'], smoke_report_sha256=SMOKE_SHA,
        schedule_sha256=digest(json_bytes(plan)), schedule_and_history_prefix_verified=True,
        journal_pairs_verified=counts['attempts'], master_tables=112, master_parameters=229376,
        master_values_and_fp16_casts_finite=True, masters=table_records,
        optimizer_state_tensors=112, optimizer_step=32, scaler=scale,
        startup_frozen_ledger_tensors=507, startup_frozen_ledger_sha256=digest(json_bytes(ledger)),
        current_bound_source_hashes_verified=True, cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-report', 'smoke-report', 'work-dir', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('train_report', 'smoke_report', 'work_dir', 'report'):
        path = getattr(args, name)
        require(not path.is_symlink(), f'Symlink argument rejected: {name}')
        setattr(args, name, path.resolve())
    require(args.work_dir.is_dir(), 'Training work directory is absent')
    snapshot = args.report.with_name(args.report.stem+'.train_snapshot.json')
    journal_snapshot = args.report.with_name(args.report.stem+'.journal_prefix.jsonl')
    for path in (args.report, snapshot, journal_snapshot):
        require(not path.exists() and not path.is_symlink(), f'Preserving existing output: {path}')
        require(path not in (args.train_report, args.smoke_report)
                and path != args.work_dir and args.work_dir not in path.parents, 'Audit output overlaps input')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output = {'format':'MAMBA2_PROTOTYPE_CHECKPOINT32_CPU_AUDIT_V1', 'complete':False,
        'started_at_unix':time.time(), 'script_sha256':sha(__file__),
        'train_report':str(args.train_report), 'smoke_report':str(args.smoke_report),
        'work_dir':str(args.work_dir), 'gpu_used':False, 'intermediate_PPL_performed':False,
        'intermediate_export_performed':False, 'fresh_GPU_frozen_hash_performed':False,
        'resume_performed':False, 'checkpoint_selection_by_quality':False,
        'scope':'First periodic checkpoint accounting and CPU tensor integrity only; frozen hashes are the startup ledger, not a fresh step32 GPU content audit.'}
    try:
        audit(args, output, snapshot, journal_snapshot)
    except BaseException as error:
        output.update(error_type=type(error).__name__, error=str(error), cuda_initialized=torch.cuda.is_initialized())
        raise
    finally:
        output['ended_at_unix'] = time.time()
        write_new(args.report, json_bytes(output))
    print(json.dumps({'complete':True, 'report':str(args.report), 'sha256':sha(args.report)}), flush=True)


if __name__ == '__main__':
    main()
