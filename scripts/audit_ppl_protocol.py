#!/usr/bin/env python3
"""Recompute paired full-test window statistics from saved reports; stdlib only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(rows):
    targets = sum(row['target_tokens'] for row in rows)
    baseline = sum(row['baseline_nll'] for row in rows)
    candidate = sum(row['candidate_nll'] for row in rows)
    return {'windows': len(rows), 'target_tokens': targets,
            'baseline_ppl': math.exp(baseline / targets),
            'candidate_ppl': math.exp(candidate / targets),
            'ppl_relative_percent': 100 * math.expm1((candidate - baseline) / targets),
            'excess_nll_per_token': (candidate - baseline) / targets}


def correlation(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def audit(baseline, candidate):
    identity = ('suite', 'execution', 'source_checkpoint_sha256', 'tokenizer_sha256',
                'runtime_source_sha256', 'evaluation_source_sha256', 'environment', 'dataset')
    if not baseline.get('complete') or not candidate.get('complete'):
        raise ValueError('Both reports must be complete')
    for field in identity:
        if baseline[field] != candidate[field]:
            raise ValueError(f'Paired identity differs: {field}')
    if baseline['suite'] != 'full':
        raise ValueError('This audit requires the full-test protocol')
    left, right = baseline['ppl'], candidate['ppl']
    for field in ('target_tokens', 'execution', 'cache_dtype', 'coverage', 'logits_chunk_tokens'):
        if left[field] != right[field]:
            raise ValueError(f'PPL protocol differs: {field}')
    if left['coverage'] != 'full split' or len(left['windows']) != len(right['windows']):
        raise ValueError('Reports need the same complete split coverage')
    rows = []
    for index, (a, b) in enumerate(zip(left['windows'], right['windows'])):
        for field in ('start', 'input_tokens', 'target_tokens', 'token_sha256_int64le'):
            if a[field] != b[field]:
                raise ValueError(f'Window {index} differs: {field}')
        if a['start'] != index * 2048 or not 0 < a['target_tokens'] <= 2048:
            raise ValueError('Unexpected full-test window geometry')
        if index < len(left['windows']) - 1 and a['target_tokens'] != 2048:
            raise ValueError('Only the final full-test window may be shorter')
        for window in (a, b):
            if not math.isfinite(window['nll']) or not math.isclose(
                    math.exp(window['nll'] / window['target_tokens']),
                    window['ppl'], rel_tol=1e-12):
                raise ValueError(f'Window {index} NLL/PPL mismatch')
        rows.append({'index': index, 'start': a['start'], 'target_tokens': a['target_tokens'],
                     'baseline_nll': a['nll'], 'candidate_nll': b['nll'],
                     'baseline_ppl': a['ppl'], 'candidate_ppl': b['ppl'],
                     'excess_nll': b['nll'] - a['nll'],
                     'excess_nll_per_token': (b['nll'] - a['nll']) / a['target_tokens'],
                     'relative_percent': 100 * (b['ppl'] / a['ppl'] - 1)})
    if len(rows) <= 5:
        raise ValueError('Need more than five windows for this diagnostic')
    for report in (baseline, candidate):
        windows = report['ppl']['windows']
        targets = sum(row['target_tokens'] for row in windows)
        nll = sum(row['nll'] for row in windows)
        if targets != report['dataset']['total_tokens'] - 1 or targets != report['ppl']['target_tokens']:
            raise ValueError('Full coverage target count mismatch')
        if not math.isclose(nll, report['ppl']['nll'], rel_tol=1e-12) or not math.isclose(
                math.exp(nll / targets), report['ppl']['ppl'], rel_tol=1e-12):
            raise ValueError('Aggregate NLL/PPL mismatch')
    excess = [row['excess_nll_per_token'] for row in rows]
    ratios = [row['relative_percent'] for row in rows]
    worst = sorted(rows, key=lambda row: row['excess_nll'])
    total_excess = sum(row['excess_nll'] for row in rows)
    return {'complete': True, 'identity_fields_equal': list(identity), 'aggregate': aggregate(rows),
            'worsened_windows': sum(value > 0 for value in excess),
            'improved_windows': sum(value < 0 for value in excess),
            'unchanged_windows': sum(value == 0 for value in excess),
            'relative_percent_percentiles': {str(p): percentile(ratios, p)
                                            for p in (0, .05, .25, .5, .75, .95, 1)},
            'relative_percent_bins': {
                label: sum(lower <= value < upper for value in ratios)
                for label, lower, upper in [('below_0', -math.inf, 0), ('0_to_5', 0, 5),
                                           ('5_to_10', 5, 10), ('10_to_20', 10, 20),
                                           ('20_to_30', 20, 30), ('30_to_40', 30, 40),
                                           ('40_or_above', 40, math.inf)]},
            'window_target_lengths': dict(Counter(row['target_tokens'] for row in rows)),
            'dataset_thirds': [aggregate(rows[len(rows) * i // 3:len(rows) * (i + 1) // 3])
                              for i in range(3)],
            'pearson_correlation': {
                'window_index_vs_excess_nll_per_token': correlation(list(range(len(rows))), excess),
                'baseline_nll_per_token_vs_excess_nll_per_token': correlation(
                    [row['baseline_nll'] / row['target_tokens'] for row in rows], excess)},
            'worst_five_by_excess_nll': worst[-5:],
            'worst_five_fraction_of_excess_nll': (
                sum(row['excess_nll'] for row in worst[-5:]) / total_excess if total_excess else None),
            'excluding_worst_five_posthoc_diagnostic': aggregate(worst[:-5]),
            'final_window': rows[-1],
            'limitations': ['No new forward pass or within-window token-position loss measurements.',
                            'Dataset position is not recurrent-state age: windows reset state.',
                            'Post hoc exclusions are diagnostics, not replacement quality scores.',
                            'Descriptive correlations do not establish causality.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    blobs = {'baseline': args.baseline.read_bytes(), 'candidate': args.candidate.read_bytes()}
    result = audit(json.loads(blobs['baseline']), json.loads(blobs['candidate']))
    result['report_sha256'] = {key: hashlib.sha256(value).hexdigest() for key, value in blobs.items()}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
