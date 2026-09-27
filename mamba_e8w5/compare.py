"""Compare paired reports without merging different token or execution protocols."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(baseline, candidate):
    for report in (baseline, candidate):
        if not report.get('complete'):
            raise ValueError('Both evaluation reports must be complete')
    for field in ('suite', 'execution', 'source_checkpoint_sha256', 'tokenizer_sha256',
                  'runtime_source_sha256', 'evaluation_source_sha256'):
        if baseline[field] != candidate[field]:
            raise ValueError(f'Protocols differ: {field}')
    result = {'suite': baseline['suite'], 'execution': baseline['execution'],
              'candidate_package': candidate['package_receipt'],
              'quality_equivalence_established': False}
    if 'ppl' in baseline or 'ppl' in candidate:
        a, b = baseline['ppl'], candidate['ppl']
        if baseline['dataset'] != candidate['dataset']:
            raise ValueError('Dataset identities differ')
        for field in ('target_tokens', 'execution', 'cache_dtype', 'coverage'):
            if a[field] != b[field]:
                raise ValueError(f'PPL protocols differ: {field}')
        if len(a['windows']) != len(b['windows']):
            raise ValueError('PPL window counts differ')
        ratios = []
        for first, second in zip(a['windows'], b['windows']):
            for field in ('start', 'target_tokens', 'token_sha256_int64le'):
                if first[field] != second[field]:
                    raise ValueError(f'PPL windows differ: {field}')
            ratios.append(second['ppl']/first['ppl']-1)
        if not all(math.isfinite(x) for x in (a['ppl'], b['ppl'], *ratios)):
            raise ValueError('Nonfinite perplexity result')
        result['ppl'] = {'baseline': a['ppl'], 'candidate': b['ppl'],
                         'relative_change': b['ppl']/a['ppl']-1,
                         'worst_window_relative_change': max(ratios),
                         'target_tokens': a['target_tokens'], 'windows': len(ratios),
                         'coverage': a['coverage']}
    if 'mk' in baseline or 'mk' in candidate:
        a, b = baseline['mk'], candidate['mk']
        for field in ('protocol', 'split', 'execution', 'cache_dtype', 'generation', 'matching'):
            if a[field] != b[field]:
                raise ValueError(f'MK protocols differ: {field}')
        left = {row['id']: row for row in a['rows']}
        right = {row['id']: row for row in b['rows']}
        if set(left) != set(right) or len(left) != len(a['rows']) or len(right) != len(b['rows']):
            raise ValueError('MK cases differ or contain duplicates')
        mk = {}
        for condition in ('normal', 'target_removed'):
            paired = {'both_correct': 0, 'both_wrong': 0, 'lost': 0, 'gained': 0}
            count = 0
            for key, first in left.items():
                second = right[key]
                for field in ('condition', 'answer', 'prompt_token_sha256_int64le'):
                    if first[field] != second[field]:
                        raise ValueError(f'MK case differs: {key}/{field}')
                if first['condition'] != condition:
                    continue
                count += 1
                category = {(True, True): 'both_correct', (False, False): 'both_wrong',
                            (True, False): 'lost', (False, True): 'gained'}[
                                (first['correct'], second['correct'])]
                paired[category] += 1
            before = paired['both_correct']+paired['lost']
            after = paired['both_correct']+paired['gained']
            mk[condition] = {'baseline_correct': before, 'candidate_correct': after,
                             'count': count, 'delta_percentage_points': 100*(after-before)/count,
                             'paired': paired}
        mk['baseline_floor_warning'] = mk['normal']['baseline_correct']/mk['normal']['count'] < .25
        result['mk'] = mk
    if baseline['suite'] == 'full' and 'ppl' in result and 'mk' in result:
        normal, control = result['mk']['normal'], result['mk']['target_removed']
        expected_scope = (result['ppl']['coverage'] == 'full split' and normal['count'] == 48
                          and control['count'] == 48)
        engineering_target = (result['ppl']['relative_change'] <= .05 and
                              normal['candidate_correct'] >= normal['baseline_correct']-1 and
                              control['candidate_correct'] <= control['baseline_correct'])
        result['engineering_target'] = {
            'scope_matches_frozen_full_protocol': expected_scope,
            'numeric_thresholds_met': engineering_target,
            'recall_interpretable_above_floor': not result['mk']['baseline_floor_warning'],
            'note': 'Engineering screen only; small-sample recall is not proof of equivalence.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result['report_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
