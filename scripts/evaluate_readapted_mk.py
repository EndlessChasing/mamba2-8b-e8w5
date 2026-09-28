#!/usr/bin/env python3
"""Fixed public-v1 MK baseline for source and the final448 readapted model.

No fitting, test prompts, adapter, PPL scoring, or model selection. Generation
calls the frozen native function; temporary hooks only inspect its outputs and
cache. Prefill cache rounding is not tokenwise rounding of the input prompt.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
FORMAT = 'MAMBA2_READAPTED_MK_BASELINE_V1'
PROTOCOL_SHA = '560a4d6c8c0805a56474aa9d1401bf4f373e0f1f3e6ed1edeaeee6e80026b177'
CURRENT_EVAL_SHA = '3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4'
CURRENT_MANIFEST_SHA = '3ede80b2883d8b8987cb3e2198fd8badc2dbbb5af03c8fe6cac7f5f9f5b62047'
SOURCE, CURRENT = 'source_fp16', 'current_axis_readapted_small'
CODE_PINS = {
    'mamba_e8w5/axis_small_base.py': '4e7026475fc97b005105b27043a15bae83c576934612b1e346576438eef4da70',
    'scripts/evaluate_axis_small.py': 'f2c2309f1dee45d32b8c4fcb9114a8d0f594cb56277b3a194a039070c361ac4c',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
}
MATCH = re.compile(r'(?<!\d)\d{6}(?!\d)')
RECORD_PATTERNS = (
    re.compile(r'^(\d{6}): (\d{6})$', re.MULTILINE),
    re.compile(r'^Key (\d{6}) has value (\d{6})\.$', re.MULTILINE),
    re.compile(r'^(\d{6}) -> (\d{6})$', re.MULTILINE),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError(f'Bound file identity differs: {path}')


def predict(output):
    match = MATCH.search(output)
    return match.group() if match else None


def wilson(correct, count):
    if type(correct) is not int or type(count) is not int or not 0 <= correct <= count or count < 1:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return {'correct': correct, 'count': count, 'accuracy': p,
            'wilson95': [0. if correct == 0 else max(0., center - radius),
                         1. if correct == count else min(1., center + radius)]}


def exact_mcnemar(gained, lost):
    if any(type(x) is not int or x < 0 for x in (gained, lost)):
        raise ValueError('Invalid discordant counts')
    n = gained + lost
    if not n:
        return 1.
    # Conditional two-sided Binomial(n, .5) test; no SciPy dependency.
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / (1 << n))


def summarize(rows):
    summary, cells = {}, []
    for condition in ('normal', 'target_removed'):
        chosen = [x for x in rows if x['condition'] == condition]
        summary[condition] = wilson(sum(x['correct'] for x in chosen), len(chosen))
        for size in (16, 64):
            for template in range(3):
                cell = [x for x in chosen if x['N'] == size and x['template'] == template]
                cells.append({'N': size, 'template': template, 'condition': condition,
                              **wilson(sum(x['correct'] for x in cell), len(cell)),
                              'query_positions': [x['query_position'] for x in cell]})
    return {'summary': summary, 'cells': cells}


def paired_comparison(source_rows, current_rows):
    if len(source_rows) != len(current_rows):
        raise ValueError('Paired case counts differ')
    result = {}
    for condition in ('normal', 'target_removed'):
        counts = dict(both_correct=0, both_wrong=0, gained=0, lost=0)
        for a, b in zip(source_rows, current_rows):
            for key in ('id', 'condition', 'answer', 'prompt_token_sha256_int64le', 'N', 'template', 'query_position'):
                if a[key] != b[key]:
                    raise ValueError('Paired prompt identity differs')
            if a['condition'] != condition:
                continue
            category = {(True, True): 'both_correct', (False, False): 'both_wrong',
                        (False, True): 'gained', (True, False): 'lost'}[(a['correct'], b['correct'])]
            counts[category] += 1
        n = sum(counts.values())
        result[condition] = {'source_correct': counts['both_correct'] + counts['lost'],
            'current_correct': counts['both_correct'] + counts['gained'], 'count': n,
            'delta_percentage_points': 100 * (counts['gained'] - counts['lost']) / n,
            'paired': counts, 'mcnemar_exact_two_sided_p': exact_mcnemar(counts['gained'], counts['lost']),
            'direction': 'higher recall is better' if condition == 'normal' else 'target-absent matches are false recalls; lower is better'}
    result['interpretation'] = 'Descriptive paired synthetic baseline; no equivalence, adapter-selection or promotion gate.'
    return result


def prepare_cases(evaluation, tokenizer):
    cases = evaluation.synthetic_mk_cases('validation', samples_per_cell=64)
    if cases != evaluation.synthetic_mk_cases('validation', samples_per_cell=64) or len(cases) != 768:
        raise ValueError('Frozen generator is not deterministic or coverage differs')
    ids, token_hashes, replay_ids = set(), set(), []
    lengths = []
    for index in range(0, len(cases), 2):
        normal, removed = cases[index:index + 2]
        size, template = normal['N'], normal['template']
        if size not in (16, 64) or template not in range(3):
            raise ValueError('Unexpected public-v1 cell')
        if normal['condition'] != 'normal' or removed['condition'] != 'target_removed' or removed['id'] != normal['id'] + '-removed':
            raise ValueError('Normal/control ordering differs')
        for key in ('N', 'template', 'query_position', 'key', 'answer'):
            if normal[key] != removed[key]:
                raise ValueError('Control query binding differs')
        records = [RECORD_PATTERNS[template].findall(row['prompt']) for row in (normal, removed)]
        query = (str(normal['key']), normal['answer'])
        position = normal['query_position']
        if not 0 <= position < size or records[0][position] != query:
            raise ValueError('Normal queried binding is absent or misplaced')
        for record in records:
            if len(record) != size or len({k for k, _ in record}) != size or len({v for _, v in record}) != size:
                raise ValueError('Record count or unique binding keys/values differs')
        if (sum(k == query[0] for k, _ in records[0]) != 1 or sum(v == query[1] for _, v in records[0]) != 1
                or any(k == query[0] or v == query[1] for k, v in records[1])
                or normal['answer'] in removed['prompt']
                or [i for i, (a, b) in enumerate(zip(*records)) if a != b] != [position]):
            raise ValueError('Target-removed control contains a target or changes another record')
        for row in (normal, removed):
            encoded = tokenizer.encode(row['prompt'])
            digest = evaluation.token_digest(encoded)
            if not encoded or len(encoded) + 12 > 4096 or row['id'] in ids or digest in token_hashes:
                raise ValueError('Empty, overlong or duplicated MK input')
            ids.add(row['id']); token_hashes.add(digest); lengths.append(len(encoded))
            row.update(prompt_tokens=len(encoded), prompt_token_sha256_int64le=digest)
            if normal['id'].endswith('-s0'):
                replay_ids.append(row['id'])
    if len(replay_ids) != 12:
        raise ValueError('Exactly12 first-cell normal/control replays required')
    cells = {}
    for row in cases:
        key = f"n{row['N']}-t{row['template']}-{row['condition']}"
        cells[key] = cells.get(key, 0) + 1
    if len(cells) != 12 or set(cells.values()) != {64}:
        raise ValueError('Exact384 normal+384 control coverage required')
    legacy = evaluation.synthetic_mk_cases('validation', samples_per_cell=2)
    legacy_hashes = {evaluation.token_digest(tokenizer.encode(row['prompt'])) for row in legacy}
    overlap = sorted(row['id'] for row in cases if row['prompt_token_sha256_int64le'] in legacy_hashes)
    serialized = json.dumps(cases, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return cases, {'generator': 'frozen public-mamba8-mk-v1', 'split': 'validation',
        'samples_per_cell': 64, 'normal_cases': 384, 'target_removed_cases': 384,
        'cell_counts': cells, 'unique_case_ids': len(ids), 'unique_prompt_token_digests': len(token_hashes),
        'minimum_prompt_tokens': min(lengths), 'maximum_prompt_tokens': max(lengths),
        'maximum_prompt_plus_generation_tokens': max(lengths) + 12,
        'prompt_plan_sha256_utf8_canonical_json': hashlib.sha256(serialized).hexdigest(),
        'replay_case_ids': replay_ids, 'legacy24_prompt_token_overlap_case_ids': overlap,
        'legacy24_case_id_overlap_count': len(ids & {row['id'] for row in legacy}),
        'exposure': 'Observed development family, not untouched confirmation. Increasing sample count changes later RNG draws; ID overlap alone is not input overlap.'}


def identities(model):
    result = {name: {'object_id': id(value), 'data_ptr': value.data_ptr(), 'version': value._version,
                     'dtype': str(value.dtype), 'shape': list(value.shape), 'stride': list(value.stride()),
                     'requires_grad': value.requires_grad} for name, value in model.named_parameters()}
    if len(result) != 507 or any(row['dtype'] != 'torch.float16' or row['requires_grad'] for row in result.values()):
        raise ValueError('Expected507 frozen FP16 native parameters')
    return result


def inspect_cache(cache):
    pairs = cache.key_value_memory_dict
    tensors = [tensor for pair in pairs.values() for tensor in pair]
    if len(pairs) != 56 or len(tensors) != 112 or any(t.dtype != torch.float16 for t in tensors):
        raise ValueError('Expected56 pairs of native FP16 conv/SSM state')
    if not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
        raise ValueError('Nonfinite native conv/SSM cache')
    return {'layers': 56, 'tensors': 112, 'dtype': 'torch.float16', 'finite': True,
            'bytes': sum(t.numel() * t.element_size() for t in tensors)}


@torch.inference_mode()
def score_case(model, tokenizer, evaluation, case):
    observations = {'forward_calls': 0}

    def observe(_module, _args, kwargs, output):
        if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
            raise ValueError('Nonfinite native backbone hidden state')
        cache = kwargs.get('inference_params')
        if cache is None:
            raise ValueError('Frozen generation did not pass a native cache')
        if observations['forward_calls'] == 0:
            if cache.seqlen_offset != 0 or output.shape[1] != case['prompt_tokens']:
                raise ValueError('Fresh native whole-prompt prefill required')
            observations['prefill_cache'] = inspect_cache(cache)
        elif cache is not observations['cache']:
            raise ValueError('Generation replaced the prompt cache')
        observations['cache'] = cache
        observations['forward_calls'] += 1

    handle = model.backbone.register_forward_hook(observe, with_kwargs=True)
    try:
        with torch.autocast(device_type='cuda', enabled=False):
            output, generated, count, memory = evaluation.generate_greedy(
                model, tokenizer, case['prompt'], max_new_tokens=12, execution='prefill')
        if count != case['prompt_tokens'] or not 1 <= len(generated) <= 12:
            raise ValueError('Frozen generated length/prompt identity differs')
        final_cache = inspect_cache(observations['cache'])
        if final_cache['bytes'] != memory or observations['prefill_cache'] != final_cache:
            raise ValueError('Cache coverage or dtype changed during generation')
        prediction = predict(output)
        return {**{key: value for key, value in case.items() if key != 'prompt'},
                'output': output, 'generated_ids': generated, 'prediction': prediction,
                'correct': prediction == case['answer'], 'cache_bytes': memory,
                'finite_hidden_every_backbone_call': True, 'backbone_calls': observations['forward_calls'],
                'prefill_cache': observations['prefill_cache'], 'final_cache': final_cache}
    finally:
        handle.remove()
        observations.clear()


def score_arm(model, tokenizer, evaluation, shared, expected, cases, plan, arm, args, report):
    entry = report['arms'][arm] = {'complete': False, 'rows': [], 'replays': []}
    expected_identity = identities(model)
    entry['identities507_before'] = expected_identity
    entry['actual507_before'] = shared.audit_model(model, expected)
    started = time.monotonic()
    try:
        for index, case in enumerate(cases):
            entry['rows'].append(score_case(model, tokenizer, evaluation, case))
            if (index + 1) % 16 == 0:
                write_json(args.report, report)
                print(f"[{arm}] {index + 1}/{len(cases)} MK prompts", flush=True)
        entry.update(summarize(entry['rows']))
        by_id = {row['id']: row for row in entry['rows']}
        for case in cases:
            if case['id'] not in plan['replay_case_ids']:
                continue
            repeated = score_case(model, tokenizer, evaluation, case)
            if repeated != by_id[case['id']]:
                entry['replays'].append({'id': case['id'], 'exact_equal': False, 'actual': repeated})
                raise ValueError(f"Same-arm prompt replay differs: {case['id']}")
            entry['replays'].append({'id': case['id'], 'exact_equal': True,
                                    'generated_ids': repeated['generated_ids'], 'output': repeated['output'],
                                    'prediction': repeated['prediction'], 'correct': repeated['correct']})
        if len(entry['replays']) != 12:
            raise ValueError('Incomplete12-case replay')
        entry['replay_exact'] = True
        entry['complete'] = True
    finally:
        entry['actual507_after'] = shared.audit_model(model, expected)
        entry['identities507_after'] = identities(model)
        entry['identity_storage_version_unchanged'] = entry['identities507_after'] == expected_identity
        entry['elapsed_seconds'] = time.monotonic() - started
        if not entry['identity_storage_version_unchanged']:
            entry['complete'] = False
            raise ValueError('Native scoring mutated a model parameter identity/storage/version')
        write_json(args.report, report)


def self_test():
    assert predict('Value: 123456\n789012') == '123456'
    assert predict('0123456') is None and predict('12345') is None and predict('No value.') is None
    assert predict('Answer: 012345.') == '012345'
    assert wilson(0, 384)['wilson95'][0] == 0.
    assert wilson(384, 384)['wilson95'][1] == 1.
    assert exact_mcnemar(0, 0) == 1. and exact_mcnemar(3, 0) == exact_mcnemar(0, 3) == .25
    assert exact_mcnemar(5, 5) == 1.
    before = [{'id': str(i), 'condition': condition, 'answer': '123456',
               'prompt_token_sha256_int64le': str(i), 'N': 16, 'template': 0,
               'query_position': 0, 'correct': condition == 'normal'}
              for i, condition in enumerate(('normal', 'target_removed'))]
    after = [{**row, 'correct': not row['correct']} for row in before]
    paired = paired_comparison(before, after)
    assert paired['normal']['paired']['lost'] == 1 and paired['normal']['delta_percentage_points'] == -100.
    assert paired['target_removed']['paired']['gained'] == 1
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'cuda_initialized': False,
            'checks': ['Frozen first-six-digit matcher boundaries and leading zero',
                       'Wilson endpoint ranges', 'Exact two-sided discordant binomial symmetry and known cases',
                       'Paired lost/gained orientation including adverse target-removed matches']}


def run(args, report):
    pins = {str((ROOT / name).resolve()): digest for name, digest in CODE_PINS.items()}
    pins[str(ROOT / 'docs/MK_RESURFACE_PROTOCOL.md')] = PROTOCOL_SHA
    pins[str(ROOT / 'reports/axis_small_readaptation_v1_eval.json')] = CURRENT_EVAL_SHA
    pins[str(ROOT / 'artifacts/axis_small_readaptation_v1/manifest.json')] = CURRENT_MANIFEST_SHA
    pins[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in pins.items():
        checked(path, digest)
    shared = importlib.import_module('mamba_e8w5.axis_small_base')
    exported = importlib.import_module('evaluate_axis_small')
    evaluation = importlib.import_module('mamba_e8w5.evaluation')
    context = shared.verify_inputs()
    # The frozen export verifier needs provenance CLI fields, but none are tunable here.
    export_args = argparse.Namespace(overlay_dir=ROOT / 'artifacts/axis_small_readaptation_v1',
        training_report=ROOT / 'reports/axis_small_readaptation_v1_train.json', training_checkpoint=None,
        smoke_report=ROOT / 'reports/axis_small_readaptation_v1_smoke.json',
        diagnostic_report=ROOT / 'reports/axis_small_diagnostic_v1.json')
    manifest, values, proof = exported.verify_export(export_args, context)
    previous = json.loads((ROOT / 'reports/axis_small_readaptation_v1_eval.json').read_text())
    if previous.get('complete') is not True or previous['integrity'] != proof:
        raise ValueError('Actual final448 export differs from the completed current PPL receipt')
    expected = {SOURCE: context['source_expected507'], CURRENT: proof['resolved507_candidate_fp16_sha256']}
    if expected[CURRENT] != previous['full_validation']['arms']['current_axis_readapted_small']['actual507_before']['unchanged_content_sha256']:
        raise ValueError('Current507 content identity differs from the measured final model')
    # Prepared prose windows are verified only as provenance; none are scored.
    del context['training_windows'], context['heldout_windows']
    tokenizer = shared.runtime.SentencePieceTokenizer(ROOT / 'models/source')
    cases, plan = prepare_cases(evaluation, tokenizer)
    report.update(self_test=self_test(), input_binding=context['binding'], export_integrity=proof,
        checked_input_sha256=pins, tokenizer_sha256=tokenizer.sha256, prompt_plan=plan, cases=cases,
        expected507_by_arm=expected, arms={}, current_storage=proof['storage'],
        prior_ppl_reference={'report_sha256': CURRENT_EVAL_SHA, 'ppl': 7.622396587826496,
                             'measured_again_here': False})
    write_json(args.report, report)
    if args.verify_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise ValueError('CPU verification requires CUDA_VISIBLE_DEVICES empty and CUDA uninitialized')
        report.update(complete=True, mode='verify_only', cuda_initialized=False,
                      actual_final448_export_verified=True, prompt_bindings_verified=True)
        report['final_base_input_recheck'] = shared.final_recheck(context)
        for path, digest in proof['checked_input_sha256'].items():
            if path in pins and pins[path] != digest:
                raise ValueError('Conflicting extra/model input binding')
            checked(path, digest)
        for path, digest in pins.items():
            checked(path, digest)
        report['final_extra_input_recheck'] = {'checked_input_sha256': {**proof['checked_input_sha256'], **pins}}
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    report['environment'] = shared.runtime.environment_receipt()
    model = None
    try:
        model = shared.runtime.load_source_model(ROOT / 'models/source').eval().requires_grad_(False)
        report['source_package_receipt'] = model._package_receipt
        score_arm(model, tokenizer, evaluation, shared, expected[SOURCE], cases, plan, SOURCE, args, report)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        model = shared.load_model(context).eval().requires_grad_(False)
        report['direct_axis_old393_load_audit'] = model._axis_load_audit
        report['base_package_receipt'] = model._package_receipt
        fixed_identity = shared.large_identity(model)
        report['installed_readapted393_audit'] = shared.install_small(model, values, expected[CURRENT])
        report['fixed114_after_readapted_install'] = shared.verify_fixed_large(model, context, fixed_identity)
        score_arm(model, tokenizer, evaluation, shared, expected[CURRENT], cases, plan, CURRENT, args, report)
        report['final_fixed114_audit'] = shared.verify_fixed_large(model, context, fixed_identity)
        report['comparison'] = paired_comparison(report['arms'][SOURCE]['rows'], report['arms'][CURRENT]['rows'])
        report.update(complete=True, mode='native_prefill_baseline', scored_prompts_per_arm=768,
                      replay_prompts_per_arm=12)
    finally:
        if model is not None:
            del model
        gc.collect()
        report['gpu_memory'] = shared.runtime.gpu_memory_receipt()
        report['final_base_input_recheck'] = shared.final_recheck(context)
        for path, digest in {**proof['checked_input_sha256'], **pins}.items():
            checked(path, digest)
        report['final_extra_input_recheck'] = {'checked_input_sha256': {**proof['checked_input_sha256'], **pins}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/readapted_mk_baseline_v1.json')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.report = args.report.resolve()
    if args.report.parent != ROOT / 'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('Fresh report directly under reports required')
    if args.verify_only and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run CPU verification with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(8)
    report = {'format': FORMAT, 'complete': False, 'script_sha256': sha(__file__),
        'protocol_sha256': PROTOCOL_SHA, 'arm_order': [SOURCE, CURRENT],
        'execution': 'native SSD prefill plus recurrent greedy generation',
        'cache_dtype': 'float16', 'prompt_cache_rounding': 'end of prefill, not every input token',
        'generation': 'full256000 vocabulary, greedy, maximum12 tokens, stop at EOS',
        'matching': 'first standalone six-digit integer equals answer',
        'training_performed': False, 'adapter_enabled': False, 'test_prompts_loaded': False,
        'ppl_evaluated': False, 'candidate_selection_performed': False, 'publication_performed': False,
        'limitations': ['Observed synthetic development baseline; not unseen confirmation or original Resurface MK.',
                       'Native FP16 weight expansion; no compressed residency or prompt-tokenwise state claim.',
                       'Baseline only: no MK-repair or PPL-preservation claim for an adapter.']}
    started = time.monotonic()
    try:
        run(args, report)
    except BaseException as error:
        report.update(complete=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(args.report, report)
    print(json.dumps({'complete': report['complete'], 'report': str(args.report),
                      'mode': report.get('mode'), 'comparison': report.get('comparison')}), flush=True)


if __name__ == '__main__':
    main()
