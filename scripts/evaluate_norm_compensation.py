#!/usr/bin/env python3
"""Paired evaluation of only the final128-update, reloaded FP16 norm overlay.

Development uses the frozen four validation windows and24 MK cases. Complete
validation is optional and only runs after the predeclared development gate;
then source, original parent and candidate are compared in the same process.
No test split, intermediate checkpoint selection, or publication is performed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.norm_overlay import (sha, selected_norms, verify_overlay, apply_overlay,
                                    tensor_sha, PARENT_MANIFEST_SHA256)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def paired_dev_gate(parent, candidate):
    a, b = parent['ppl'], candidate['ppl']
    if a['target_tokens'] != b['target_tokens'] or len(a['windows']) != len(b['windows']):
        raise ValueError('Paired PPL coverage differs')
    for before, after in zip(a['windows'], b['windows']):
        if any(before[key] != after[key] for key in ('start', 'target_tokens', 'token_sha256_int64le')):
            raise ValueError('Paired PPL input identity differs')
    before_rows, after_rows = parent['mk']['rows'], candidate['mk']['rows']
    if len(before_rows) != 24 or len(after_rows) != 24:
        raise ValueError('Requires all24 predeclared development MK cases')
    paired = []
    for before, after in zip(before_rows, after_rows):
        if any(before[key] != after[key] for key in ('id', 'condition', 'prompt_token_sha256_int64le')):
            raise ValueError('Paired MK input identity differs')
        paired.append({'id': before['id'], 'condition': before['condition'],
                       'parent_correct': before['correct'], 'candidate_correct': after['correct']})
    am, bm = parent['mk']['summary'], candidate['mk']['summary']
    for rows, summary in ((before_rows, am), (after_rows, bm)):
        if len({row['id'] for row in rows}) != 24:
            raise ValueError('MK case identities are not unique')
        for condition in ('normal', 'target_removed'):
            subset = [row for row in rows if row['condition'] == condition]
            if len(subset) != 12 or sum(bool(row['correct']) for row in subset) != summary[condition]['correct']:
                raise ValueError('MK summary or condition coverage differs')
    ppl_pass = b['ppl'] <= .99 * a['ppl']
    normal_pass = bm['normal']['correct'] >= am['normal']['correct']
    removed_pass = bm['target_removed']['correct'] <= am['target_removed']['correct']
    return {'gate_passed': ppl_pass and normal_pass and removed_pass,
            'ppl_gate_passed': ppl_pass, 'normal_MK_gate_passed': normal_pass,
            'target_removed_MK_gate_passed': removed_pass,
            'mean_nll_improvement': (a['nll'] - b['nll']) / a['target_tokens'],
            'ppl_relative_change': b['ppl'] / a['ppl'] - 1, 'paired_MK': paired}


def replace_norm_refs(model, references):
    for name, value in references.items():
        owner, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(owner), leaf, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Pinned tokenizer; source checkpoint used only for optional post-gate reference validation')
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--overlay-dir', type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path, default=Path('calibration/v1/manifest.json'))
    parser.add_argument('--calibration-tokens', type=Path, default=Path('calibration/v1/calibration_tokens.pt'))
    parser.add_argument('--protocol', type=Path, default=Path('docs/NORM_COMPENSATION_PROTOCOL.md'))
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--training-checkpoint', type=Path, required=True)
    parser.add_argument('--smoke-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--full-validation-if-improved', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    parent, overlay, receipt = verify_overlay(args.parent_dir, args.overlay_dir,
        calibration_manifest=args.calibration_manifest, calibration_tokens=args.calibration_tokens,
        protocol=args.protocol, training_report=args.training_report,
        training_checkpoint=args.training_checkpoint, smoke_report=args.smoke_report)
    report = {'complete': False, 'format': 'MAMBA2_E8W5_NORM_EVALUATION_V1',
              'scope': 'Fixed final-step norm compensation; no publication approval',
              'script_sha256': sha(__file__), 'protocol_sha256': sha(args.protocol),
              'overlay_receipt': receipt, 'verify_only': args.verify_only}
    if args.verify_only:
        report.update(complete=True, quality_evaluated=False)
        write_json(args.report, report)
        return
    import torch
    from mamba_e8w5.runtime import (SentencePieceTokenizer, load_quantized_model, load_source_model,
                                    environment_receipt, gpu_memory_receipt, token_digest)
    from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
    from mamba_e8w5.evaluation import evaluate_ppl, evaluate_mk, ppl_windows
    from audit_decoded import audit_loaded_model
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    if tokenizer.sha256 != parent['tokenizer_sha256']:
        raise ValueError('Tokenizer differs')
    receipt['tokenizer_bytes'] = tokenizer.path.stat().st_size
    receipt['logical_raw_plus_manifests_tokenizer_bytes'] = receipt['logical_candidate_data_bytes'] + sum(receipt['manifest_bytes'].values()) + receipt['tokenizer_bytes']
    report.update(runtime_source_sha256=sha(ROOT/'mamba_e8w5/runtime.py'),
                  evaluation_source_sha256=sha(ROOT/'mamba_e8w5/evaluation.py'),
                  norm_overlay_source_sha256=sha(ROOT/'mamba_e8w5/norm_overlay.py'),
                  decoded_audit_source_sha256=sha(ROOT/'scripts/audit_decoded.py'),
                  codec_source_sha256=sha(ROOT/'mamba_e8w5/codec.py'),
                  environment=environment_receipt(), execution='prefill',
                  cache_dtype='FP16 recurrent MK cache; SSD scan for prose prefill')
    report['declared_dev_gate'] = {'candidate_PPL_at_most_parent_times': .99,
        'normal_MK': 'correct count must not decrease', 'target_removed_MK': 'correct count must not increase',
        'limitation': '12 normal cases cannot establish full-test recall preservation'}
    write_json(args.report, report)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', WIKITEXT_REVISION)
    reference_path = ROOT/'reports/e8w5_dev_prefill.json'
    reference = json.loads(reference_path.read_text())
    if dataset != reference['dataset'] or any(report[name+'_source_sha256'] != reference[name+'_source_sha256'] for name in ('runtime','evaluation')):
        raise ValueError('Frozen development dataset or runtime/evaluation changed')
    report['historical_dev_protocol_reference_sha256'] = sha(reference_path)
    report['dataset'] = dataset
    windows = ppl_windows(ids, 1024, 4)
    if len(windows) != len(reference['ppl']['windows']):
        raise ValueError('Development window count differs')
    for (first, tokens), previous in zip(windows, reference['ppl']['windows']):
        if (first, len(tokens)-1, token_digest(tokens.numpy())) != (previous['start'],previous['target_tokens'],previous['token_sha256_int64le']):
            raise ValueError('Development token identity differs')
    model = load_quantized_model(args.parent_dir)
    if model._package_receipt['manifest_sha256'] != PARENT_MANIFEST_SHA256:
        raise ValueError('Loaded parent identity differs')
    report['parent_package_receipt'] = model._package_receipt
    report['loaded_parent_full_audit'] = audit_loaded_model(model, parent, args.parent_dir)
    report['parent_dev'] = {'ppl': evaluate_ppl(model, windows),
                            'mk': evaluate_mk(model, tokenizer, 'validation', 2, 'prefill')}
    write_json(args.report, report)
    parent_norms = {name:model.get_parameter(name) for name in selected_norms()}
    report['overlay_application'] = apply_overlay(model, args.overlay_dir, overlay)
    candidate_norms = {name:model.get_parameter(name) for name in selected_norms()}
    report['candidate_package_receipt'] = {key:receipt[key] for key in
        ('parent_manifest_sha256','overlay_manifest_sha256','resolved_file_ledger_sha256','logical_candidate_data_bytes')}
    report['candidate_package_receipt']['format'] = 'Original2-sweep parent plus reloaded final FP16 norm replacement'
    report['candidate_dev'] = {'ppl': evaluate_ppl(model, windows),
                               'mk': evaluate_mk(model, tokenizer, 'validation', 2, 'prefill')}
    report['dev_comparison'] = paired_dev_gate(report['parent_dev'], report['candidate_dev'])
    write_json(args.report, report)
    if args.full_validation_if_improved and report['dev_comparison']['gate_passed']:
        full_windows = ppl_windows(ids, 2048, None)
        if len(ids)-1 != 264764:
            raise ValueError('Full validation target identity differs')
        report['candidate_full_validation'] = evaluate_ppl(model, full_windows)
        write_json(args.report, report)
        try:
            replace_norm_refs(model, parent_norms)
            if any(tensor_sha(model.get_parameter(name)) != tensor_sha(value) for name,value in parent_norms.items()):
                raise ValueError('Restored parent norms differ')
            report['parent_full_validation_same_process'] = evaluate_ppl(model, full_windows)
        finally:
            replace_norm_refs(model, candidate_norms)
        write_json(args.report, report)
        # This declared reference-only dependency is used after the development
        # gate, never to load the candidate or choose a training checkpoint.
        source = load_source_model(args.source_dir)
        report['source_reference_checkpoint_sha256'] = parent['source_checkpoint_sha256']
        report['source_full_validation_same_process'] = evaluate_ppl(source, full_windows)
        del source
        source_result, parent_result, candidate_result = (report[key] for key in
            ('source_full_validation_same_process','parent_full_validation_same_process','candidate_full_validation'))
        if any(result['target_tokens'] != 264764 for result in (source_result,parent_result,candidate_result)):
            raise ValueError('Full validation coverage differs')
        for a,b,c in zip(source_result['windows'],parent_result['windows'],candidate_result['windows']):
            for key in ('start','target_tokens','token_sha256_int64le'):
                if a[key] != b[key] or b[key] != c[key]:
                    raise ValueError('Full-validation paired token identity differs')
        report['full_validation_comparison'] = {
            'candidate_vs_parent_ppl_relative_change': candidate_result['ppl']/parent_result['ppl']-1,
            'candidate_vs_source_ppl_relative_change': candidate_result['ppl']/source_result['ppl']-1,
            'parent_vs_source_ppl_relative_change': parent_result['ppl']/source_result['ppl']-1,
            'source_plus5_percent_PPL_target_passed': candidate_result['ppl'] <= 1.05*source_result['ppl'],
            'full_quality_target_established': False,
            'scope': 'Paired full validation only. Source-relative PPL is separate from parent-relative dev gate; full recall preservation and untouched test quality are not established.'}
    else:
        report['full_validation_skipped'] = 'Not requested' if not args.full_validation_if_improved else 'Predeclared development gate did not pass'
    report.update(complete=True, quality_evaluated=True, elapsed_seconds=time.perf_counter()-started,
                  gpu_memory=gpu_memory_receipt())
    write_json(args.report, report)
    print(json.dumps({'complete': True, 'parent_dev_PPL': report['parent_dev']['ppl']['ppl'],
                      'candidate_dev_PPL': report['candidate_dev']['ppl']['ppl'],
                      'dev_gate_passed': report['dev_comparison']['gate_passed'],
                      'full_validation_evaluated': 'candidate_full_validation' in report}), flush=True)


if __name__ == '__main__':
    main()
