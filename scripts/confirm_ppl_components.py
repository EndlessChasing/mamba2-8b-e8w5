#!/usr/bin/env python3
"""Confirm E8 versus W5 attribution over the entire frozen validation split.

Exactly four interventions: original, full E8/W5, E8 only, and W5 vocabularies
only. No test data, MK, training, or parameter search is performed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.calibration import load_wikitext_tokens
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from mamba_e8w5.runtime import (SentencePieceTokenizer, environment_receipt,
    gpu_memory_receipt, load_quantized_model, load_source_model, sha256_file)
from diagnose_ppl_components import endpoint_gate, deltas, write_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--prior-diagnostic', type=Path, default=Path('reports/ppl_components_dev.json'))
    ap.add_argument('--report', type=Path, default=Path('reports/ppl_components_full_validation.json'))
    args = ap.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.prior_diagnostic.read_text())
    if not prior['complete'] or len(prior['decoded_E8_fp16_hashes']) != 112:
        raise ValueError('Requires complete prior factorial with all112 decoded E8 hashes')
    for name in ('runtime', 'evaluation'):
        if sha256_file(ROOT/'mamba_e8w5'/f'{name}.py') != prior[name+'_source_sha256']:
            raise ValueError(f'Frozen {name} differs')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation', prior['dataset']['revision_argument'])
    if dataset != prior['dataset'] or len(ids) != 264765:
        raise ValueError('Validation tokenstream differs from frozen diagnostic')
    windows = ppl_windows(ids, 2048, None)
    if sum(len(w)-1 for _, w in windows) != 264764:
        raise ValueError('Incomplete target coverage')
    source = load_source_model(args.source_dir)
    quantized = load_quantized_model(args.raw_dir)
    if quantized._package_receipt != prior['quantized_receipt']:
        raise ValueError('Quantized package identity differs from prior112-hash evidence')
    if source._package_receipt != prior['source_receipt']:
        raise ValueError('Source identity differs')
    manifest = json.loads((args.raw_dir/'manifest.json').read_text())
    if {name: entry['decoded_fp16_sha256'] for name, entry in manifest['matrices'].items()} != prior['decoded_E8_fp16_hashes']:
        raise ValueError('Decoded tensor ledger differs from prior verified evidence')
    original, coded = dict(source.named_parameters()), dict(quantized.named_parameters())
    projections = sorted(k for k in original if k.endswith(('mixer.in_proj.weight', 'mixer.out_proj.weight')))
    groups = {'E8_projections': projections, 'W5_embedding': ['backbone.embedding.weight'], 'W5_lm_head': ['lm_head.weight']}
    changed = set().union(*[set(names) for names in groups.values()])
    if original.keys() != coded.keys() or len(projections) != 112:
        raise ValueError('Unexpected parameter inventory')
    unchanged = set(original)-changed
    for name in unchanged:
        if not torch.equal(original[name], coded[name]):
            raise ValueError(f'Unchanged parameter differs: {name}')
    bindings = {name: (source.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1]) for name in changed}

    def assign(mask):
        for bit, names in zip(mask, groups.values()):
            for name in names:
                module, leaf = bindings[name]
                setattr(module, leaf, coded[name] if bit == '1' else original[name])

    # The prior endpoint probe explained the ownership question without claiming
    # to explain a tiny historical cross-run numerical discrepancy. Require the
    # direct-object/swap parity again in this process, while explicitly recording
    # (rather than treating as exact) any prior-process numerical discrepancy.
    dev_windows = ppl_windows(ids, 1024, 4)
    direct_q_dev = evaluate_ppl(quantized, dev_windows)
    prior_anchor_gate = endpoint_gate(direct_q_dev, prior['factorial']['111']['ppl'], require_pass=False)
    assign('111')
    swapped_q_dev = evaluate_ppl(source, dev_windows)
    same_process_gate = endpoint_gate(swapped_q_dev, direct_q_dev)
    assign('000')
    report = {
        'complete': False, 'suite': 'entire validation confirmation', 'execution': 'prefill',
        'scope': 'Four predeclared factor settings; entire validation split; no MK/test/training/tuning',
        'script_sha256': sha256_file(__file__),
        'imported_diagnostic_utilities_sha256': sha256_file(Path(__file__).with_name('diagnose_ppl_components.py')),
        'runtime_source_sha256': prior['runtime_source_sha256'],
        'evaluation_source_sha256': prior['evaluation_source_sha256'],
        'prior_diagnostic_sha256': sha256_file(args.prior_diagnostic),
        'decoded_E8_evidence': {'kind': 'Reuse verified112 tensor SHA ledger from prior diagnostic, same freshly verified package files and frozen decoder',
                              'matrices': 112, 'prior_report_sha256': sha256_file(args.prior_diagnostic),
                              'manifest_sha256': quantized._package_receipt['manifest_sha256'],
                              'codec_source_sha256': sha256_file(ROOT/'mamba_e8w5/codec.py'),
                              'tensor_hashes': prior['decoded_E8_fp16_hashes']},
        'source_receipt': source._package_receipt, 'quantized_receipt': quantized._package_receipt,
        'dataset': dataset, 'environment': environment_receipt(),
        'coverage': {'split': 'validation', 'seqlen': 2048, 'windows': len(windows), 'target_tokens': 264764,
                     'rule': 'Every next-token target exactly once; shared boundary input; zero state for each window'},
        'anchor': {'direct_quantized_dev': direct_q_dev, 'swapped_quantized_dev': swapped_q_dev,
                   'prior_current_Q_comparison': prior_anchor_gate, 'same_process_Q_gate': same_process_gate,
                   'note': 'Previous diagnostic observed tiny cross-process Q numerical differences despite decoded tensor identity; only current-process direct-versus-swapped parity is a strict gate. Historical differences are retained explicitly.'},
        'unchanged_parameter_tensors_verified_equal': len(unchanged),
        'bit_order': list(groups), 'cases': {},
        'interpretation': 'PPL changes are not additive. Report conditional mean NLL effects, and distinguish this2048-token validation confirmation from the1024-token four-window screen.'}
    write_report(args.report, report)
    try:
        for mask in ('000', '111', '100', '011'):
            assign(mask)
            result = evaluate_ppl(source, windows, execution='prefill')
            if result['target_tokens'] != 264764 or len(result['windows']) != len(windows):
                raise ValueError('Unexpected PPL coverage')
            entry = {'quantized_factors': [key for key, bit in zip(groups, mask) if bit == '1'], 'ppl': result}
            report['cases'][mask] = entry
            if mask != '000':
                entry['delta_from_source'] = deltas(result, report['cases']['000']['ppl'])
            write_report(args.report, report)
            print(json.dumps({'stage': 'full_validation', 'mask': mask, 'ppl': result['ppl'], 'nll': result['nll'],
                              'mean_nll_delta': entry.get('delta_from_source', {}).get('mean_nll_delta_nats_per_token', 0.)}), flush=True)
    finally:
        assign('000')
    if any(source.get_parameter(name) is not p for name, p in original.items()):
        raise RuntimeError('Original parameter references were not restored')
    effects = {}
    for label, pairs in {'E8_projections': [('000', '100'), ('011', '111')],
                         'W5_both_vocabularies': [('000', '011'), ('100', '111')]}.items():
        effects[label] = [{'from': first, 'to': second,
                          **deltas(report['cases'][second]['ppl'], report['cases'][first]['ppl'])}
                         for first, second in pairs]
    report.update(complete=True, original_parameter_references_restored=True, conditional_nll_effects=effects,
                  elapsed_seconds=time.perf_counter()-started, gpu_memory=gpu_memory_receipt())
    write_report(args.report, report)
    print(json.dumps({'complete': True, 'cases': len(report['cases']), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
