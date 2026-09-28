#!/usr/bin/env python3
"""Fixed beta1 paired diagnostic with exact same-process execution controls.

The failed v1 and successful one-window probe remain immutable. Historical
rounding differences are recorded; no temperature fit or quality advancement.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import diagnose_output_calibration as fixed
import diagnose_low_rank_generalization as old_pair

native = fixed.native
METRICS_SHA = '116c69a5151a2497209408ed4fef7561fcae38624b20e6bdf659b40b3126fd64'
PAIRED_SHA = 'e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879'
PROBE_SHA = 'b83fab9ae854e5b3622ee0960fcdc51e3d4e12210651cc5718ea2ea2be067e40'
DOCUMENT_SHA = '57058d5f78777a5cf12127bef628bde49f9e2a589a862b2cc459db6da63b6ecb'
HISTORICAL_RELATIVE_PPL_LIMIT = .001
ARMS = fixed.ARMS
ENV_ALLOWLIST = ('MAMBA_DETERMINISTIC', 'TRITON_CACHE_AUTOTUNING',
    'TRITON_AUTOTUNE_BLOCK_SIZE_M', 'TRITON_AUTOTUNE_BLOCK_SIZE_N',
    'TRITON_AUTOTUNE_BLOCK_SIZE_K', 'TRITON_AUTOTUNE_BLOCK_SIZE_DSTATE',
    'TRITON_CACHE_DIR', 'TRITON_INTERPRET', 'CUBLAS_WORKSPACE_CONFIG',
    'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')
AUTOTUNERS = {
    'mamba_ssm.ops.triton.ssd_chunk_scan': ('_chunk_scan_fwd_kernel', '_chunk_scan_fwd_kernel_wip'),
    'mamba_ssm.ops.triton.ssd_chunk_state': ('_chunk_cumsum_fwd_kernel', '_chunk_state_fwd_kernel'),
    'mamba_ssm.ops.triton.ssd_state_passing': ('_state_passing_fwd_kernel',),
    'mamba_ssm.ops.triton.ssd_bmm': ('_bmm_chunk_fwd_kernel',),
}


def reproducibility_snapshot():
    """Read existing process settings only; never import or execute a kernel."""
    observed = {}
    for module_name, names in AUTOTUNERS.items():
        module = sys.modules.get(module_name)
        for name in names:
            tuner = getattr(module, name, None) if module is not None else None
            cache = getattr(tuner, 'cache', None) if tuner is not None else None
            configs = getattr(tuner, 'configs', None) if tuner is not None else None
            observed[module_name+'.'+name] = {'module_loaded': module is not None,
                'attribute_present': tuner is not None,
                'cache_key_and_config_repr': [{'key': repr(key), 'config': repr(value)}
                    for key,value in cache.items()] if isinstance(cache,dict) else None,
                'available_config_repr': [repr(value) for value in configs] if isinstance(configs,(list,tuple)) else None,
                'best_config_repr': repr(getattr(tuner,'best_config')) if tuner is not None and hasattr(tuner,'best_config') else None}
    return {'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'deterministic_algorithms_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cuda_matmul_allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'environment_allowlist': {name: os.environ.get(name) for name in ENV_ALLOWLIST},
        'already_loaded_autotuner_metadata': observed,
        'scope': 'Missing/empty cache means not observed, not fixed; Python metadata only, no kernel imports or setting changes.'}


@torch.inference_mode()
def score_pair(source, student, window, chunk_tokens=64):
    """Copy the old native CE/KL sequence; append frozen beta1 metrics afterward."""
    device = source.backbone.embedding.weight.device
    if student.backbone.embedding.weight.device != device:
        raise ValueError('Paired models must share a device')
    ids, targets = window[None, :-1].to(device), window[None, 1:].to(device)
    collected = {'source': {}, 'student': {}}
    source_chunks, student_chunks, kl_chunks = [], [], []
    with torch.autocast(device_type=device.type, enabled=False):
        teacher_hidden = source.backbone(ids)
        student_hidden = student.backbone(ids)
        if (teacher_hidden.dtype != torch.float16 or student_hidden.dtype != torch.float16
                or not torch.isfinite(teacher_hidden).all() or not torch.isfinite(student_hidden).all()):
            raise ValueError('Expected finite native FP16 paired hidden states')
        for first in range(0, targets.shape[1], chunk_tokens):
            end = min(first+chunk_tokens, targets.shape[1])
            teacher_logits = source.lm_head(teacher_hidden[:, first:end]).float()
            student_logits = student.lm_head(student_hidden[:, first:end]).float()
            if (teacher_logits.shape != student_logits.shape or not torch.isfinite(teacher_logits).all()
                    or not torch.isfinite(student_logits).all()):
                raise FloatingPointError('Invalid full-vocabulary paired logits')
            truth = targets[:, first:end].flatten()
            teacher_ce = F.cross_entropy(teacher_logits.flatten(0,1), truth, reduction='sum')
            student_ce = F.cross_entropy(student_logits.flatten(0,1), truth, reduction='sum')
            teacher_logp = F.log_softmax(teacher_logits, dim=-1)
            student_logp = F.log_softmax(student_logits, dim=-1)
            kl = F.kl_div(student_logp, teacher_logp, reduction='sum', log_target=True)
            if not all(torch.isfinite(value) for value in (teacher_ce, student_ce, kl)):
                raise FloatingPointError('Nonfinite paired CE/KL')
            source_chunks.append(float(teacher_ce)); student_chunks.append(float(student_ce)); kl_chunks.append(float(kl))
            # New diagnostics execute only after the unchanged head/CE/KL sequence.
            for name, logits in (('source', teacher_logits), ('student', student_logits)):
                moments = fixed.output_statistics(logits.flatten(0,1), truth)
                for key, value in moments.items():
                    collected[name].setdefault(key, []).append(value.detach().cpu())
                del moments
            del teacher_logits, student_logits, teacher_logp, student_logp, teacher_ce, student_ce, kl, logits, value
    vectors = {name: {key: torch.cat(parts) for key, parts in values.items()} for name, values in collected.items()}
    source_row = fixed.summarize_vectors(vectors['source'], source_chunks)
    student_row = fixed.summarize_vectors(vectors['student'], student_chunks)
    chunks = [min(chunk_tokens, targets.numel()-first) for first in range(0, targets.numel(), chunk_tokens)]
    source_row['chunk_target_counts'] = chunks; student_row['chunk_target_counts'] = chunks
    control = {'target_tokens': targets.numel(), 'teacher_ce_chunk_sums': source_chunks,
        'student_ce_chunk_sums': student_chunks, 'teacher_kl_chunk_sums': kl_chunks,
        'source_nll': math.fsum(source_chunks), 'student_nll': math.fsum(student_chunks),
        'teacher_to_student_kl_sum': math.fsum(kl_chunks)}
    return source_row, student_row, control, vectors['source']['ce'].double(), vectors['student']['ce'].double()


def exact_control(actual, expected):
    keys = ('teacher_ce_chunk_sums', 'student_ce_chunk_sums', 'teacher_kl_chunk_sums')
    flags = {key: fixed.equal_doubles(actual[key], expected[key]) for key in keys}
    flags.update({key: fixed.equal_doubles([actual[key]], [expected[key]])
        for key in ('source_nll', 'student_nll', 'teacher_to_student_kl_sum')})
    flags['target_tokens'] = actual['target_tokens'] == expected['target_tokens'] == 2047
    return {'all_exact': all(flags.values()), 'fields_exact': flags}


def historical_drift(rows, prior, arm):
    details = []
    for index, row in enumerate(rows):
        old = prior['reserved_results']['best_small_e8w5' if arm == 'source_fp16' else arm][index]
        reference = old['teacher_ce_chunk_sums' if arm == 'source_fp16' else 'student_ce_chunk_sums']
        nll = old['source_nll' if arm == 'source_fp16' else 'student_nll']
        difference = [a-b for a,b in zip(row['ce_chunk_sums'], reference)]
        details.append({'index': index, 'prior_ce_chunk_sums': reference,
            'current_minus_prior_ce_chunks': difference,
            'chunk_bitwise_equal': [fixed.equal_doubles([a],[b]) for a,b in zip(row['ce_chunk_sums'],reference)],
            'current_minus_prior_nll': row['nll']-nll,
            'maximum_absolute_chunk_difference': max(map(abs,difference))})
    summary = fixed.aggregate(rows)
    prior_ppl = prior['reserved_summary']['ppl'][arm]
    relative = summary['ppl']/prior_ppl-1
    return {'per_window': details, 'prior_ppl': prior_ppl, 'current_ppl': summary['ppl'],
        'relative_ppl_drift': relative, 'absolute_relative_ppl_limit': HISTORICAL_RELATIVE_PPL_LIMIT,
        'within_descriptive_reproducibility_envelope': abs(relative) <= HISTORICAL_RELATIVE_PPL_LIMIT,
        'historical_bitwise_equality_claimed': False}


def run(args, report):
    document = ROOT/'docs/OUTPUT_CALIBRATION_DIAGNOSTIC_V2_PROTOCOL.md'
    extra = {ROOT/'scripts/diagnose_output_calibration.py': METRICS_SHA,
        ROOT/'scripts/diagnose_low_rank_generalization.py': PAIRED_SHA,
        ROOT/'reports/output_calibration_anchor_probe_v1.json': PROBE_SHA, document: DOCUMENT_SHA}
    for path, digest in extra.items():
        if digest.startswith('PENDING') or native.sha(path) != digest:
            raise ValueError(f'New paired protocol/input is not frozen: {path}')
    prior, manifest, data, heldout, plan, base_hashes, candidate_hashes, checked, integrity = fixed.verify_inputs()
    checked.update({str(path): digest for path,digest in extra.items()})
    checked[str(Path(__file__).resolve())] = native.sha(__file__)
    report.update(binding={'prior_evaluation_sha256': fixed.EVALUATION_SHA,
        'candidate_manifest_sha256': fixed.MANIFEST_SHA, 'probe_sha256': PROBE_SHA,
        'protocol_document_sha256': DOCUMENT_SHA, 'checked_input_sha256': checked}, integrity=integrity,
        dataset=data['dataset'], window_plan=plan, rows={arm: [] for arm in ARMS},
        repeated_source_rows=[], source_repeat_controls=[], paired_ce_kl={}, window0_old_paired_controls={})
    native.write_json(args.report, report)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest'); torch.cuda.reset_peak_memory_stats()
    report['environment'] = native.environment_receipt()
    report['reproducibility_before'] = reproducibility_snapshot()
    source = native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
    if source._package_receipt != prior['source_package_receipt']:
        raise ValueError('Source checkpoint identity differs')
    report['source_package_receipt'] = source._package_receipt
    report['source_coverage'] = native.audit_parameter_coverage(source)
    source_hashes = {name: native.tensor_sha_fp16(v) for name,v in source.named_parameters()}
    if len(source_hashes)!=507: raise ValueError('Expected507 source parameters')
    report['source507_initial_fp16_sha256'] = source_hashes
    source_identity = fixed.model_identity(source)
    model = native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    small = json.loads((ROOT/'artifacts/small_compensation_v1/manifest.json').read_text())
    native.small_overlay.apply_overlay(model, ROOT/'artifacts/small_compensation_v1', small)
    model.eval().requires_grad_(False)
    report['baseline507_actual_audit'] = native.audit_frozen_values(model,base_hashes)
    report['baseline_coverage'] = native.audit_parameter_coverage(model)
    ce_by_arm = {}

    def score_pass(arm):
        before = fixed.model_identity(model)
        controls = report['paired_ce_kl'][arm] = []
        source_parts, student_parts = [], []
        for i,(window,row_plan) in enumerate(zip(heldout,plan)):
            if i == 0:
                reference = old_pair.score_window(source,model,window,chunk_tokens=64)
                report['window0_old_paired_controls'][arm] = {'reference': reference}
                native.write_json(args.report,report)
            sr,st,actual,source_ce,student_ce = score_pair(source,model,window)
            sr.update(row_plan); st.update(row_plan)
            report['rows'][arm].append(st); controls.append(actual)
            if arm == ARMS[1]:
                report['rows'][ARMS[0]].append(sr)
            else:
                report['repeated_source_rows'].append(sr)
            if i == 0:
                comparison = exact_control(actual,reference)
                report['window0_old_paired_controls'][arm].update(actual=actual,comparison=comparison)
                native.write_json(args.report,report)  # Persist actual failing row/control before assertion.
                if not comparison['all_exact']:
                    raise ValueError(f'Same-process frozen paired window0 control differs: {arm}')
            if arm == ARMS[2]:
                first = report['rows'][ARMS[0]][i]
                repeat = {'index':i,'ce_exact':fixed.equal_doubles(sr['ce_chunk_sums']+[sr['nll']],
                    first['ce_chunk_sums']+[first['nll']]),
                    'all_metric_hashes_exact': sr['per_target_statistic_sha256']==first['per_target_statistic_sha256']}
                report['source_repeat_controls'].append(repeat)
                if not repeat['ce_exact'] or not repeat['all_metric_hashes_exact']:
                    native.write_json(args.report,report)
                    raise ValueError(f'Source same-process repeat differs at window{i}')
            source_parts.append(source_ce); student_parts.append(student_ce)
            if (i+1)%8==0:
                print(f'[output calibration v2] source+{arm} {i+1}/64',flush=True)
                native.write_json(args.report,report)
        if fixed.model_identity(model)!=before: raise ValueError('Paired inference changed parameter identity')
        if arm==ARMS[1]: ce_by_arm[ARMS[0]]=torch.cat(source_parts)
        ce_by_arm[arm]=torch.cat(student_parts)

    if source.lm_head.weight.shape[0]!=256000 or model.lm_head.weight.shape[0]!=256000:
        raise ValueError('Full256000-token vocabulary required')
    score_pass(ARMS[1])
    report['baseline507_after_audit']=native.audit_frozen_values(model,base_hashes)
    projections={entry['source_key'] for entry in native.expected_inventory().values()}
    inherited={name:v for name,v in model.named_parameters() if name not in projections}
    if len(inherited)!=395: raise ValueError('Expected395 inherited tensors')
    merged={}
    for label,entry in native.expected_inventory().items():
        path=native.checked_file(ROOT/'artifacts/teacher_kl_compensation_v1',entry['file'],manifest['files'][entry['file']])
        B,A=native.read_factors(path,expected_shape=entry['shape'],expected_rank=4)
        name=entry['source_key']; base=model.get_parameter(name)
        if native.tensor_sha_fp16(base)!=base_hashes[name]: raise ValueError('Baseline projection changed')
        value=native.merge_native_projection(base,B,A); digest=native.tensor_sha_fp16(value)
        if digest!=candidate_hashes[name]: raise ValueError(f'Previously measured candidate merge differs:{label}')
        owner,leaf=name.rsplit('.',1)
        setattr(model.get_submodule(owner),leaf,torch.nn.Parameter(value,requires_grad=False)); merged[label]=digest
    del base,value,B,A
    gc.collect(); torch.cuda.empty_cache()
    if any(model.get_parameter(name) is not value for name,value in inherited.items()):
        raise ValueError('Candidate replaced an inherited parameter')
    report['candidate112_merged_fp16_sha256']=merged
    report['candidate395_inherited_audit']=native.audit_frozen_values(model,{n:base_hashes[n] for n in inherited})
    report['candidate507_before_audit']=native.audit_frozen_values(model,candidate_hashes)
    report['candidate_coverage']=native.audit_parameter_coverage(model)
    score_pass(ARMS[2])
    report['candidate507_after_audit']=native.audit_frozen_values(model,candidate_hashes)
    report['source507_after_audit']=native.audit_frozen_values(source,source_hashes)
    if fixed.model_identity(source)!=source_identity: raise ValueError('Source parameter identity changed')
    report['summaries']={arm:fixed.aggregate(rows) for arm,rows in report['rows'].items()}
    report['source_surprisal_bins']=fixed.surprisal_bins(ce_by_arm)
    report['paired_windows']=[{**p,'candidate_minus_baseline_mean_nll':
        report['rows'][ARMS[2]][i]['mean_nll']-report['rows'][ARMS[1]][i]['mean_nll'],
        'per_target_means':{arm:report['rows'][arm][i]['per_target_means'] for arm in ARMS}}
        for i,p in enumerate(plan)]
    report['teacher_kl']={arm:{'sum':math.fsum(r['teacher_to_student_kl_sum'] for r in rows),
        'mean':math.fsum(r['teacher_to_student_kl_sum'] for r in rows)/131008}
        for arm,rows in report['paired_ce_kl'].items()}
    report['historical_drift']={arm:historical_drift(rows,prior,arm) for arm,rows in report['rows'].items()}
    report['reproducibility_after']=reproducibility_snapshot()
    report['source_repeat_all64_exact']=len(report['source_repeat_controls'])==64 and all(
        r['ce_exact'] and r['all_metric_hashes_exact'] for r in report['source_repeat_controls'])
    report['per_target_arrays_serialized']=False
    report['per_target_ce_arrays_retained_bytes']=sum(v.numel()*v.element_size() for v in ce_by_arm.values())
    report['computation_complete']=True
    native.write_json(args.report,report)
    for path,digest in checked.items():
        if native.sha(path)!=digest: raise ValueError(f'Bound input changed:{path}')
    if not all(r['within_descriptive_reproducibility_envelope'] for r in report['historical_drift'].values()):
        raise ValueError('Historical aggregate PPL drift exceeds declared0.1% descriptive envelope')
    report.update(complete=True,numerical_controls_accepted=True,gpu_memory=native.gpu_memory_receipt())


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(); args.report=args.report.resolve()
    if args.report.parent!=ROOT/'reports' or args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():
        raise ValueError('Require a new report directly under reports/')
    torch.set_num_threads(8); started=time.perf_counter()
    report={'format':'MAMBA2_OUTPUT_CALIBRATION_DIAGNOSTIC_V2','complete':False,
        'computation_complete':False,'numerical_controls_accepted':False,'posthoc':True,
        'script_sha256':native.sha(__file__),'inverse_temperature_beta':1.,'temperature_fitting':False,
        'temperature_sweep':False,'candidate_advancement':False,'quality_gate':None,
        'validation_or_test_computation':False,'metric_protocol':{**fixed.PROTOCOL,
            'arm_order':[['source_fp16','best_small_e8w5'],['source_fp16','teacher_kl_e8w5']],
            'ce_anchor':'Exact same-process old/new paired CE/KL on window0 of each pass and source CE/statistic hashes across64 windows; historical chunk differences explicitly recorded.',
            'historical_absolute_relative_ppl_limit':HISTORICAL_RELATIVE_PPL_LIMIT,
            'historical_bitwise_equality_claimed':False},
        'limitations':['Already observed64 TRAIN windows; descriptive analysis only.',
            'Historical bitwise equality is not claimed; every drift is retained and aggregate PPL must stay within0.1%.',
            'Same-process frozen paired window0 CE/KL and all64 source CE/metric repeats require exact equality.',
            'The historical envelope is an execution-reproducibility limit, not a model-quality gate.',
            'Triton/autotuner observations may explain rounding differences but do not establish their cause.',
            'No fitted beta, Newton estimate, candidate selection, promotion or new test/validation result.']}
    try: run(args,report)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()); raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started; native.write_json(args.report,report)
    print(json.dumps({'complete':True,'report':str(args.report),'sha256':native.sha(args.report),
        'ppl':{arm:value['ppl'] for arm,value in report['summaries'].items()},'teacher_kl':report['teacher_kl']}),flush=True)


if __name__=='__main__': main()
