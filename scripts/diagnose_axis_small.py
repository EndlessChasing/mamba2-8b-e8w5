#!/usr/bin/env python3
"""Descriptive current-axis small-tensor swap; no fitting or validation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mamba_e8w5 import axis_small_base as base
import evaluate_output_axis_residual as reused

prior_stage = reused.prior_stage
native = reused.native
CURRENT = 'current_old_small'
ORIGINAL = 'original_source_small'
REPEAT = 'current_old_small_repeat'
ARMS = (CURRENT,ORIGINAL,REPEAT)
HELPER_SHA = '4e7026475fc97b005105b27043a15bae83c576934612b1e346576438eef4da70'
REUSED_CODE = {
    'scripts/evaluate_output_axis_residual.py':'70e5f98e49f30ccc099d1ef78b4c82c96f188d10bb073a12a70f659cd48581f9',
    'scripts/evaluate_input_axis_residual.py':'c8dece7ec209b74ef65244086c7c245f08a59ea344124559f964e44c8a2540f0',
    'scripts/diagnose_vocab_w4.py':'c1dd8343a204923d6799f5b4eed31a6f0ffee1bc0774b2deaa15f34e0d54091d',
    'scripts/evaluate_teacher_kl_compensation.py':'b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071',
    'scripts/evaluate_small_compensation.py':'2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be',
}


def report_inputs(context):
    """Exclude tensor values and fitting windows from the diagnostic receipt."""
    return {'binding':context['binding'],'resolved_files':context['resolved_files'],
        'resolved507_by_arm':{CURRENT:context['current_expected507'],
            ORIGINAL:context['original_small_expected507'],REPEAT:context['current_expected507']},
        'fixed114_keys':list(context['fixed_large_keys']),
        'source393_cast_receipt':context['source393_cast_receipt'],
        'old393_receipt':context['old393_receipt'],
        'data_manifest_sha256':base.DATA_SHA,'dataset':context['data_manifest']['dataset'],
        'window_plan':context['heldout_plan'],
        'data_use':{'heldout_windows_scored':64,'targets_per_arm':131008,
                    'training_windows_integrity_checked_only':448,'training_tokens_scored':0}}


def run(args,report):
    if HELPER_SHA.startswith('PENDING'):
        raise ValueError('Shared helper must be reviewed and frozen before preflight')
    base.checked_file(ROOT/'mamba_e8w5/axis_small_base.py',HELPER_SHA)
    extra = {ROOT/name:digest for name,digest in REUSED_CODE.items()}
    extra[Path(__file__).resolve()] = base.runtime.sha256_file(__file__)
    for path,digest in extra.items():
        base.checked_file(path,digest)
    report['self_test'] = base.self_test()
    context = base.verify_inputs()
    report.update(report_inputs(context),reused_code_sha256=REUSED_CODE,helper_source_sha256=HELPER_SHA,
                  arms={},correctness_passed=False)
    # verify_data checks both prepared sets once; fitting windows are never scored.
    del context['training_windows']
    report['extra_checked_input_sha256'] = {str(path):digest for path,digest in extra.items()}
    native.write_json(args.report,report)
    if args.cpu_preflight:
        if torch.cuda.is_initialized():
            raise ValueError('CPU preflight must not initialize CUDA')
        report.update(complete=True,mode='cpu_preflight',cuda_initialized=False,
            source393_cast_verified=True,current117_actual_files_verified=True,
            no_hessian_payloads_read=True,no_superseded_projection_payloads_read=True)
        return
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
    report['environment'] = base.runtime.environment_receipt()
    report['reproducibility_before'] = prior_stage.reproducibility_snapshot()
    model = base.load_model(context)
    report['direct_current_load_audit'] = model._axis_load_audit
    report['package_receipt'] = model._package_receipt
    fixed_identity = base.large_identity(model)
    originals = {name:model.get_parameter(name) for name in base.small_inventory()}

    def restore():
        for name,value in originals.items():
            owner,leaf = name.rsplit('.',1)
            setattr(model.get_submodule(owner),leaf,value)

    try:
        for arm in ARMS:
            if arm == ORIGINAL:
                base.install_small(model,context['source_small'],context['source_expected507'])
            elif arm == REPEAT:
                restore()
            expected = context['original_small_expected507'] if arm==ORIGINAL else context['current_expected507']
            entry = report['arms'][arm] = {
                'installed_small_tensor_count':393,
                'actual_changed_small_keys_vs_current':[
                    name for name in base.small_inventory() if expected[name] != context['current_expected507'][name]],
                'fixed114_before':base.verify_fixed_large(model,context,fixed_identity)}
            prior_stage.score_arm(model,context['heldout_windows'],context['heldout_plan'],expected,
                entry,args.report,report,arm)
            entry['fixed114_after'] = base.verify_fixed_large(model,context,fixed_identity)
            native.write_json(args.report,report)
        report['baseline_repeat_exact'] = prior_stage.exact_repeat(report['arms'][CURRENT],report['arms'][REPEAT])
        if not report['baseline_repeat_exact']:
            raise ValueError('Complete64-window current-small baseline repeat differs')
        report['comparison'] = prior_stage.paired_comparison(report['arms'],(ORIGINAL,),CURRENT)
        historical = context['prior']['development']['axis_in_out_w4_w5']['summary']['ppl']
        actual = report['arms'][CURRENT]['summary']['ppl']
        report['historical_baseline_drift'] = {'prior_development_ppl':historical,'current_development_ppl':actual,
            'relative_ppl_change':actual/historical-1,'scope':'Descriptive cross-process drift; not a strict equality or selection gate.'}
        report['fixed114_verified'] = True
        report['correctness_passed'] = True
        report['complete'] = True
    finally:
        restore()
        report['final_current507_restored_audit'] = base.audit_model(model,context['current_expected507'])
        report['final_fixed114_restored_audit'] = base.verify_fixed_large(model,context,fixed_identity)
        report['final_input_recheck'] = base.final_recheck(context)
        for path,digest in extra.items():
            base.checked_file(path,digest)
        report['reproducibility_after'] = prior_stage.reproducibility_snapshot()
        report['gpu_memory'] = base.runtime.gpu_memory_receipt()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-preflight',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    args.report=(args.report or ROOT/'reports'/('axis_small_diagnostic_v1_cpu.json' if args.cpu_preflight else 'axis_small_diagnostic_v1.json')).resolve()
    if args.report.parent!=ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('Fresh report directly under reports/ is required')
    torch.set_num_threads(8)
    report={'format':'MAMBA2_AXIS_SMALL_DIAGNOSTIC_V1','complete':False,
        'script_sha256':base.runtime.sha256_file(__file__),'protocol_sha256':base.PROTOCOL_SHA,
        'arm_order':list(ARMS),'training_performed':False,'validation_loaded':False,
        'candidate_advancement':False,'initialization_selection_performed':False,
        'publication_performed':False,'limitations':[
            'These64 reserved TRAIN windows have been observed; this is descriptive development evidence.',
            'Diagnostic direction does not select training initialization or prove adaptation cannot help.',
            'Native FP16 evaluation does not establish compact runtime residency.']}
    started=time.monotonic()
    try:
        run(args,report)
    except BaseException as error:
        report.update(complete=False,correctness_passed=False,error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        native.write_json(args.report,report)
    print(json.dumps({'complete':report['complete'],'report':str(args.report),
        'correctness_passed':report.get('correctness_passed'),'comparison':report.get('comparison')}),flush=True)


if __name__=='__main__':main()
