#!/usr/bin/env python3
"""Bounded deterministic baseline traces; no adapter, fitting or quality gate.

Set process-start numerical settings before torch, Mamba, or repository imports.
Run this same frozen script in two fresh processes; reports retain exact traces
and explicitly reject any within-process repeat discrepancy.
"""
import os
import sys

if any(name == 'torch' or name.startswith(('torch.', 'mamba_ssm.', 'mamba_e8w5.')) for name in sys.modules):
    raise RuntimeError('Fresh process required before numerical/profile imports')
PROFILE_FROZEN = False  # Root must choose the exact common hard0/probe profile before launch.
TORCH_DETERMINISTIC = False
REMOVED_ENV = ('TRITON_CACHE_AUTOTUNING', 'TRITON_AUTOTUNE_BLOCK_SIZE_M',
    'TRITON_AUTOTUNE_BLOCK_SIZE_N', 'TRITON_AUTOTUNE_BLOCK_SIZE_K', 'TRITON_AUTOTUNE_BLOCK_SIZE_DSTATE')
PROFILE_NAMES = ('MAMBA_DETERMINISTIC', 'CUBLAS_WORKSPACE_CONFIG', *REMOVED_ENV)
ENV_BEFORE = {name:os.environ.get(name) for name in PROFILE_NAMES}
os.environ['MAMBA_DETERMINISTIC'] = '1'
for _name in REMOVED_ENV:
    os.environ.pop(_name, None)
if TORCH_DETERMINISTIC:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
import traceback

import torch
if TORCH_DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Always match the existing native reference precision; never enable autocast.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
FORMAT = 'MAMBA2_DETERMINISTIC_READAPTED_BASELINE_PROBE_V1'
EVALUATOR_SHA = '7d8ecefb8c28c4a74b9a9f3faed22831849a312cc65781f22404a3200d52c471'
HISTORY_SHA = 'a88f8fe75e82b992b8fc9cebeb6f681847e87aa36ea8bb1771edf979c47d551d'
DRIFT_SHA = '5bad9a6df90206c54aaa23380b66644cef101a2f86b5bab47c4b71bef0e015e2'
CHANGED = ('validation-n16-t0-s6', 'validation-n16-t0-s13-removed',
    'validation-n16-t2-s45', 'validation-n64-t0-s46-removed',
    'validation-n64-t1-s12-removed', 'validation-n64-t1-s18', 'validation-n64-t2-s23')
UNCHANGED = tuple(f'validation-n{n}-t{t}-s0' for n in (16,64) for t in range(3))
CASE_IDS = CHANGED + UNCHANGED
PPL_INDEXES = (0,1,2,3)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def profile():
    return {'version':'root-fixed deterministic readapted baseline probe v1',
        'MAMBA_DETERMINISTIC':os.environ.get('MAMBA_DETERMINISTIC'),
        'removed_environment':{name:os.environ.get(name) for name in REMOVED_ENV},
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'torch_deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision()}


def logit_receipt(logits, tensor_hash):
    if logits.dtype != torch.float16 or tuple(logits.shape) != (1,1,256000) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite native FP16 full-vocabulary head')
    # CPU only diagnostic math, no writes to the tensor consumed by greedy.
    cpu=logits.detach().cpu().contiguous().reshape(-1)
    floats=cpu.float();first=int(floats.argmax());masked=floats.clone();masked[first]=-torch.inf
    second=int(masked.argmax());ids=[first,second]
    bits=cpu.view(torch.int16)
    return {'dtype':'torch.float16','shape':[1,1,256000], 'full_logits_sha256':tensor_hash(cpu),
        'top2_ids_stable_low_id_ties':ids,'top2_fp16_bits_hex':[f'{int(bits[i])&65535:04x}' for i in ids],
        'top2_values_float32':[float(floats[i]) for i in ids],
        'top1_minus_top2_fp32':float(floats[first]-floats[second]),
        'argmax_tie_multiplicity':int((floats==floats[first]).sum())}


def cache_receipt(cache, tensor_hash):
    pairs=cache.key_value_memory_dict
    if set(pairs)!=set(range(56)):
        raise ValueError('Expected all56 cached layers')
    tensors={};count=0
    for layer in range(56):
        if len(pairs[layer])!=2:raise ValueError('Expected conv/SSM pair')
        for kind,value in zip(('conv','ssm'),pairs[layer]):
            if value.dtype!=torch.float16 or not torch.isfinite(value).all():
                raise ValueError('Invalid FP16 native cache')
            tensors[f'layer{layer}.{kind}']={'shape':list(value.shape),'dtype':'torch.float16','sha256':tensor_hash(value)}
            count+=value.numel()*value.element_size()
    if len(tensors)!=112 or count!=122028032:raise ValueError('Native cache capacity differs')
    return {'tensor_count':112,'bytes':count,'tensors':tensors}


@torch.inference_mode()
def trace_case(model,tokenizer,evaluation,mk,shared,case):
    hidden_steps=[];logits_steps=[];observed={}
    def backbone_hook(_module,_args,kwargs,output):
        if output.dtype!=torch.float16 or not torch.isfinite(output).all():raise ValueError('Invalid hidden')
        cache=kwargs.get('inference_params')
        if cache is None:raise ValueError('Frozen generation omitted cache')
        if not hidden_steps:
            if cache.seqlen_offset!=0 or output.shape[1]!=case['prompt_tokens']:
                raise ValueError('Fresh full prompt prefill required')
            observed['prefill_cache']=cache_receipt(cache,shared.tensor_hash)
            observed['cache']=cache
        elif cache is not observed['cache']:raise ValueError('Cache object changed')
        hidden_steps.append({'backbone_call':len(hidden_steps), 'seqlen_offset':cache.seqlen_offset,
            'last_position_hidden_sha256':shared.tensor_hash(output[:,-1:])})
    def head_hook(_module,_args,output):
        logits_steps.append({'generated_offset_zero_based':len(logits_steps),**logit_receipt(output,shared.tensor_hash)})
    handles=[model.backbone.register_forward_hook(backbone_hook,with_kwargs=True),model.lm_head.register_forward_hook(head_hook)]
    try:
        with torch.autocast(device_type='cuda',enabled=False):
            text,ids,count,cache_bytes=evaluation.generate_greedy(model,tokenizer,case['prompt'],max_new_tokens=12,execution='prefill')
        if count!=case['prompt_tokens'] or len(ids)!=len(logits_steps) or not 1<=len(ids)<=12:
            raise ValueError('Trace/generation coverage differs')
        if [x['top2_ids_stable_low_id_ties'][0] for x in logits_steps]!=ids:
            raise ValueError('Passive logit trace differs from frozen greedy argmax')
        final=cache_receipt(observed['cache'],shared.tensor_hash)
        if final['bytes']!=cache_bytes:raise ValueError('Cache byte accounting differs')
        prediction=mk.predict(text)
        return {**{k:v for k,v in case.items() if k!='prompt'},'output':text,'generated_ids':ids,
            'prediction':prediction,'correct':prediction==case['answer'],
            'prefill_last_position_hidden_sha256':hidden_steps[0]['last_position_hidden_sha256'],
            'backbone_hidden_steps':hidden_steps,'prefill_cache':observed['prefill_cache'],
            'final_cache':final,'logit_steps':logits_steps,
            'trace_semantics':'Passive frozen-greedy hooks; final state includes the frozen loop final generated-token backbone call unless EOS.'}
    finally:
        for handle in handles:handle.remove()
        observed.clear()


@torch.inference_mode()
def trace_prose(model,scorer,shared,window,plan):
    states=[]
    def hook(_module,_args,output):
        if output.dtype!=torch.float16 or not torch.isfinite(output).all():raise ValueError('Invalid prose hidden')
        states.append({'shape':list(output.shape),'full_hidden_sha256':shared.tensor_hash(output),
            'last_position_hidden_sha256':shared.tensor_hash(output[:,-1:])})
    handle=model.backbone.register_forward_hook(hook)
    try:row=scorer.frozen.score_window(model,window,chunk_tokens=64)
    finally:handle.remove()
    scorer.check_row(row,window,plan)
    if len(states)!=1:raise ValueError('Expected one native cache-free prose backbone call')
    return {**row,**plan,**states[0]}


def snapshots(reproducibility):
    result=reproducibility.reproducibility_snapshot()
    module=sys.modules.get('mamba_ssm.utils.determinism')
    result['mamba_determinism_utility']={'loaded':module is not None,
        'use_deterministic_mode':module.use_deterministic_mode() if module is not None else None,
        'file':str(Path(module.__file__).resolve()) if module is not None else None,
        'sha256':sha(module.__file__) if module is not None else None}
    result['installed_source_sha256']={}
    for name in ('mamba_ssm.modules.mamba2','mamba_ssm.ops.triton.selective_state_update',
                 'mamba_ssm.ops.triton.ssd_chunk_scan','mamba_ssm.ops.triton.ssd_chunk_state',
                 'mamba_ssm.ops.triton.ssd_bmm','mamba_ssm.ops.triton.ssd_state_passing','triton.runtime.autotuner'):
        module=sys.modules.get(name)
        if module is not None and getattr(module,'__file__',None):
            result['installed_source_sha256'][name]={'file':module.__file__,'sha256':sha(module.__file__)}
    return result


def prepare(args,report):
    if not PROFILE_FROZEN:raise ValueError('Root numerical profile selection is pending; fail closed')
    if sha(ROOT/'scripts/evaluate_resurface_readapted.py')!=EVALUATOR_SHA:raise ValueError('Frozen loader/scorer changed')
    frozen=importlib.import_module('evaluate_resurface_readapted');checked={}
    for name,digest in frozen.CODE_PINS.items():frozen.bind(ROOT/name,digest,checked)
    frozen.bind(ROOT/'scripts/evaluate_resurface_readapted.py',EVALUATOR_SHA,checked)
    frozen.bind(__file__,sha(__file__),checked)
    shared=importlib.import_module('mamba_e8w5.axis_small_base');scorer=importlib.import_module('evaluate_input_axis_residual')
    for pins in (shared.CODE,scorer.frozen.CODE):
        for name,digest in pins.items():
            if Path(name).suffix=='.py':frozen.bind(ROOT/name,digest,checked)
    context,values,expected,prior=frozen.prepare_current(args,shared,checked)
    old=frozen.read_bound(ROOT/'reports/readapted_mk_baseline_v1.json',HISTORY_SHA,checked)
    drift=frozen.read_bound(ROOT/'reports/resurface_mk_baseline_drift_provisional.json',DRIFT_SHA,checked)
    if tuple(row['id'] for row in drift['changed_rows'])!=CHANGED:
        raise ValueError('Fixed drift case list differs')
    mk=importlib.import_module('evaluate_readapted_mk');evaluation=importlib.import_module('mamba_e8w5.evaluation')
    tokenizer=shared.runtime.SentencePieceTokenizer(ROOT/'models/source')
    cases,all_plan=mk.prepare_cases(evaluation,tokenizer)
    if cases!=old['cases']:raise ValueError('Actual historical768 prompt/token plan differs')
    by_id={row['id']:row for row in cases}
    if len(set(CASE_IDS))!=13 or any(key in CHANGED for key in UNCHANGED):raise ValueError('Case selection overlap')
    chosen=[by_id[key] for key in CASE_IDS]
    path=ROOT/'training_data/teacher_kl_compensation_v1/heldout_tokens.pt'
    frozen.bind(path,'a3fc0803052890d973b74f91581a6bfc1d2f93ed28db5725f23c16eb8ec20546',checked)
    windows=torch.load(path,map_location='cpu',weights_only=True)
    if windows.dtype!=torch.long or tuple(windows.shape)!=(64,2048):raise ValueError('Prose data coverage differs')
    plan=[prior['development_plan'][i] for i in PPL_INDEXES]
    selected=[windows[i] for i in PPL_INDEXES]
    if any(evaluation.token_digest(window.tolist())!=row['token_sha256_int64le'] for window,row in zip(selected,plan)):
        raise ValueError('Fixed first-four prose identity differs')
    del windows
    selection={'mk_ids':list(CASE_IDS),'mk_plan':[{k:v for k,v in row.items() if k!='prompt'} for row in chosen],
        'prose_indexes':list(PPL_INDEXES),'prose_plan':plan,'repeats_per_item':2,
        'order':'13MK cases in listed order, adjacent fresh-cache repeats; then first4prose windows, adjacent repeats',
        'max_generated_tokens':12,'adapter_enabled':False,'historical_comparison_is_descriptive':True}
    selection['sha256_canonical_json']=hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    report.update(selection=selection,checked_input_sha256=checked,profile=profile(),
        current_expected507=expected,scope='Bounded numerical baseline probe only; no MK/PPL quality gate, adapter or validation split.')
    return frozen,shared,scorer,mk,evaluation,tokenizer,context,values,expected,chosen,selected,plan,checked


def self_test():
    assert len(CASE_IDS)==len(set(CASE_IDS))==13 and PPL_INDEXES==(0,1,2,3)
    cpu=torch.full((1,1,256000),-3.,dtype=torch.float16);cpu[0,0,11]=2.;cpu[0,0,17]=2.
    def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    row=logit_receipt(cpu,digest)
    assert row['top2_ids_stable_low_id_ties']==[11,17] and row['top1_minus_top2_fp32']==0. and row['argmax_tie_multiplicity']==2
    assert row['top2_fp16_bits_hex']==['4000','4000']
    assert not torch.cuda.is_initialized()
    return {'passed':True,'checks':['Fixed13/4 bounded selection','NativeFP16 bit representation and deterministic low-ID top2/tie margin','CUDA uninitialized']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--run-label',required=True);p.add_argument('--verify-only',action='store_true')
    args=p.parse_args();args.report=args.report.resolve();torch.set_num_threads(8)
    if args.report.parent!=ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():
        raise ValueError('Fresh report under reports required')
    if args.verify_only and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preflight must hideCUDA')
    report={'format':FORMAT,'complete':False,'run_label':args.run_label,'script_sha256':sha(__file__),
        'profile_environment_before':ENV_BEFORE,'profile_configured_before_torch_import':True,
        'quality_measured':False,'quality_gate_changed':False,'adapter_enabled':False,'fitting_performed':False}
    started=time.monotonic();model=None;checked={}
    try:
        report['self_test']=self_test()
        frozen,shared,scorer,mk,evaluation,tokenizer,context,values,expected,cases,windows,plan,checked=prepare(args,report)
        if args.verify_only:
            if torch.cuda.is_initialized():raise ValueError('CPU preflight initializedCUDA')
            report.update(complete=True,mode='verify_only',cuda_initialized=False)
            return
        torch.cuda.reset_peak_memory_stats();report['environment']=shared.runtime.environment_receipt()
        model=shared.load_model(context).eval().requires_grad_(False)
        report['direct_old393_load_audit']=model._axis_load_audit
        report['installed_readapted393_audit']=shared.install_small(model,values,expected)
        identities=mk.identities(model);report['model507_before']=shared.audit_model(model,expected)
        reproduction=importlib.import_module('diagnose_output_calibration_v2')
        report['numerical_before']=snapshots(reproduction)
        if report['numerical_before']['mamba_determinism_utility']['use_deterministic_mode'] is not True:
            raise ValueError('Installed Mamba deterministic mode is not enabled')
        report['mk']=[];report['prose']=[];write(args.report,report)
        for case in cases:
            first=trace_case(model,tokenizer,evaluation,mk,shared,case)
            second=trace_case(model,tokenizer,evaluation,mk,shared,case)
            result={'id':case['id'],'first':first,'second':second,'exact_equal':first==second}
            report['mk'].append(result);write(args.report,report)
            if not result['exact_equal']:raise RuntimeError(f'Within-process native trace differs:{case["id"]}')
            print(f'[{args.run_label}] MK {len(report["mk"])}/13 exact',flush=True)
        for index,window,entry in zip(PPL_INDEXES,windows,plan):
            first=trace_prose(model,scorer,shared,window,entry);second=trace_prose(model,scorer,shared,window,entry)
            result={'index':index,'first':first,'second':second,'exact_equal':first==second}
            report['prose'].append(result);write(args.report,report)
            if not result['exact_equal']:raise RuntimeError(f'Within-process prose trace differs:{index}')
            print(f'[{args.run_label}] prose {len(report["prose"])}/4 exact',flush=True)
        report['model507_after']=shared.audit_model(model,expected)
        report['model_identity_before']=identities;report['model_identity_after']=mk.identities(model)
        if identities!=report['model_identity_after']:raise ValueError('Frozen model identity mutated')
        report['numerical_after']=snapshots(reproduction)
        report['all_same_process_repeats_exact']=True
        report.update(complete=True,mode='bounded_native_trace')
    except BaseException as error:
        report.update(complete=False,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if model is not None:
            report['final_model507_audit']=shared.audit_model(model,expected)
            report['gpu_memory']=shared.runtime.gpu_memory_receipt()
            del model;gc.collect()
        for path,digest in checked.items():
            if sha(path)!=digest:raise ValueError(f'Input changed:{path}')
        report['final_bound_input_count']=len(checked)
        report['elapsed_seconds']=time.monotonic()-started;write(args.report,report)


if __name__=='__main__':main()
