"""Paired native-output projection moments on a frozen candidate prefix.

Teacher targets are the actual Linear outputs, including native FP16 rounding.
H and K are uncentered moments, never centered covariances. No projection weight,
small tensor, inference cache or model training mode is modified by this module.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F

DEFAULT_SIX=tuple(f'layer{layer}.{part}' for layer in (0,18,55) for part in ('in_proj','out_proj'))


def projection_path(label):
    match=re.fullmatch(r'layer(0|[1-9][0-9]*)\.(in_proj|out_proj)',label)
    if not match:raise ValueError(f'Invalid projection label: {label!r}')
    return f'backbone.layers.{int(match[1])}.mixer.{match[2]}'


def _labels(selected_names):
    names=tuple(selected_names)
    if not names or len(names)!=len(set(names)):raise ValueError('Select at least one unique projection')
    for name in names:projection_path(name)
    return names


def _batches(windows):
    sequence=[windows] if isinstance(windows,torch.Tensor) and windows.ndim==1 else windows
    for batch in sequence:
        if not isinstance(batch,torch.Tensor) or batch.dtype!=torch.int64 or batch.ndim not in (1,2):
            raise ValueError('Each batch must be an int64 token tensor of shape[length] or[batch,length]')
        if batch.ndim==1:batch=batch[None]
        if min(batch.shape)<=0:raise ValueError('Empty token batch')
        yield batch


def _token_hash(batch):
    return hashlib.sha256(batch.detach().cpu().contiguous().numpy().astype('<i8',copy=False).tobytes()).hexdigest()


def _energy(value):
    # FP32 products, FP64 reduction/accumulation; no full FP64 activation copy.
    return value.float().square().sum(dtype=torch.float64)


def _metadata(model,names,require_fp16):
    if model.training:raise ValueError('Caller must set both runtime models to eval mode')
    modules={name:model.get_submodule(projection_path(name)) for name in names}
    metadata={}
    for name,module in modules.items():
        if not isinstance(module,torch.nn.Linear) or module.bias is not None:
            raise ValueError(f'Requires a native bias-free Linear: {name}')
        if require_fp16 and module.weight.dtype!=torch.float16:raise ValueError(f'Projection is not FP16: {name}')
        mixer=model.get_submodule(projection_path(name).rsplit('.',1)[0])
        if getattr(mixer,'use_mem_eff_path',False):raise ValueError('Fused projection path bypasses Linear hooks')
        metadata[name]={'shape':list(module.weight.shape),'weight_dtype':str(module.weight.dtype),
                        'device':str(module.weight.device)}
    devices={module.weight.device for module in modules.values()}
    if len(devices)!=1:raise ValueError('Selected projections must be on one device')
    device=next(iter(devices))
    if device.type=='cuda' and (torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest'):
        raise ValueError('Caller must disable TF32 and set float32 matmul precision to highest')
    return modules,metadata,device


def _weight_identity(modules):
    return {name:(id(module.weight),module.weight.data_ptr(),module.weight._version,
                  tuple(module.weight.shape),tuple(module.weight.stride()),module.weight.dtype)
            for name,module in modules.items()}


def _paired_forward(teacher,candidate,windows,names,consume,*,split_label,require_fp16):
    if not isinstance(split_label,str) or not split_label.strip():raise ValueError('Explicit split_label is required')
    teacher_modules,teacher_meta,device=_metadata(teacher,names,require_fp16)
    candidate_modules,candidate_meta,other_device=_metadata(candidate,names,require_fp16)
    if device!=other_device:raise ValueError('Teacher and candidate must use the same device')
    if any(teacher_meta[n]['shape']!=candidate_meta[n]['shape'] for n in names):raise ValueError('Paired projection shapes differ')
    before_teacher,before_candidate=_weight_identity(teacher_modules),_weight_identity(candidate_modules)
    phase=None;cache={};teacher_seen={};candidate_seen={};handles=[];batch_receipts=[]
    retained_peak_bytes=0

    def validate(name,args,output):
        if len(args)!=1 or not isinstance(args[0],torch.Tensor) or not isinstance(output,torch.Tensor):
            raise ValueError(f'Expected plain Linear tensor call: {name}')
        x=args[0]
        m,n=teacher_meta[name]['shape']
        if x.ndim!=3 or output.ndim!=3 or x.shape[:-1]!=output.shape[:-1] or x.shape[-1]!=n or output.shape[-1]!=m:
            raise ValueError(f'Unexpected native activation shape: {name}')
        if require_fp16 and (x.dtype!=torch.float16 or output.dtype!=torch.float16):raise ValueError('Native projection activation is not FP16')
        if not torch.isfinite(x).all() or not torch.isfinite(output).all():raise FloatingPointError(f'Nonfinite native projection: {name}')
        return x

    def teacher_hook(name):
        def hook(_module,args,output):
            nonlocal retained_peak_bytes
            if phase!='teacher':return
            teacher_seen[name]=teacher_seen.get(name,0)+1
            if teacher_seen[name]!=1:raise ValueError(f'Teacher projection called more than once per batch: {name}')
            x=validate(name,args,output)
            # Clone protects the captured native output against later view reuse.
            cache[name]={'target':output.detach().clone(),'teacher_input_energy':_energy(x),
                         'teacher_input_shape':list(x.shape),'teacher_output_shape':list(output.shape),
                         'teacher_input_dtype':str(x.dtype),'teacher_output_dtype':str(output.dtype)}
            retained_peak_bytes=max(retained_peak_bytes,sum(item['target'].numel()*item['target'].element_size() for item in cache.values()))
        return hook

    def candidate_hook(name):
        def hook(_module,args,output):
            if phase!='candidate':return
            candidate_seen[name]=candidate_seen.get(name,0)+1
            if candidate_seen[name]!=1 or name not in cache:raise ValueError(f'Unpaired candidate projection: {name}')
            x=validate(name,args,output)
            record=cache.pop(name)
            if record['target'].shape!=output.shape:raise ValueError('Teacher/candidate token axes differ')
            consume(name,x.detach(),output.detach(),record)
        return hook

    try:
        for name in names:
            handles.append(teacher_modules[name].register_forward_hook(teacher_hook(name)))
            handles.append(candidate_modules[name].register_forward_hook(candidate_hook(name)))
        with torch.no_grad():
            for index,batch in enumerate(_batches(windows)):
                if cache:raise RuntimeError('Unconsumed teacher activation from previous batch')
                teacher_seen.clear();candidate_seen.clear()
                receipt={'batch_index':index,'shape':list(batch.shape),'tokens':batch.numel(),'tokens_sha256_int64le':_token_hash(batch)}
                ids=batch.to(device=device)
                phase='teacher';teacher.backbone(ids)
                if set(teacher_seen)!=set(names):raise ValueError('Teacher hooks did not cover every selected projection')
                phase='candidate';candidate.backbone(ids)
                if set(candidate_seen)!=set(names) or cache:raise ValueError('Candidate hooks did not consume every paired output')
                phase=None;batch_receipts.append(receipt)
        if not batch_receipts:raise ValueError('No token batches were supplied')
        if before_teacher!=_weight_identity(teacher_modules) or before_candidate!=_weight_identity(candidate_modules):
            raise RuntimeError('Projection parameter identity/version changed during collection')
    finally:
        phase=None;cache.clear()
        for handle in handles:handle.remove()
    return {'split_label':split_label,'selected_names':list(names),'batches':batch_receipts,
            'tokens':sum(row['tokens'] for row in batch_receipts),'teacher':teacher_meta,'candidate':candidate_meta,
            'teacher_target':'actual native projection output; includes FP16 output rounding',
            'prefix':'unchanged candidate model; replacements are never installed',
            'zero_state_each_batch':True,'require_fp16':require_fp16,
            'peak_retained_teacher_output_bytes':retained_peak_bytes,
            'parameter_identity_and_version_unchanged':True,
            'statistics_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32)}


@torch.no_grad()
def collect_projection_statistics(teacher,candidate,windows,*,selected_names=DEFAULT_SIX,split_label,require_fp16=True):
    """Return normalized FP32 H[n,n], K[m,n] and per-token native energies.

    Each iterable item is one independent token window or batch of windows.
    Tensor[windows,length] is iterated as separate single-window batches.
    Only current-batch teacher outputs are retained; H/K stay on the model device.
    Call separately for fit and projection-fit holdout; no states/statistics carry.
    """
    names=_labels(selected_names);accum={}
    def consume(name,x,output,record):
        values=x.reshape(-1,x.shape[-1]).float()
        targets=record['target'].reshape(-1,output.shape[-1]).float()
        actual=output.reshape_as(targets).float()
        count,n=values.shape;m=targets.shape[1]
        if name not in accum:
            accum[name]={'H':torch.zeros(n,n,device=x.device,dtype=torch.float32),
                'K':torch.zeros(m,n,device=x.device,dtype=torch.float32),'tokens':0,
                'teacher_energy_per_output':torch.zeros(m,device=x.device,dtype=torch.float64),
                'candidate_native_error_per_output':torch.zeros(m,device=x.device,dtype=torch.float64),
                'teacher_input_energy':torch.zeros((),device=x.device,dtype=torch.float64),
                'candidate_input_energy':torch.zeros((),device=x.device,dtype=torch.float64),
                'candidate_native_energy':torch.zeros((),device=x.device,dtype=torch.float64),
                'shape':[m,n],'teacher_input_dtype':record['teacher_input_dtype'],
                'teacher_output_dtype':record['teacher_output_dtype'],'candidate_input_dtype':str(x.dtype),
                'candidate_output_dtype':str(output.dtype)}
        state=accum[name]
        state['H'].addmm_(values.T,values)
        state['K'].addmm_(targets.T,values)
        state['tokens']+=count
        state['teacher_energy_per_output']+=targets.square().sum(0,dtype=torch.float64)
        state['candidate_native_error_per_output']+=(actual-targets).square().sum(0,dtype=torch.float64)
        state['teacher_input_energy']+=record['teacher_input_energy']
        state['candidate_input_energy']+=_energy(values)
        state['candidate_native_energy']+=_energy(actual)
    receipt=_paired_forward(teacher,candidate,windows,names,consume,split_label=split_label,require_fp16=require_fp16)
    for name,state in accum.items():
        count=state['tokens']
        if count!=receipt['tokens']:raise RuntimeError(f'Projection token count differs: {name}')
        state['H'].div_(count);state['K'].div_(count)
        if not torch.isfinite(state['H']).all() or not torch.isfinite(state['K']).all():raise FloatingPointError('Nonfinite FP32 moment accumulation')
        state['teacher_energy_per_output'].div_(count);state['candidate_native_error_per_output'].div_(count)
        state['teacher_energy']=float(state['teacher_energy_per_output'].sum())
        state['candidate_native_mse_per_token']=float(state['candidate_native_error_per_output'].sum())
        state['candidate_native_mse_per_output_element']=state['candidate_native_mse_per_token']/state['shape'][0]
        for key in ('teacher_input_energy','candidate_input_energy','candidate_native_energy'):
            state[key]=float(state[key]/count)
        state['H_frobenius_norm']=float(torch.linalg.vector_norm(state['H']))
        state['K_frobenius_norm']=float(torch.linalg.vector_norm(state['K']))
        state['H_asymmetry_max_abs']=float((state['H']-state['H'].T).abs().max())
        state['normalization']='sum over matched token positions divided by tokens; output coordinates are summed'
        state['H_formula']='X_candidate.T @ X_candidate / N (uncentered)'
        state['K_formula']='Y_teacher_native.T @ X_candidate / N (uncentered)'
    return {**receipt,'format':'MAMBA2_PAIRED_PROJECTION_STATISTICS_V1','matrices':accum,
            'moment_dtype':'float32','scalar_energy_accumulation_dtype':'float64'}


@torch.no_grad()
def score_dense_from_statistics(weight,matrix_stats,*,quadratic_dtype=torch.float64):
    """Quadratic of REAL-linear W@X, not native FP16 matmul output rounding.

    FP64 scoring reduces cancellation but H/K were accumulated/stored in FP32.
    A small negative raw result is reported and clamped only for displayed MSE;
    a materially negative result is rejected as inconsistent moments.
    """
    h,k=matrix_stats['H'],matrix_stats['K']
    if list(weight.shape)!=matrix_stats['shape'] or k.shape!=weight.shape or h.shape!=(weight.shape[1],weight.shape[1]):
        raise ValueError('Weight/statistic shape mismatch')
    if quadratic_dtype not in (torch.float32,torch.float64):raise ValueError('Quadratic dtype must be FP32 or FP64')
    w=weight.to(device=h.device,dtype=quadratic_dtype)
    hd=h.to(quadratic_dtype);kd=k.to(quadratic_dtype)
    if not all(bool(torch.isfinite(t).all()) for t in (w,hd,kd)):raise FloatingPointError('Nonfinite quadratic input')
    energy=float(matrix_stats['teacher_energy'])
    raw=float(((w@hd)*w).sum(dtype=torch.float64)-2*(w*kd).sum(dtype=torch.float64)+energy)
    tolerance=1e-5*max(1.,abs(energy))
    if not math.isfinite(raw) or raw < -tolerance:raise FloatingPointError('Invalid/ materially negative quadratic')
    mse=max(raw,0.)
    return {'quadratic_mse_per_token_raw':raw,'mse_per_token':mse,
            'mse_per_output_element':mse/weight.shape[0],
            'relative_mse':mse/energy if energy>0 else None,
            'relative_output_mse':mse/energy if energy>0 else None,
            'mean_output_squared_error':mse/weight.shape[0],'teacher_energy':energy,'tokens':matrix_stats['tokens'],
            'negative_roundoff_clamped':raw<0,'negative_roundoff_tolerance':tolerance,
            'quadratic_dtype':str(quadratic_dtype),'moment_dtype':str(h.dtype),
            'prediction_semantics':'real-linear output on candidate inputs; does not reproduce native FP16 output rounding'}


@torch.no_grad()
def score_native_fp16(teacher,candidate,windows,replacements,*,split_label):
    """Score independent FP16 replacement matrices on an unchanged prefix.

    Each supplied weight is tested on that projection's CURRENT candidate input.
    Later inputs do not see earlier replacements. This is not an installed six-
    matrix model evaluation; callers must measure that separately.
    """
    names=_labels(replacements.keys());prepared={};accum={}
    for name,weight in replacements.items():
        module=candidate.get_submodule(projection_path(name))
        if not isinstance(weight,torch.Tensor) or weight.dtype!=torch.float16 or weight.shape!=module.weight.shape:
            raise ValueError(f'Expected an already-rounded FP16 replacement: {name}')
        if not torch.isfinite(weight).all():raise FloatingPointError('Nonfinite replacement weight')
        prepared[name]=weight.detach().to(module.weight.device)
    def consume(name,x,output,record):
        target=record['target'].float()
        # This counterfactual never replaces the module's actual output.
        predicted=F.linear(x,prepared[name])
        if predicted.dtype!=torch.float16 or not torch.isfinite(predicted).all():raise FloatingPointError('Invalid native replacement output')
        if name not in accum:
            accum[name]={'tokens':0,'shape':list(prepared[name].shape),'teacher_energy_sum':torch.zeros((),device=x.device,dtype=torch.float64),
                'squared_error_sum':torch.zeros((),device=x.device,dtype=torch.float64),
                'baseline_squared_error_sum':torch.zeros((),device=x.device,dtype=torch.float64),
                'squared_error_per_output_sum':torch.zeros(target.shape[-1],device=x.device,dtype=torch.float64),
                'baseline_error_per_output_sum':torch.zeros(target.shape[-1],device=x.device,dtype=torch.float64),
                'teacher_energy_per_output_sum':torch.zeros(target.shape[-1],device=x.device,dtype=torch.float64)}
        state=accum[name];difference=predicted.float()-target
        state['tokens']+=x.numel()//x.shape[-1]
        state['teacher_energy_sum']+=_energy(target)
        state['squared_error_sum']+=_energy(difference)
        state['baseline_squared_error_sum']+=_energy(output.float()-target)
        state['squared_error_per_output_sum']+=difference.reshape(-1,difference.shape[-1]).square().sum(0,dtype=torch.float64)
        state['baseline_error_per_output_sum']+=(output.float()-target).reshape(-1,target.shape[-1]).square().sum(0,dtype=torch.float64)
        state['teacher_energy_per_output_sum']+=target.reshape(-1,target.shape[-1]).square().sum(0,dtype=torch.float64)
    receipt=_paired_forward(teacher,candidate,windows,names,consume,split_label=split_label,require_fp16=True)
    scores={}
    for name,state in accum.items():
        n=state['tokens']
        if n!=receipt['tokens']:raise RuntimeError('Native score token count mismatch')
        energy=float(state['teacher_energy_sum']/n);mse=float(state['squared_error_sum']/n)
        baseline=float(state['baseline_squared_error_sum']/n)
        scores[name]={'tokens':n,'shape':state['shape'],'teacher_energy':energy,'mse_per_token':mse,
            'mse_per_output_element':mse/state['shape'][0],'relative_mse':mse/energy if energy>0 else None,
            'relative_output_mse':mse/energy if energy>0 else None,'mean_output_squared_error':mse/state['shape'][0],
            'baseline_native_mse_per_token':baseline,'relative_reduction_vs_baseline':1-mse/baseline if baseline>0 else None,
            'mse_per_output':(state['squared_error_per_output_sum']/n).cpu().tolist(),
            'prediction_semantics':'actual FP16 F.linear output on unchanged candidate prefix'}
        if name.endswith('.in_proj') and state['shape'][0]==18560:
            partitions={}
            for role,start,end in (('z',0,8192),('x',8192,16384),('B',16384,17408),('C',17408,18432),('dt',18432,18560)):
                error=float(state['squared_error_per_output_sum'][start:end].sum()/n)
                target_energy=float(state['teacher_energy_per_output_sum'][start:end].sum()/n)
                original_error=float(state['baseline_error_per_output_sum'][start:end].sum()/n)
                partitions[role]={'rows':[start,end],'mse_per_token':error,'mean_output_squared_error':error/(end-start),
                    'mse_per_output_element':error/(end-start),'teacher_energy':target_energy,
                    'relative_output_mse':error/target_energy if target_energy>0 else None,
                    'baseline_native_mse_per_token':original_error,
                    'relative_reduction_vs_baseline':1-error/original_error if original_error>0 else None}
            scores[name]['partitions']=partitions
    return {**receipt,'format':'MAMBA2_NATIVE_PROJECTION_SCORE_V1','matrices':scores,
            'replacements_installed':False,'combined_model_quality_evaluated':False}
