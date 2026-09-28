#!/usr/bin/env python3
"""Discarded decoder timing/parity probe; no model, optimizer or candidate writes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba_e8w5 import codec
from mamba_e8w5.learned_e8_codebook import decode_e8_with_prototypes

PARENT_SHA='ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
PINS={'mamba_e8w5/codec.py':'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
      'mamba_e8w5/learned_e8_codebook.py':'b244b93dabb9a85487f4745588e7c0dd922274768223da94bf368aecfe10843f'}
LABELS=('layer0.in_proj','layer0.out_proj')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    h=hashlib.sha256();flat=value.detach().reshape(-1)
    for first in range(0,flat.numel(),16<<20):
        h.update(flat[first:first+(16<<20)].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_report(path,report):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def explicit_mm_rotate_last(x,inverse=False):
    """Algebraically equivalent rotations; bitwise equivalence is NOT assumed.

    Flatten the Hadamard rows into one2D GEMM. Apply the DCT to a2D packed
    [k,batch*p] matrix, then restore the original contiguous output layout.
    The same frozen FP32 factors and multiply order (Hadamard then DCT) are used.
    """
    c,h=codec.rotation_factors(x.shape[-1],str(x.device))
    k,p=c.shape[0],h.shape[0]
    original_shape=x.shape
    grouped=x.reshape(-1,k,p)
    first=grouped.reshape(-1,p).mm(h).reshape(-1,k,p)
    packed=first.permute(1,0,2).reshape(k,-1)
    second=(c.T if inverse else c).mm(packed)
    return second.reshape(k,grouped.shape[0],p).permute(1,0,2).contiguous().reshape(original_shape)


@contextmanager
def rotation_override(mode):
    original=codec.rotate_last
    if mode=='explicit_mm':codec.rotate_last=explicit_mm_rotate_last
    elif mode!='frozen':raise ValueError(mode)
    try:yield
    finally:codec.rotate_last=original


def timed_trial(payload,cb,initial,cotangent,mode):
    master=initial.detach().clone().requires_grad_(True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    first=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
    started=time.perf_counter()
    with rotation_override(mode):
        first.record()
        weight=decode_e8_with_prototypes(payload,cb,master.half(),device='cuda')
        middle.record();middle.synchronize()
        forward_seconds=time.perf_counter()-started
        backward_started=time.perf_counter()
        gradient,=torch.autograd.grad(weight,master,grad_outputs=cotangent)
        last.record();last.synchronize()
        backward_seconds=time.perf_counter()-backward_started
    if not torch.isfinite(weight).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite probe output/gradient')
    if gradient.dtype!=torch.float32 or not torch.count_nonzero(gradient):
        raise RuntimeError('Missing finite nonzero FP32 table derivative')
    result={'forward_wall_seconds':forward_seconds,'backward_wall_seconds':backward_seconds,
        'total_wall_seconds':forward_seconds+backward_seconds,
        'forward_cuda_event_ms':first.elapsed_time(middle),'backward_cuda_event_ms':middle.elapsed_time(last),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'weight_sha256':tensor_sha(weight),'gradient_sha256':tensor_sha(gradient),
        'gradient_norm':float(gradient.double().norm()),'gradient_nonzero':int(torch.count_nonzero(gradient))}
    return result,weight.detach(),gradient.detach()


def parity(weight,gradient,reference_weight,reference_gradient):
    bits=weight.view(torch.int16)!=reference_weight.view(torch.int16)
    difference=(gradient-reference_gradient).double()
    norm=float(reference_gradient.double().norm())
    weight_difference=(weight.float()-reference_weight.float()).abs()
    return {'decoded_fp16_bitwise_equal':not bool(bits.any()),'decoded_fp16_mismatched_bits_elements':int(bits.sum()),
        'decoded_fp16_elements':weight.numel(),'decoded_fp16_max_abs_difference':float(weight_difference.max()),
        'table_gradient_bitwise_equal':tensor_sha(gradient)==tensor_sha(reference_gradient),
        'table_gradient_max_abs_difference':float(difference.abs().max()),
        'table_gradient_relative_l2_difference':float(difference.norm())/norm if norm>0 else None,
        'gradient_allclose_rtol_1e_3_atol_1e_5':bool(torch.allclose(gradient,reference_gradient,rtol=1e-3,atol=1e-5)),
        'gradient_tolerance_is_diagnostic_only':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir',type=Path,default=ROOT/'artifacts/e8w5_v1')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--include-cuda-payloads',action='store_true')
    args=parser.parse_args();args.parent_dir=args.parent_dir.resolve();args.report=args.report.resolve()
    if args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():raise FileExistsError(args.report)
    if args.report==args.parent_dir or args.parent_dir in args.report.parents:
        raise ValueError('Probe report cannot modify the immutable parent directory')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    bound={ROOT/name:digest for name,digest in PINS.items()}
    bound[args.parent_dir/'manifest.json']=PARENT_SHA
    for path,digest in bound.items():
        if sha(path)!=digest:raise ValueError(f'Frozen input differs: {path}')
    manifest=json.loads((args.parent_dir/'manifest.json').read_text())
    for label in LABELS:
        path=args.parent_dir/f'{label}.e8';entry=manifest['files'][path.name]
        if path.is_symlink() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise ValueError(f'Parent matrix identity differs: {label}')
        bound[path]=entry['sha256']
    started=time.perf_counter()
    report={'format':'MAMBA2_PROTOTYPE_DECODER_PERFORMANCE_PROBE_V1','complete':False,
        'script_sha256':sha(__file__),'bound_input_sha256':{str(p):d for p,d in bound.items()},
        'scope':'Two fixed original projections only; discarded timing/parity; no model/optimizer/training/candidate modification',
        'training_updates':0,'optimizer_created':False,'labels':list(LABELS),'trials':[],
        'repetitions':'First invocation then2 warm repetitions per case; factor caches persist across cases, first is not globally cold',
        'timing':'Wall spans include validation/copies; CUDA-event spans may include device idle time waiting for host dispatch',
        'cotangent':'Dense FP16 rowsign*(((column%17)-8)/1024); no language/model inputs',
        'nonzero_table':'FP32 linspace(-0.002,0.002,2048).reshape(256,8), rounded half by exact decoder entrypoint',
        'candidate_promotion_performed':False}
    write_report(args.report,report)
    original_rotation=codec.rotate_last
    try:
        torch.set_num_threads(8);torch.manual_seed(20260927);torch.cuda.manual_seed_all(20260927)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(),'threads':torch.get_num_threads(),'tf32_matmul':False}
        cb,_=codec.load_reference_primitives();cb=cb.cuda()
        base_hashes={name:tensor_sha(getattr(cb,name)) for name in ('grid','grid_packed_abs')}
        states={'zero':torch.zeros(256,8,device='cuda',dtype=torch.float32),
                'nonzero':torch.linspace(-.002,.002,2048,device='cuda',dtype=torch.float32).reshape(256,8)}
        for label in LABELS:
            cpu_payload=codec.read_e8(args.parent_dir/f'{label}.e8')
            payload_hashes={name:tensor_sha(value) for name,value in cpu_payload.items() if isinstance(value,torch.Tensor)}
            m,g=cpu_payload['indices'].shape;n=8*g
            columns=((torch.arange(n,device='cuda',dtype=torch.int32)%17-8).half()/1024)
            rows=(1-2*(torch.arange(m,device='cuda',dtype=torch.int32)%2)).half()
            cotangent=rows[:,None]*columns[None,:]
            for location in ('cpu','cuda') if args.include_cuda_payloads else ('cpu',):
                payload={key:value.to(location) if isinstance(value,torch.Tensor) else value for key,value in cpu_payload.items()}
                for state_name,initial in states.items():
                    references=None
                    for mode in ('frozen','explicit_mm'):
                        for repetition in range(3):
                            result,weight,gradient=timed_trial(payload,cb,initial,cotangent,mode)
                            if references is None:references=(weight.clone(),gradient.clone())
                            result.update(label=label,shape=[m,n],payload_location=location,table_state=state_name,
                                implementation=mode,repetition=repetition,
                                parity_vs_frozen_first=parity(weight,gradient,*references))
                            report['trials'].append(result);write_report(args.report,report)
                            print(json.dumps({k:result[k] for k in ('label','payload_location','table_state','implementation',
                                'repetition','forward_wall_seconds','backward_wall_seconds','parity_vs_frozen_first')}),flush=True)
                            del weight,gradient
                    del references
                for key,digest in payload_hashes.items():
                    if tensor_sha(payload[key])!=digest:raise RuntimeError('Probe changed frozen payload values')
                del payload
            del cotangent,cpu_payload
        if codec.rotate_last is not original_rotation:raise RuntimeError('Rotation monkeypatch was not restored')
        if {name:tensor_sha(getattr(cb,name)) for name in base_hashes}!=base_hashes:raise RuntimeError('Probe mutated codebook')
        for path,digest in bound.items():
            if sha(path)!=digest:raise RuntimeError(f'Probe modified a bound input: {path}')
        groups={}
        for row in report['trials']:
            key='/'.join(row[k] for k in ('label','payload_location','table_state','implementation'))
            groups.setdefault(key,[]).append(row)
        report['summary']={key:{'first_total_seconds':rows[0]['total_wall_seconds'],
            'warm_mean_forward_seconds':sum(r['forward_wall_seconds'] for r in rows[1:])/2,
            'warm_mean_backward_seconds':sum(r['backward_wall_seconds'] for r in rows[1:])/2,
            'warm_mean_total_seconds':sum(r['total_wall_seconds'] for r in rows[1:])/2,
            'all_decoded_fp16_bitwise_equal':all(r['parity_vs_frozen_first']['decoded_fp16_bitwise_equal'] for r in rows),
            'worst_gradient_relative_l2_difference':max(r['parity_vs_frozen_first']['table_gradient_relative_l2_difference'] for r in rows)}
            for key,rows in groups.items()}
        report.update(complete=True,elapsed_seconds=time.perf_counter()-started,
            input_hashes_unchanged=True,codebook_unchanged=True,monkeypatch_restored=True,
            explicit_mm_fp16_parity_pass=all(row['parity_vs_frozen_first']['decoded_fp16_bitwise_equal']
                for row in report['trials'] if row['implementation']=='explicit_mm'))
        write_report(args.report,report)
        print(json.dumps({'complete':True,'report_sha256':sha(args.report),'summary':report['summary']}),flush=True)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write_report(args.report,report)
        raise
    finally:
        codec.rotate_last=original_rotation


if __name__=='__main__':main()
