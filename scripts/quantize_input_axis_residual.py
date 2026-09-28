#!/usr/bin/env python3
"""Fixed56 input projections: original source/H, two-sweep E8+axis4 overlay."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_teacher_kl_compensation as native
from mamba_e8w5 import codec
from mamba_e8w5.input_axis_residual import FORMAT,LABELS,SHAPE,RECIPE,read_axis_e8,verify_overlay
from mamba_e8w5.runtime import MODEL_CONFIG,load_source_state,checkpoint_path

PROTOCOL_SHA='bf4af74313812a5f20373e9cfa68654b2cbe39a66841788ae4c66e965aa80088'
CALIBRATION_SHA='70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
W4_REPORT_SHA='3ae1bb986ddfa7f2e514fb2451885a83feee5b6c5cc3a1924f248b4a259e353b'


def tensor_sha(tensor):
    value=hashlib.sha256()
    for start in range(0,len(tensor),128):
        value.update(tensor[start:start+128].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def verify_inputs():
    parent_dir=ROOT/'artifacts/e8w5_v1';small_dir=ROOT/'artifacts/small_compensation_v1'
    pins={ROOT/'docs/INPUT_AXIS_RESIDUAL_PROTOCOL.md':PROTOCOL_SHA,
        ROOT/'reports/vocab_w4_v1.json':W4_REPORT_SHA,parent_dir/'manifest.json':native.PARENT_SHA,
        small_dir/'manifest.json':native.SMALL_SHA,small_dir/'other_fp16.pt':native.SMALL_VALUES_SHA,
        ROOT/'calibration/v1/manifest.json':CALIBRATION_SHA,
        checkpoint_path(ROOT/'models/source'):native.SOURCE_CHECKPOINT_SHA256}
    pins.update({ROOT/name:digest for name,digest in native.FROZEN.items()})
    pins[ROOT/'scripts/evaluate_teacher_kl_compensation.py']='b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071'
    pins[ROOT/'scripts/evaluate_small_compensation.py']='2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be'
    parent=json.loads((parent_dir/'manifest.json').read_text());cal=json.loads((ROOT/'calibration/v1/manifest.json').read_text())
    if (not parent['complete'] or not cal['complete'] or cal['calibration_split']!='train' or cal['evaluation_data_used'] is not False
            or cal['model_config']!=MODEL_CONFIG or parent['model_config']!=MODEL_CONFIG
            or cal['source_checkpoint_sha256']!=native.SOURCE_CHECKPOINT_SHA256
            or parent['binding']['hessian_manifest_sha256']!=CALIBRATION_SHA):raise ValueError('Original calibration/source differs')
    for name,digest in parent['binding']['code_sha256'].items():pins[ROOT/'mamba_e8w5'/name]=digest
    for name,digest in parent['binding']['quip_sha256'].items():pins[ROOT/'third_party/quip-sharp'/name]=digest
    for name,row in parent['files'].items():pins[parent_dir/name]=row['sha256']
    hessians={}
    for label in LABELS:
        entry=cal['matrices'][label];path=ROOT/'calibration/v1'/entry['file']
        if path.name!=label+'.pt' or entry['shape']!=[4096,4096] or entry['sha256']!=parent['matrices'][label]['hessian_sha256']:
            raise ValueError('Original input Hessian differs')
        pins[path]=entry['sha256'];hessians[label]=path
    for name in ('scripts/quantize_input_axis_residual.py','mamba_e8w5/input_axis_residual.py'):
        pins[ROOT/name]=native.sha(ROOT/name)
    for path,digest in pins.items():
        if native.sha(path)!=digest:raise ValueError(f'Bound input differs: {path}')
    _,small,receipt=native.small_overlay.verify_overlay(parent_dir,small_dir,initialization_dir=ROOT/'artifacts/norm_compensation_v1')
    inherited={name:{**row,'origin':'parent'} for name,row in parent['files'].items() if name not in {x+'.e8' for x in LABELS}}
    inherited['other_fp16.pt']={**small['files']['other_fp16.pt'],'origin':'small_compensation_v1'}
    binding={'protocol_sha256':PROTOCOL_SHA,'source_checkpoint_sha256':native.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256':native.TOKENIZER_SHA256,'calibration_manifest_sha256':CALIBRATION_SHA,
        'codec_source_sha256':native.sha(ROOT/'mamba_e8w5/codec.py'),'runtime_source_sha256':native.sha(ROOT/'mamba_e8w5/runtime.py'),
        'quip_sha256':parent['binding']['quip_sha256'],'builder_source_sha256':native.sha(__file__),
        'verifier_source_sha256':native.sha(ROOT/'mamba_e8w5/input_axis_residual.py'),
        'checked_input_sha256':{str(p):digest for p,digest in pins.items()}}
    return parent,cal,hessians,inherited,receipt,binding


def self_test():
    cb,_=codec.load_reference_primitives();m,n=16,128;count=m*n//8
    high=torch.tensor([0,1,32768,65535],dtype=torch.int32).repeat(count//4)
    low=torch.arange(count,dtype=torch.int32)%16;codes=(high<<4)|low
    payload={'indices':codes.reshape(m,n//8),'balance':torch.ones(n,dtype=torch.float16),
        'input_sign':torch.ones(n,dtype=torch.int8),'output_sign':torch.ones(m,dtype=torch.int8),
        'scale':torch.tensor(.5,dtype=torch.float32),'axis_residual_amplitude':.25}
    direct=cb.grid[high.long()].clone()
    for i in range(count):direct[i,int(low[i])//2]+=.25*(1-2*(int(low[i])%2))
    assert torch.equal(codec.decode_codes(payload['indices'],cb,.25).reshape(-1,8),direct)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'tiny.e8';codec.write_e8(path,payload,{'shape':[m,n]})
        disk,layout=read_axis_e8(path,[m,n]);raw=path.read_bytes();off=layout['header_bytes']
        assert np.array_equal(np.frombuffer(raw[off:off+count*2],dtype='<u2'),high.numpy())
        plane=np.frombuffer(raw[off+count*2:off+count*2+count//2],dtype=np.uint8)
        assert np.array_equal(plane&15,low.numpy()[::2]) and np.array_equal(plane>>4,low.numpy()[1::2])
        for key in ('indices','balance','input_sign','output_sign','scale'):assert torch.equal(disk[key],payload[key])
        actual=codec.decode_e8(disk,cb,device='cpu');expected=codec.decode_e8(payload,cb,device='cpu')
        assert tensor_sha(actual)==tensor_sha(expected) and torch.isfinite(actual).all()
        # Independent dense inverse for power-of-two toy geometry.
        def hadamard(size):
            h=torch.ones(1,1)
            while len(h)<size:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
            return h/math.sqrt(size)
        reference=(hadamard(m)@(direct.reshape(m,n)*.5)@hadamard(n)).half()
        torch.testing.assert_close(actual,reference,atol=.002,rtol=.002)
        for variant in (raw[:-1],raw+b'X',b'BADMAGIC'+raw[8:]):
            bad=Path(directory)/'bad.e8';bad.write_bytes(variant)
            try:read_axis_e8(bad,[m,n])
            except (ValueError,AssertionError):pass
            else:raise AssertionError('Invalid byte layout was accepted')
        for amplitude in (0.,-1.,float('nan'),float('inf')):
            header={**layout['header'],'axis_residual_amplitude':amplitude};encoded=json.dumps(header).encode()
            bad.write_bytes(b'ME8HD001'+struct.pack('<I',len(encoded))+encoded+raw[off:])
            try:read_axis_e8(bad,[m,n])
            except ValueError:pass
            else:raise AssertionError('Invalid amplitude accepted')
    assert not torch.cuda.is_initialized()
    return {'passed':True,'cuda_initialized':False,'checks':['uint16 upper plane and low-first nibble plane',
        'All16 signed-axis choices including uint16 boundary codes','Native FP16 disk decode identity and independent dense inverse',
        'Truncation/trailing bytes/magic rejection','Zero/negative/nonfinite amplitude rejection']}


def run(args,report):
    parent,cal,hessians,inherited,base,binding=verify_inputs()
    report.update(binding=binding,model_config=MODEL_CONFIG,recipe=RECIPE,parent_manifest_sha256=native.PARENT_SHA,
        small_manifest_sha256=native.SMALL_SHA,source_checkpoint_sha256=native.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=native.TOKENIZER_SHA256,inherited_files=inherited,matrices={},files={})
    args.out_dir.mkdir(exist_ok=False);manifest_path=args.out_dir/'manifest.json'
    native.write_json(manifest_path,report);native.write_json(args.report,report)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');report['environment']=native.environment_receipt()
    source=load_source_state(ROOT/'models/source');cb,ldlq=codec.load_reference_primitives();cb=cb.cuda()
    if cb.grid_packed_abs.cpu().numpy().astype('<i4').tobytes()!=(ROOT/'artifacts/e8w5_v1/e8_codebook.bin').read_bytes():
        raise ValueError('Shared E8 codebook changed')
    with torch.inference_mode():
        for layer,label in enumerate(LABELS):
            started=time.monotonic();key=f'backbone.layers.{layer}.mixer.in_proj.weight';original=source[key]
            if original.dtype!=torch.bfloat16 or list(original.shape)!=SHAPE:raise ValueError('Original BF16 input geometry differs')
            hp=hessians[label];hsha=native.sha(hp)
            if hsha!=cal['matrices'][label]['sha256']:raise ValueError('Hessian changed before quantization')
            h=torch.load(hp,map_location='cuda',weights_only=True)
            if h.dtype!=torch.float32 or tuple(h.shape)!=(4096,4096) or not torch.isfinite(h).all():raise ValueError('Invalid original Hessian')
            weight=original.to('cuda');torch.cuda.reset_peak_memory_stats()
            restored,info,payload=codec.vector_quantize(weight,h,cb,ldlq,1000+2*layer,
                damping=.01,scale_override=.9,tune_iters=2,use_feedback=True,residual_bits=4)
            amp=payload['axis_residual_amplitude'];idx=payload['indices']
            if not math.isfinite(amp) or amp<=0 or int(idx.min())<0 or int(idx.max())>=1<<20:
                raise ValueError('Invalid untruncated20-bit quantizer payload')
            target=args.out_dir/(label+'.e8');pending=target.with_suffix('.e8.part')
            with pending.open('xb'):pass
            codec.write_e8(pending,payload,info);disk,layout=read_axis_e8(pending,SHAPE)
            if not torch.equal(disk['indices'],idx):raise ValueError('20-bit stored indices differ')
            decoded=codec.decode_e8(disk,cb,device='cuda')
            if not torch.isfinite(restored).all() or not torch.isfinite(decoded).all():raise ValueError('Nonfinite restored FP16 projection')
            restored_sha=tensor_sha(restored);decoded_sha=tensor_sha(decoded)
            if decoded_sha!=restored_sha:raise ValueError('Native FP16 readback differs from quantizer output')
            info.update(file=target.name,source_key=key,source_dtype=str(original.dtype),source_tensor_sha256=tensor_sha(original),
                hessian_sha256=hsha,bytes=pending.stat().st_size,sha256=native.sha(pending),decoded_fp16_sha256=decoded_sha,
                disk_roundtrip_fp16_equal=True,packed_indices_equal=True,layout=layout,
                elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            if target.exists():raise FileExistsError(target)
            os.replace(pending,target);report['matrices'][label]=info
            report['files'][target.name]={k:info[k] for k in ('bytes','sha256','decoded_fp16_sha256')}
            native.write_json(manifest_path,report);native.write_json(args.report,report)
            print(json.dumps({'matrix':layer+1,'total':56,'label':label,'seconds':info['elapsed_seconds'],'bytes':info['bytes']}),flush=True)
            del weight,h,restored,payload,disk,decoded,idx;torch.cuda.empty_cache()
    for path,digest in binding['checked_input_sha256'].items():
        if native.sha(path)!=digest:raise ValueError(f'Input changed during quantization: {path}')
    report['data_file_bytes']=sum(x['bytes'] for x in report['files'].values())
    original_bytes=sum(parent['files'][label+'.e8']['bytes'] for label in LABELS)
    report['storage']={'new_overlay_bytes':report['data_file_bytes'],'replaced_e8_bytes':original_bytes,
        'raw_delta_including_changed_headers':report['data_file_bytes']-original_bytes,
        'extra_code_plane_bytes':sum(x['layout']['axis_nibble_bytes'] for x in report['matrices'].values()),
        'baseline_logical_data_bytes':base['logical_candidate_data_bytes'],'complete_distribution_built':False}
    if report['storage']['extra_code_plane_bytes']!=266076160:raise ValueError('Unexpected residual plane capacity')
    report['complete']=True;native.write_json(manifest_path,report);verify_overlay(args.out_dir)
    report['manifest_sha256']=native.sha(manifest_path);report['manifest_bytes']=manifest_path.stat().st_size


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/input_axis_residual_v1')
    parser.add_argument('--cpu-preflight',action='store_true');args=parser.parse_args()
    args.report=args.report.resolve();args.out_dir=args.out_dir.resolve()
    if args.report.parent!=ROOT/'reports' or args.out_dir.parent!=ROOT/'artifacts':raise ValueError('Use project reports/artifacts paths')
    if args.report.exists() or args.report.with_suffix('.json.tmp').exists() or (not args.cpu_preflight and args.out_dir.exists()):
        raise FileExistsError('Fresh output paths required; no overwrite/resume')
    report={'format':FORMAT,'complete':False,'script_sha256':native.sha(__file__),'fitting_language_loss':False,
        'mode':'cpu_preflight' if args.cpu_preflight else 'quantization','quantization_matrices':56};started=time.monotonic()
    try:
        if args.cpu_preflight:
            report['self_test']=self_test();parent,cal,hs,inherited,base,binding=verify_inputs()
            report.update(binding=binding,hessian_count=len(hs),inherited_file_count=len(inherited),cuda_initialized=torch.cuda.is_initialized())
            assert not report['cuda_initialized'];report['complete']=True
        else:run(args,report)
    except BaseException as error:
        report.update(complete=False,error=repr(error),traceback=traceback.format_exc())
        if not args.cpu_preflight and (args.out_dir/'manifest.json').exists():native.write_json(args.out_dir/'manifest.json',report)
        raise
    finally:report['elapsed_seconds']=time.monotonic()-started;native.write_json(args.report,report)


if __name__=='__main__':main()
