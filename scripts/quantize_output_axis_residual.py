#!/usr/bin/env python3
"""Fixed56 output-axis quantization on the measured input-axis primary base."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import quantize_input_axis_residual as input_stage
from mamba_e8w5 import codec,input_axis_residual
from mamba_e8w5.output_axis_residual import (FORMAT,LABELS,SHAPE,RECIPE,PROTOCOL_SHA,
    INPUT_MANIFEST_SHA,INPUT_REPORT_SHA,read_axis_e8,verify_overlay)
from mamba_e8w5.runtime import MODEL_CONFIG,load_source_state
native=input_stage.native
tensor_sha=input_stage.tensor_sha
INPUT_BUILDER_SHA='b06c947c0b034c645ea66f70432eb929df7c7791c31bc4f720d42fe65d960aa7'
INPUT_VERIFIER_SHA='baff8c56c875258830ac00076f880105b7933a173f57ba0bf38e27cbf1436c06'


def verify_inputs():
    if (native.sha(ROOT/'scripts/quantize_input_axis_residual.py')!=INPUT_BUILDER_SHA
            or native.sha(ROOT/'mamba_e8w5/input_axis_residual.py')!=INPUT_VERIFIER_SHA):
        raise ValueError('Frozen input-stage helpers changed')
    parent,cal,_,_,accepted,old_binding=input_stage.verify_inputs()
    directory=ROOT/'artifacts/input_axis_residual_v1';report_path=ROOT/'reports/input_axis_residual_v1_eval.json'
    pins={Path(p):digest for p,digest in old_binding['checked_input_sha256'].items()}
    pins.update({ROOT/'docs/OUTPUT_AXIS_RESIDUAL_PROTOCOL.md':PROTOCOL_SHA,directory/'manifest.json':INPUT_MANIFEST_SHA,
        report_path:INPUT_REPORT_SHA,ROOT/'scripts/quantize_input_axis_residual.py':INPUT_BUILDER_SHA,
        ROOT/'mamba_e8w5/input_axis_residual.py':INPUT_VERIFIER_SHA})
    for path,digest in pins.items():
        if native.sha(path)!=digest:raise ValueError(f'Bound output-stage input differs: {path}')
    prior=json.loads(report_path.read_text());inputs=input_axis_residual.verify_overlay(directory)
    if (prior['complete'] is not True or prior['primary_arm']!='axis_in_w4_w5'
            or prior['full_validation']['performed'] is not True
            or prior['integrity']['overlay_manifest_sha256']!=INPUT_MANIFEST_SHA
            or inputs['binding']['source_checkpoint_sha256']!=native.SOURCE_CHECKPOINT_SHA256):
        raise ValueError('Measured input-axis candidate identity differs')
    inherited={name:{**row,'origin':'parent'} for name,row in parent['files'].items()
        if name not in {x+'.e8' for x in LABELS}}
    for name,row in inputs['files'].items():
        inherited[name]={**row,'origin':'input_axis_residual_v1'};pins[directory/name]=row['sha256']
    small=json.loads((ROOT/'artifacts/small_compensation_v1/manifest.json').read_text())
    inherited['other_fp16.pt']={**small['files']['other_fp16.pt'],'origin':'small_compensation_v1'}
    w4=json.loads((ROOT/'reports/vocab_w4_v1.json').read_text())['files']['embedding.uniform']
    inherited['embedding.uniform']={'bytes':w4['actual_file_bytes'],'sha256':w4['sha256'],'origin':'vocab_w4_v1'}
    pins[ROOT/'artifacts/vocab_w4_v1/embedding.uniform']=w4['sha256']
    hessians={}
    for label in LABELS:
        entry=cal['matrices'][label];path=ROOT/'calibration/v1'/entry['file']
        if (path.name!=label+'.pt' or entry['shape']!=[8192,8192]
                or entry['sha256']!=parent['matrices'][label]['hessian_sha256']):raise ValueError('Original output Hessian differs')
        hessians[label]=path;pins[path]=entry['sha256']
    for name in ('scripts/quantize_output_axis_residual.py','mamba_e8w5/output_axis_residual.py'):
        pins[ROOT/name]=native.sha(ROOT/name)
    for path,digest in pins.items():
        if native.sha(path)!=digest:raise ValueError(f'Bound output-stage file differs: {path}')
    baseline_bytes=sum(row['bytes'] for row in inherited.values())+sum(parent['files'][label+'.e8']['bytes'] for label in LABELS)
    if (len(inherited)!=61 or baseline_bytes!=3021487451
            or baseline_bytes!=prior['integrity']['resolved_raw_data_bytes_by_arm']['axis_in_w4_w5']):
        raise ValueError('Paired baseline raw accounting differs')
    expected=prior['integrity']['resolved507_fp16_sha256_by_arm']['axis_in_w4_w5']
    if len(expected)!=507:raise ValueError('Missing measured paired-baseline507 hashes')
    for layer in range(56):
        if expected[f'backbone.layers.{layer}.mixer.in_proj.weight']!=inputs['matrices'][f'layer{layer}.in_proj']['decoded_fp16_sha256']:
            raise ValueError('Inherited input-axis tensor identity differs')
    binding={**old_binding,'protocol_sha256':PROTOCOL_SHA,'input_manifest_sha256':INPUT_MANIFEST_SHA,
        'input_evaluation_sha256':INPUT_REPORT_SHA,'input_builder_source_sha256':INPUT_BUILDER_SHA,
        'input_verifier_source_sha256':INPUT_VERIFIER_SHA,'builder_source_sha256':native.sha(__file__),
        'verifier_source_sha256':native.sha(ROOT/'mamba_e8w5/output_axis_residual.py'),
        'checked_input_sha256':{str(p):digest for p,digest in pins.items()}}
    return parent,cal,hessians,inherited,baseline_bytes,binding


def self_test():
    receipt=input_stage.self_test()
    if LABELS!=tuple(f'layer{i}.out_proj' for i in range(56)) or SHAPE!=[4096,8192]:raise AssertionError('Output contract differs')
    receipt['reused_frozen_test_source_sha256']=INPUT_BUILDER_SHA
    receipt['output_contract']={'matrices':56,'shape':SHAPE,'seeds':[1001,1111],'extra_code_plane_bytes':117440512}
    return receipt


def run(args,report):
    parent,cal,hessians,inherited,baseline_bytes,binding=verify_inputs()
    report.update(binding=binding,model_config=MODEL_CONFIG,recipe=RECIPE,parent_manifest_sha256=native.PARENT_SHA,
        small_manifest_sha256=native.SMALL_SHA,source_checkpoint_sha256=native.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=native.TOKENIZER_SHA256,input_manifest_sha256=INPUT_MANIFEST_SHA,input_evaluation_sha256=INPUT_REPORT_SHA,
        inherited_files=inherited,matrices={},files={})
    args.out_dir.mkdir(exist_ok=False);manifest_path=args.out_dir/'manifest.json'
    native.write_json(manifest_path,report);native.write_json(args.report,report)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');report['environment']=native.environment_receipt()
    source=load_source_state(ROOT/'models/source');cb,ldlq=codec.load_reference_primitives();cb=cb.cuda()
    if cb.grid_packed_abs.cpu().numpy().astype('<i4').tobytes()!=(ROOT/'artifacts/e8w5_v1/e8_codebook.bin').read_bytes():
        raise ValueError('Shared E8 codebook changed')
    with torch.inference_mode():
        for layer,label in enumerate(LABELS):
            started=time.monotonic();key=f'backbone.layers.{layer}.mixer.out_proj.weight';original=source[key]
            if original.dtype!=torch.bfloat16 or list(original.shape)!=SHAPE:raise ValueError('Original BF16 output geometry differs')
            hp=hessians[label];hsha=native.sha(hp)
            if hsha!=cal['matrices'][label]['sha256']:raise ValueError('Output Hessian changed before quantization')
            h=torch.load(hp,map_location='cuda',weights_only=True)
            if h.dtype!=torch.float32 or tuple(h.shape)!=(8192,8192) or not torch.isfinite(h).all():raise ValueError('Invalid original output Hessian')
            weight=original.to('cuda');torch.cuda.reset_peak_memory_stats()
            restored,info,payload=codec.vector_quantize(weight,h,cb,ldlq,1001+2*layer,
                damping=.01,scale_override=.9,tune_iters=2,use_feedback=True,residual_bits=4)
            amp=payload['axis_residual_amplitude'];idx=payload['indices']
            if not math.isfinite(amp) or amp<=0 or int(idx.min())<0 or int(idx.max())>=1<<20:raise ValueError('Invalid20-bit quantizer payload')
            target=args.out_dir/(label+'.e8');pending=target.with_suffix('.e8.part')
            with pending.open('xb'):pass
            codec.write_e8(pending,payload,info);disk,layout=read_axis_e8(pending,SHAPE)
            if not torch.equal(disk['indices'],idx):raise ValueError('20-bit stored output indices differ')
            decoded=codec.decode_e8(disk,cb,device='cuda')
            if not torch.isfinite(restored).all() or not torch.isfinite(decoded).all():raise ValueError('Nonfinite restored output FP16')
            restored_sha=tensor_sha(restored);decoded_sha=tensor_sha(decoded)
            if decoded_sha!=restored_sha:raise ValueError('Output FP16 disk readback differs from quantizer')
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
        if native.sha(path)!=digest:raise ValueError(f'Input changed during output quantization: {path}')
    new_bytes=sum(x['bytes'] for x in report['files'].values());original_bytes=sum(parent['files'][label+'.e8']['bytes'] for label in LABELS)
    delta=new_bytes-original_bytes;report['data_file_bytes']=new_bytes
    report['storage']={'new_overlay_bytes':new_bytes,'replaced_e8_bytes':original_bytes,'raw_delta_including_changed_headers':delta,
        'extra_code_plane_bytes':sum(x['layout']['axis_nibble_bytes'] for x in report['matrices'].values()),
        'paired_baseline_logical_data_bytes':baseline_bytes,'resolved_candidate_raw_data_bytes':baseline_bytes+delta,
        'inherited_raw_data_bytes':sum(row['bytes'] for row in inherited.values()),'complete_distribution_built':False,
        'scope':'Resolved model data count replaces56 originals; manifests/software/tokenizer excluded. Physical overlays are not added twice.'}
    if report['storage']['extra_code_plane_bytes']!=117440512:raise ValueError('Unexpected output residual-plane capacity')
    if report['storage']['resolved_candidate_raw_data_bytes']!=report['storage']['inherited_raw_data_bytes']+new_bytes:
        raise ValueError('Resolved output model accounting differs')
    report['complete']=True;native.write_json(manifest_path,report);verify_overlay(args.out_dir)
    report['manifest_sha256']=native.sha(manifest_path);report['manifest_bytes']=manifest_path.stat().st_size


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/output_axis_residual_v1')
    parser.add_argument('--cpu-preflight',action='store_true');args=parser.parse_args()
    args.report=(args.report or ROOT/'reports'/('output_axis_residual_v1_cpu.json' if args.cpu_preflight else 'output_axis_residual_v1_quantization.json')).resolve()
    args.out_dir=args.out_dir.resolve()
    if args.report.parent!=ROOT/'reports' or args.out_dir.parent!=ROOT/'artifacts':raise ValueError('Use project reports/artifacts paths')
    if args.report.exists() or args.report.with_suffix('.json.tmp').exists() or (not args.cpu_preflight and args.out_dir.exists()):
        raise FileExistsError('Fresh output paths required; no overwrite/resume')
    report={'format':FORMAT,'complete':False,'script_sha256':native.sha(__file__),'fitting_language_loss':False,
        'mode':'cpu_preflight' if args.cpu_preflight else 'quantization','quantization_matrices':56};started=time.monotonic()
    try:
        if args.cpu_preflight:
            report['self_test']=self_test();parent,cal,hs,inherited,base,binding=verify_inputs()
            report.update(binding=binding,hessian_count=len(hs),inherited_file_count=len(inherited),
                paired_baseline_raw_data_bytes=base,cuda_initialized=torch.cuda.is_initialized())
            assert not report['cuda_initialized'];report['complete']=True
        else:run(args,report)
    except BaseException as error:
        report.update(complete=False,error=repr(error),traceback=traceback.format_exc())
        if not args.cpu_preflight and (args.out_dir/'manifest.json').exists():native.write_json(args.out_dir/'manifest.json',report)
        raise
    finally:report['elapsed_seconds']=time.monotonic()-started;native.write_json(args.report,report)


if __name__=='__main__':main()
