#!/usr/bin/env python3
"""Four fixed W4 vocabulary interventions on previously observed TRAIN windows.

No fitting, temperature, validation, MK, promotion, or checkpoint selection.
W4 files are generated directly from the original BF16 checkpoint and reloaded.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_teacher_kl_compensation as native
import prepare_teacher_kl_data as prepared
from diagnose_output_calibration_v2 import reproducibility_snapshot
from mamba_e8w5 import codec
from mamba_e8w5.runtime import load_source_state,checkpoint_path

PROTOCOL_SHA='3e0748b1393400a5d4ac168587330ba587d84a3171f86e243aa08dda34e091be'
HISTORY_SHA='d8acf27dfa500d18a6d4b25dc6d95e3edaa10a0ccc27f49369cb02ef2649e59a'
CODE={**native.FROZEN,**prepared.CODE_BOUND,
    'scripts/evaluate_teacher_kl_compensation.py':'b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071',
    'scripts/evaluate_small_compensation.py':'2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be',
    'scripts/prepare_teacher_kl_data.py':'4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13',
    'scripts/diagnose_output_calibration_v2.py':'232e82d6da7ca32a0375a37a999dfbcedb68f34a1bf40013ba282d627cf09caa',
    'scripts/diagnose_output_calibration.py':'116c69a5151a2497209408ed4fef7561fcae38624b20e6bdf659b40b3126fd64',
    'scripts/diagnose_low_rank_generalization.py':'e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879',
    'mamba_e8w5/quantize.py':'7a63da784932e53eecfa49e1403d140e142f05d97b1621729af7fe3f20b03b08'}
VOCAB=('backbone.embedding.weight','lm_head.weight')
FILES=('embedding.uniform','lm_head.uniform')
ARMS=(('w5_w5',(0,0)),('w4_embedding',(1,0)),('w4_head',(0,1)),('w4_both',(1,1)))


def verify_inputs():
    pins={ROOT/'docs/VOCAB_W4_DIAGNOSTIC_PROTOCOL.md':PROTOCOL_SHA,
        ROOT/'reports/small_compensation_v1_eval.json':native.BASELINE_SHA,
        ROOT/'reports/teacher_kl_compensation_v1_eval.json':HISTORY_SHA,
        ROOT/'artifacts/e8w5_v1/manifest.json':native.PARENT_SHA,
        ROOT/'artifacts/small_compensation_v1/manifest.json':native.SMALL_SHA,
        ROOT/'artifacts/small_compensation_v1/other_fp16.pt':native.SMALL_VALUES_SHA,
        checkpoint_path(ROOT/'models/source'):native.SOURCE_CHECKPOINT_SHA256}
    pins.update({ROOT/name:digest for name,digest in {**CODE,**prepared.BOUND}.items()})
    for path,digest in pins.items():
        if native.sha(path)!=digest:raise ValueError(f'Pinned input differs: {path}')
    prior=json.loads((ROOT/'reports/small_compensation_v1_eval.json').read_text())
    history=json.loads((ROOT/'reports/teacher_kl_compensation_v1_eval.json').read_text())
    audit_path=ROOT/'scripts/audit_decoded.py';pins[audit_path]=prior['loaded_tensor_audit_source_sha256']
    if native.sha(audit_path)!=pins[audit_path]:raise ValueError('Loaded-tensor helper changed')
    parent,small,receipt=native.small_overlay.verify_overlay(ROOT/'artifacts/e8w5_v1',
        ROOT/'artifacts/small_compensation_v1',initialization_dir=ROOT/'artifacts/norm_compensation_v1')
    for name,row in parent['files'].items():pins[ROOT/'artifacts/e8w5_v1'/name]=row['sha256']
    directory=ROOT/'training_data/teacher_kl_compensation_v1'
    data,training,windows=prepared.verify_data(directory,native.DATA_SHA);del training
    for name in ('manifest.json','training_tokens.pt','heldout_tokens.pt'):pins[directory/name]=native.sha(directory/name)
    hashes={row['name']:row['decoded_fp16_sha256'] for row in prior['parent_loaded_tensor_audit']['tensors']}
    hashes.update(prior['small_loaded_tensor_audit']['small']['loaded_and_export_fp16_sha256'])
    if len(hashes)!=507 or tuple(windows.shape)!=(64,2048):raise ValueError('Incorrect tensor/window coverage')
    if history['baseline507_actual_content_audit']['unchanged_content_sha256']!=hashes:
        raise ValueError('Historical accepted model identity differs')
    pins[Path(__file__).resolve()]=native.sha(__file__)
    return parent,small,receipt,data,windows,hashes,history,{str(p):h for p,h in pins.items()}


def raw_tensor_hash(tensor):
    digest=hashlib.sha256()
    for first in range(0,len(tensor),128):
        digest.update(tensor[first:first+128].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def uniform_layout(path):
    header,offset=codec.uniform_header(path);count=math.prod(header['shape'])
    codes=count*header['bits']//8;scales=count//128*2
    if path.stat().st_size!=offset+codes+scales:raise ValueError('Uniform byte accounting differs')
    with path.open('rb') as stream:header_bytes=stream.read(offset)
    return {'header':header,'header_bytes':offset,'header_sha256':hashlib.sha256(header_bytes).hexdigest(),
        'packed_code_bytes':codes,'scale_count':count//128,'fp16_scale_bytes':scales,
        'actual_file_bytes':path.stat().st_size,'sha256':native.sha(path)}


@torch.inference_mode()
def score_window(model,window,chunk_tokens=64):
    tokens=window.to(next(model.parameters()).device);chunks=[]
    with torch.autocast(device_type=tokens.device.type,enabled=False):
        hidden=model.backbone(tokens[:-1][None])
        if hidden.dtype!=torch.float16 or not torch.isfinite(hidden).all():raise ValueError('Invalid FP16 hidden')
        for position in range(0,len(tokens)-1,chunk_tokens):
            end=min(position+chunk_tokens,len(tokens)-1)
            logits=model.lm_head(hidden[:,position:end])
            if logits.dtype!=torch.float16 or not torch.isfinite(logits).all():raise ValueError('Invalid native FP16 head')
            loss=float(F.cross_entropy(logits.float().flatten(0,1),tokens[position+1:end+1],reduction='sum'))
            if not math.isfinite(loss):raise ValueError('Nonfinite CE')
            chunks.append(loss);del logits
    nll=math.fsum(chunks);count=len(tokens)-1
    return {'target_tokens':count,'nll':nll,'ppl':math.exp(nll/count),'ce_chunk_sums':chunks,
        'chunk_target_counts':[min(chunk_tokens,count-p) for p in range(0,count,chunk_tokens)]}


def summarize(rows):
    count=sum(row['target_tokens'] for row in rows);nll=math.fsum(row['nll'] for row in rows)
    return {'windows':len(rows),'target_tokens':count,'nll':nll,'mean_nll':nll/count,'ppl':math.exp(nll/count)}


def set_vocab(model,parameters):
    for name,value in zip(VOCAB,parameters):
        module,leaf=name.rsplit('.',1);setattr(model.get_submodule(module),leaf,value)


def run(args,report):
    parent,small,receipt,data,windows,hashes,history,pins=verify_inputs()
    report.update(binding={'protocol_sha256':PROTOCOL_SHA,'source_checkpoint_sha256':native.SOURCE_CHECKPOINT_SHA256,
        'parent_manifest_sha256':native.PARENT_SHA,'small_manifest_sha256':native.SMALL_SHA,
        'small_values_sha256':native.SMALL_VALUES_SHA,'data_manifest_sha256':native.DATA_SHA,
        'historical_report_sha256':HISTORY_SHA,'checked_input_sha256':pins},base_package=receipt,
        dataset=data['dataset'],window_plan=data['heldout_windows'],files={},arms={})
    native.write_json(args.report,report);args.out_dir.mkdir(parents=False,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
    report['environment']=native.environment_receipt();report['reproducibility_before']=reproducibility_snapshot()
    state=load_source_state(ROOT/'models/source');source={name:state[name] for name in VOCAB};del state
    for name,filename in zip(VOCAB,FILES):
        weight=source[name]
        if weight.dtype!=torch.bfloat16 or tuple(weight.shape)!=(256000,4096):raise ValueError('Original BF16 vocabulary geometry differs')
        path=args.out_dir/filename
        if path.exists() or path.with_name(path.name+'.partial').exists():raise FileExistsError(path)
        before=raw_tensor_hash(weight)
        info=codec.write_uniform(path,weight,bits=4,device='cuda',rows_per_chunk=256)
        if raw_tensor_hash(weight)!=before:raise ValueError('Original vocabulary tensor changed')
        info.update(uniform_layout(path),source_key=name,source_dtype=str(weight.dtype),source_tensor_bytes=weight.numel()*weight.element_size(),
            source_tensor_sha256=before,conversion='Original BF16 -> writer FP32 row chunk; FP16 scales; stored W4 -> FP16 readback')
        if info['actual_file_bytes']!=540672079:raise ValueError('W4 file size differs from the fixed group128 format')
        report['files'][filename]=info;pins[str(path)]=info['sha256'];native.write_json(args.report,report)
    del source,weight;gc.collect()
    model=native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    native.small_overlay.apply_overlay(model,ROOT/'artifacts/small_compensation_v1',small)
    model.eval().requires_grad_(False);original=tuple(dict(model.named_parameters())[name] for name in VOCAB)
    report['coverage']=native.audit_parameter_coverage(model);report['base507_initial']=native.audit_frozen_values(model,hashes)
    decoded=tuple(torch.nn.Parameter(codec.read_uniform(args.out_dir/f,device='cuda'),requires_grad=False) for f in FILES)
    decoded_hashes={name:native.tensor_sha_fp16(value) for name,value in zip(VOCAB,decoded)}
    report['stored_w4_decoded_fp16_sha256']=decoded_hashes
    original_refs=dict(model.named_parameters())
    try:
        for arm,flags in (*ARMS,('w5_w5_repeat',(0,0))):
            replacements=tuple(decoded[i] if flags[i] else original[i] for i in range(2));set_vocab(model,replacements)
            expected=dict(hashes);changed=[name for name,flag in zip(VOCAB,flags) if flag]
            expected.update({name:decoded_hashes[name] for name in changed})
            for name,value in model.named_parameters():
                if name not in changed and value is not original_refs[name]:raise ValueError('Unrelated parameter reference changed')
            entry={'changed_parameters':changed,'coverage':native.audit_parameter_coverage(model),
                'actual507_before':native.audit_frozen_values(model,expected),'windows':[]}
            report['arms'][arm]=entry
            for index,(window,plan) in enumerate(zip(windows,data['heldout_windows'])):
                row=score_window(model,window);row.update(plan,index=index);entry['windows'].append(row)
                if (index+1)%16==0:print(f'[{arm}] {index+1}/64',flush=True);native.write_json(args.report,report)
            entry['summary']=summarize(entry['windows']);entry['actual507_after']=native.audit_frozen_values(model,expected)
            if entry['summary']['target_tokens']!=131008:raise ValueError('Scoring target count changed')
            native.write_json(args.report,report)
        baseline=report['arms']['w5_w5'];repeat=report['arms']['w5_w5_repeat']
        report['baseline_repeat_exact']=baseline['windows']==repeat['windows'] and baseline['summary']==repeat['summary']
        native.write_json(args.report,report)
        if not report['baseline_repeat_exact']:raise ValueError('Same-process baseline repeat differs')
        base_nll=baseline['summary']['nll'];base_ppl=baseline['summary']['ppl']
        report['comparison']={name:{'ppl_relative_change':report['arms'][name]['summary']['ppl']/base_ppl-1,
            'mean_nll_delta':(report['arms'][name]['summary']['nll']-base_nll)/131008,
            'windows_improve':sum(a['nll']<b['nll'] for a,b in zip(report['arms'][name]['windows'],baseline['windows'])),
            'per_window_mean_nll_delta':[(a['nll']-b['nll'])/2047 for a,b in zip(report['arms'][name]['windows'],baseline['windows'])]}
            for name,_ in ARMS[1:]}
        historical=history['reserved_results']['best_small_e8w5']
        historical_ppl=math.exp(math.fsum(row['student_nll'] for row in historical)/131008)
        report['historical_baseline_drift']={'prior_ppl':historical_ppl,'current_ppl':base_ppl,
            'relative_ppl_change':base_ppl/historical_ppl-1,'scope':'Descriptive cross-process comparison; no historical bitwise gate.'}
        report['mean_nll_interaction']=(report['arms']['w4_both']['summary']['nll']-report['arms']['w4_embedding']['summary']['nll']
            -report['arms']['w4_head']['summary']['nll']+base_nll)/131008
        w5={f:uniform_layout(ROOT/'artifacts/e8w5_v1'/f) for f in FILES}
        report['storage']={'original_w5_files':w5,'new_w4_file_bytes':sum(x['actual_file_bytes'] for x in report['files'].values()),
            'raw_saving_if_both_replaced':sum(w5[f]['actual_file_bytes']-report['files'][f]['actual_file_bytes'] for f in FILES),
            'base_logical_data_bytes':receipt['logical_candidate_data_bytes'],'complete_distribution_built':False}
        if report['storage']['raw_saving_if_both_replaced']!=262144000:raise ValueError('W5-to-W4 raw saving differs')
        report['complete']=True
    finally:
        set_vocab(model,original);report['base507_restored']=native.audit_frozen_values(model,hashes)
        report['references_restored']=all(value is original_refs[name] for name,value in model.named_parameters())
        report['reproducibility_after']=reproducibility_snapshot();report['gpu_memory']=native.gpu_memory_receipt()
        for path,digest in pins.items():
            if native.sha(path)!=digest:raise ValueError(f'Bound file changed: {path}')


def self_test():
    codes=np.arange(16,dtype=np.uint8).repeat(8)
    assert np.array_equal(codec.unpack_uniform_codes(codec.pack_uniform_codes(codes,4),4),codes)
    weight=torch.linspace(-3,3,5*256).reshape(5,256).to(torch.bfloat16);weight[0,:128]=0
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'fixture.uniform';info=codec.write_uniform(path,weight,bits=4,device='cpu',rows_per_chunk=2)
        x=weight.float().reshape(-1,128);s=(x.abs().amax(-1,keepdim=True)/7).half().float();s=torch.where(s==0,1,s)
        expected=(torch.round(x/s).clamp(-7,7)*s).half().reshape_as(weight)
        # Unsigned offset codes do not retain IEEE negative-zero signs.
        # Match that format contract before requiring byte-exact readback.
        expected=torch.where(expected==0,torch.zeros_like(expected),expected)
        actual=codec.read_uniform(path)
        assert raw_tensor_hash(expected)==raw_tensor_hash(actual)
        assert raw_tensor_hash(codec.read_uniform_rows(path,1,4))==raw_tensor_hash(expected[1:4])
        assert [len(v) for _,v in codec.iter_uniform(path,chunk_rows=2)]==[2,2,1]
        layout=uniform_layout(path);assert layout['packed_code_bytes']==640 and layout['fp16_scale_bytes']==20
        try:codec.write_uniform(Path(tmp)/'invalid',torch.zeros(2,129),bits=4,device='cpu')
        except ValueError:pass
        else:raise AssertionError('Partial group must be rejected')
    # Native FP16 scoring and a non-full final chunk must use target weighting.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Embedding(23,8,dtype=torch.float16)
            self.lm_head=torch.nn.Linear(8,23,bias=False,dtype=torch.float16)
    torch.manual_seed(734);toy=Toy().eval();tokens=torch.arange(8)
    scored=score_window(toy,tokens,chunk_tokens=3)
    with torch.inference_mode():
        logits=toy.lm_head(toy.backbone(tokens[:-1][None])).float().flatten(0,1)
        expected_loss=float(F.cross_entropy(logits,tokens[1:],reduction='sum'))
    assert scored['chunk_target_counts']==[3,3,1] and scored['target_tokens']==7
    assert math.isclose(scored['nll'],expected_loss,abs_tol=3e-6)
    assert summarize([scored])['mean_nll']==scored['nll']/7
    assert not torch.cuda.is_initialized()
    return {'passed':True,'cuda_initialized':False,'checks':['All16 nibble values pack/unpack','BF16 direct-to-FP32 W4 independent scale/round oracle',
        'Zero group, cross-row readback and final1-row chunk','Partial128-value group rejection','Exact header/codes/FP16-scale byte accounting',
        'Native FP16 toy CE and3+3+1 target-weighted loss tail']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/vocab_w4_v1')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--cpu-preflight',action='store_true');args=parser.parse_args()
    if args.self_test or args.cpu_preflight:
        if args.report and (args.report.exists() or args.report.with_suffix('.json.tmp').exists()):raise FileExistsError(args.report)
        result={'complete':False,'script_sha256':native.sha(__file__)};started=time.monotonic()
        try:
            result['self_test']=self_test()
            if args.cpu_preflight:
                parent,small,receipt,data,windows,hashes,history,pins=verify_inputs()
                result['preflight']={'window_shape':list(windows.shape),'targets':windows.shape[0]*(windows.shape[1]-1),
                    'accepted_tensor_hashes':len(hashes),'checked_input_sha256':pins,'logical_base_data_bytes':receipt['logical_candidate_data_bytes']}
            result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized'];result['complete']=True
        except BaseException as error:
            result.update(error=repr(error),traceback=traceback.format_exc());raise
        finally:
            result['elapsed_seconds']=time.monotonic()-started
            if args.report:native.write_json(args.report,result)
        print(json.dumps(result,indent=2));return
    if args.report is None:parser.error('--report is required')
    args.report=args.report.resolve();args.out_dir=args.out_dir.resolve()
    if args.report.parent!=ROOT/'reports' or args.out_dir.parent!=ROOT/'artifacts':raise ValueError('Output paths must be within reports/artifacts')
    if args.report.exists() or args.report.with_suffix('.json.tmp').exists() or args.out_dir.exists():raise FileExistsError('Fresh outputs required; no resume/overwrite')
    report={'format':'MAMBA2_VOCAB_W4_DIAGNOSTIC_V1','complete':False,'script_sha256':native.sha(__file__),
        'arm_order':[name for name,_ in ARMS]+['w5_w5_repeat'],'split':'previously observed reserved TRAIN',
        'fitting_performed':False,'validation_performed':False,'mk_performed':False,'candidate_advancement':False,
        'limitations':['Native FP16 expansion is a quality reference, not compressed runtime residency.',
            'Four fixed diagnostic arms; observed TRAIN scores are not untouched test results.']}
    started=time.monotonic()
    try:run(args,report)
    except BaseException as error:
        report.update(complete=False,error=repr(error),traceback=traceback.format_exc());raise
    finally:report['elapsed_seconds']=time.monotonic()-started;native.write_json(args.report,report)


if __name__=='__main__':main()
