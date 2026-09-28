#!/usr/bin/env python3
"""Fixed three-update FP32 scalar fit, final reload, seven-window screen.

All weights stay frozen. Only a passed screen enables the established complete
validation. No resume, line search, sweep, checkpoint selection or publication.
"""
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

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_teacher_kl_compensation as native
import prepare_teacher_kl_data as prepared
from diagnose_output_calibration_v2 import reproducibility_snapshot

PROTOCOL_SHA='ade965d8381a2365705eade045b67f1eb6d60bde3b87bfbd0d92ec61b0bcac6f'
DATA_SHA='facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
BASE_REPORT_SHA='957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
CODE={**native.FROZEN,
    'scripts/evaluate_teacher_kl_compensation.py':'b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071',
    'scripts/evaluate_small_compensation.py':'2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be',
    'scripts/prepare_teacher_kl_data.py':'4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13',
    'scripts/diagnose_output_calibration_v2.py':'232e82d6da7ca32a0375a37a999dfbcedb68f34a1bf40013ba282d627cf09caa',
    'scripts/diagnose_output_calibration.py':'116c69a5151a2497209408ed4fef7561fcae38624b20e6bdf659b40b3126fd64',
    'scripts/diagnose_low_rank_generalization.py':'e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879',
    **prepared.CODE_BOUND}
ARMS=('source_fp16','best_small_e8w5','scalar_e8w5')
TRAIN_INDICES=tuple(i*447//63 for i in range(64))
SCREEN_STARTS=(362496,718848,1083392,1439744,1802240,2158592,2510848)
fp32=lambda value:struct.unpack('<f',struct.pack('<f',float(value)))[0]
BETA_MIN,BETA_MAX=fp32(.8),fp32(1.2)


def beta_value(value):
    if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
        raise ValueError('Beta must be a finite positive real scalar')
    result=fp32(value)
    if not BETA_MIN<=result<=BETA_MAX:
        raise ValueError('Beta is outside the declared rounded FP32 bounds')
    return result


def write_beta(path,value):
    value=beta_value(value)
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<f',value));stream.flush();os.fsync(stream.fileno())
    if read_beta(path)!=value:raise ValueError('Scalar roundtrip differs')
    return {'file':Path(path).name,'bytes':4,'sha256':native.sha(path),'dtype':'float32_le',
        'value':value,'hex_bytes':struct.pack('<f',value).hex()}


def read_beta(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=4:
        raise ValueError('Expected exactly four regular-file scalar bytes')
    return beta_value(struct.unpack('<f',path.read_bytes())[0])


def scalar_moments(logits,targets,beta):
    """FP32 production per-target derivatives; FP64 inputs only for CPU oracle."""
    beta=beta_value(beta)
    if (logits.ndim!=2 or logits.dtype not in (torch.float32,torch.float64)
            or targets.dtype!=torch.int64 or targets.shape!=logits.shape[:1]
            or not torch.isfinite(logits).all()):raise ValueError('Invalid scalar-loss geometry')
    logp=F.log_softmax(logits*beta,dim=-1);p=logp.exp()
    ce=F.cross_entropy(logits*beta,targets,reduction='none')
    entropy=-(p*logp).sum(-1)
    centered=logits-logits.amax(-1,keepdim=True)
    mean=(p*centered).sum(-1)
    g=mean-centered.gather(1,targets[:,None]).squeeze(1)
    h=(p*(centered-mean[:,None]).square()).sum(-1)
    identity=(ce-entropy)/beta
    if (not all(torch.isfinite(x).all() for x in (ce,g,h,identity)) or torch.any(h<0)
            or not torch.allclose(g,identity,atol=3e-4,rtol=3e-5)):
        raise ValueError('Invalid scalar derivative/curvature identity')
    return ce,g,h,float((g-identity).abs().max())


def next_beta(beta,g,h):
    beta=beta_value(beta)
    if not all(math.isfinite(v) for v in (g,h)) or h<=0:raise ValueError('Finite positive mean curvature required')
    quotient=g/h;step=max(-.05,min(.05,quotient))
    result=beta_value(max(.8,min(1.2,beta-step)))
    return result,{'beta_before':beta,'mean_g':g,'mean_h':h,'unclipped_step':quotient,
        'clipped_step':step,'beta_after_fp32':result,'beta_after_hex_le':struct.pack('<f',result).hex()}


def same_float(a,b):return struct.pack('<d',a)==struct.pack('<d',b)


@torch.inference_mode()
def score_window(model,window,beta,*,derivatives=False,source=None,chunk_tokens=64):
    """Fresh state; generalized stored length handles both TRAIN2048 and validation2049."""
    beta=beta_value(beta);device=model.backbone.embedding.weight.device
    tokens=window.to(device);ids=tokens[:-1][None];targets=tokens[1:]
    if targets.numel()==0:raise ValueError('Empty scoring window')
    values={'base':[],'scaled':[],'source':[]};gparts=[];hparts=[];per_ce=[]
    argmax_changes=0;maximum_identity_error=0.
    with torch.autocast(device_type=device.type,enabled=False):
        source_hidden=source.backbone(ids) if source is not None else None
        hidden=model.backbone(ids)
        for x in ([hidden] if source_hidden is None else [source_hidden,hidden]):
            if x.dtype!=torch.float16 or not torch.isfinite(x).all():raise ValueError('Invalid native FP16 hidden')
        for start in range(0,targets.numel(),chunk_tokens):
            end=min(start+chunk_tokens,targets.numel());truth=targets[start:end]
            if source is not None:
                sz16=source.lm_head(source_hidden[:,start:end])
                if sz16.dtype!=torch.float16:raise ValueError('Source head must compute FP16')
                sz=sz16.float().flatten(0,1)
                if not torch.isfinite(sz).all():raise ValueError('Nonfinite source logits')
                values['source'].append(float(F.cross_entropy(sz,truth,reduction='sum')))
                del sz16,sz
            z16=model.lm_head(hidden[:,start:end])
            if z16.dtype!=torch.float16:raise ValueError('Base head must compute FP16')
            z=z16.float().flatten(0,1)
            if not torch.isfinite(z).all():raise ValueError('Nonfinite base logits')
            scaled=z*beta
            base_ce=float(F.cross_entropy(z,truth,reduction='sum'))
            scaled_ce=float(F.cross_entropy(scaled,truth,reduction='sum'))
            if not math.isfinite(base_ce) or not math.isfinite(scaled_ce):raise ValueError('Nonfinite target CE')
            if beta==1. and not same_float(base_ce,scaled_ce):raise ValueError('Beta1 native CE control differs')
            values['base'].append(base_ce);values['scaled'].append(scaled_ce)
            argmax_changes+=int((z.argmax(-1)!=scaled.argmax(-1)).sum())
            if derivatives:
                ce,g,h,error=scalar_moments(z,truth,beta)
                # Native FP32 token statistics, FP64 accumulation; no vocabulary cache retained.
                per_ce.append(float(ce.double().sum()));gparts.append(float(g.double().sum()));hparts.append(float(h.double().sum()))
                maximum_identity_error=max(maximum_identity_error,error)
                del ce,g,h
            del z16,z,scaled
    if argmax_changes:raise ValueError('Positive scalar changed native argmax through numerical rounding')
    count=targets.numel()
    result={'target_tokens':count,'chunk_target_counts':[min(chunk_tokens,count-s) for s in range(0,count,chunk_tokens)],
        'ce_chunk_sums':values,'nll':{k:math.fsum(v) for k,v in values.items() if v},
        'argmax_changes':0,'argmax_checked_targets':count,'beta1_native_ce_exact':True if beta==1. else None}
    if derivatives:
        result['moments']={'per_target_ce_sum':math.fsum(per_ce),'g_sum':math.fsum(gparts),
            'h_sum':math.fsum(hparts),'maximum_gradient_identity_error':maximum_identity_error}
    return result


def aggregate(rows,arm):
    count=sum(r['target_tokens'] for r in rows);nll=math.fsum(r['nll'][arm] for r in rows)
    return {'windows':len(rows),'target_tokens':count,'nll':nll,'mean_nll':nll/count,'ppl':math.exp(nll/count)}


def verify_and_prepare():
    protocol=ROOT/'docs/SCALAR_TEMPERATURE_PROTOCOL.md'
    baseline_path=ROOT/'reports/small_compensation_v1_eval.json'
    pins={protocol:PROTOCOL_SHA,baseline_path:BASE_REPORT_SHA,
        ROOT/'artifacts/e8w5_v1/manifest.json':native.PARENT_SHA,
        ROOT/'artifacts/small_compensation_v1/manifest.json':native.SMALL_SHA,
        ROOT/'artifacts/small_compensation_v1/other_fp16.pt':native.SMALL_VALUES_SHA}
    pins.update({ROOT/name:digest for name,digest in {**CODE,**prepared.BOUND}.items()})
    for path,digest in pins.items():
        if native.sha(path)!=digest:raise ValueError(f'Pinned scalar input/code differs:{path}')
    prior=json.loads(baseline_path.read_text())
    pins[ROOT/'scripts/audit_decoded.py']=prior['loaded_tensor_audit_source_sha256']
    if native.sha(ROOT/'scripts/audit_decoded.py')!=pins[ROOT/'scripts/audit_decoded.py']:
        raise ValueError('Frozen loaded-tensor helper differs')
    parent,small,base_receipt=native.small_overlay.verify_overlay(ROOT/'artifacts/e8w5_v1',
        ROOT/'artifacts/small_compensation_v1',initialization_dir=ROOT/'artifacts/norm_compensation_v1')
    for name,row in parent['files'].items():pins[ROOT/'artifacts/e8w5_v1'/name]=row['sha256']
    data,training,reserved=prepared.verify_data(ROOT/'training_data/teacher_kl_compensation_v1',DATA_SHA)
    for name in ('manifest.json','training_tokens.pt','heldout_tokens.pt'):
        path=ROOT/'training_data/teacher_kl_compensation_v1'/name;pins[path]=native.sha(path)
    tokenizer=native.SentencePieceTokenizer(ROOT/'models/source')
    ids,dataset=native.load_wikitext_tokens(tokenizer,'train',native.WIKITEXT_REVISION)
    if dataset!=data['dataset'] or tuple(ids.shape)!=(2533678,):raise ValueError('Original TRAIN stream changed')
    fit=training[list(TRAIN_INDICES)]
    fit_plan=[data['training_windows'][i] for i in TRAIN_INDICES]
    if len(set(TRAIN_INDICES))!=64 or tuple(data['unused_starts'])!=SCREEN_STARTS:raise ValueError('Fixed data selection differs')
    if any(not torch.equal(w,ids[p['start']:p['start']+2048]) for w,p in zip(fit,fit_plan)):
        raise ValueError('Selected TRAIN tokens differ from original stream')
    families={**data['excluded_starts'],'all_teacher_kl_training':data['training_starts'],
        'observed_teacher_kl_reserved':data['heldout_starts']}
    overlap={key:sum(max(0,min(s+2048,t+2048)-max(s,t)) for s in SCREEN_STARTS for t in starts)
        for key,starts in families.items()}
    if any(overlap.values()) or any(b<a+2048 for a,b in zip(SCREEN_STARTS,SCREEN_STARTS[1:])):
        raise ValueError('Seven-window screen overlaps previous fitting/observations')
    screen=[ids[s:s+2048] for s in SCREEN_STARTS]
    screen_plan=[{'start':s,'stored_tokens':2048,'target_tokens':2047,
        'token_sha256_int64le':native.token_digest(w.numpy())} for s,w in zip(SCREEN_STARTS,screen)]
    hashes={row['name']:row['decoded_fp16_sha256'] for row in prior['parent_loaded_tensor_audit']['tensors']}
    hashes.update(prior['small_loaded_tensor_audit']['small']['loaded_and_export_fp16_sha256'])
    if len(hashes)!=507:raise ValueError('Expected507 accepted base tensors')
    pins[Path(__file__).resolve()]=native.sha(__file__)
    metadata={'dataset':dataset,'fit_indices':list(TRAIN_INDICES),'fit_windows':fit_plan,
        'fit_targets_per_pass':131008,'passes':4,'total_fit_scoring_targets':524032,
        'derivative_update_target_exposures':393024,'screen_windows':screen_plan,'screen_targets':14329,
        'screen_overlap_by_prior_family_tokens':overlap,'reserved64_used_for_fitting':False,
        'training_data_manifest_sha256':DATA_SHA}
    del training,reserved,ids
    return prior,parent,small,base_receipt,fit,screen,metadata,hashes,{str(p):h for p,h in pins.items()}


def reload_export(directory,manifest_sha,binding):
    directory=Path(directory)
    if {p.name for p in directory.iterdir()}!={'beta.bin','manifest.json'}:raise ValueError('Unexpected scalar export files')
    path=directory/'manifest.json'
    if native.sha(path)!=manifest_sha:raise ValueError('Scalar manifest changed')
    manifest=json.loads(path.read_text())
    if (manifest.get('format')!='MAMBA2_SCALAR_TEMPERATURE_V1' or manifest.get('complete') is not True
            or manifest.get('binding')!=binding or manifest.get('updates')!=3
            or len(manifest.get('history',[]))!=3):raise ValueError('Scalar manifest provenance differs')
    entry=manifest['files']['beta.bin'];scalar=directory/'beta.bin'
    if entry['bytes']!=4 or native.sha(scalar)!=entry['sha256']:raise ValueError('Scalar bytes differ')
    beta=read_beta(scalar)
    if struct.pack('<f',beta).hex()!=entry['hex_bytes'] or beta!=manifest['beta_fp32']:
        raise ValueError('Decoded scalar differs from final export')
    return beta,manifest


def paired_summary(rows):
    results={name:aggregate(rows,key) for name,key in zip(ARMS,('source','base','scaled'))}
    source,base,scaled=(results[n]['ppl'] for n in ARMS)
    return {'arms':results,'scaled_vs_base_ppl_relative_change':scaled/base-1,
        'scaled_vs_source_ppl_relative_change':scaled/source-1,
        'meaningful_improvement_reference':{'required_reduction':.01,'met':scaled<=.99*base},
        'source_plus5_percent_reference':{'maximum_ppl':1.05*source,'met':scaled<=1.05*source},
        'argmax_checked_targets':sum(r['argmax_checked_targets'] for r in rows),
        'argmax_changes':sum(r['argmax_changes'] for r in rows)}


def run(args,report):
    prior,parent,small,base_receipt,fit,screen,data,hashes,checked=verify_and_prepare()
    binding={'protocol_sha256':PROTOCOL_SHA,'source_checkpoint_sha256':native.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256':native.TOKENIZER_SHA256,'parent_manifest_sha256':native.PARENT_SHA,
        'small_manifest_sha256':native.SMALL_SHA,'small_values_sha256':native.SMALL_VALUES_SHA,
        'baseline_report_sha256':BASE_REPORT_SHA,'training_data_manifest_sha256':DATA_SHA,
        'checked_input_sha256':checked,'script_sha256':native.sha(__file__)}
    report.update(binding=binding,data=data,fit_passes=[],solver_history=[],base_package=base_receipt)
    native.write_json(args.report,report)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
    report['environment']=native.environment_receipt();report['reproducibility_before']=reproducibility_snapshot()
    model=native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    native.small_overlay.apply_overlay(model,ROOT/'artifacts/small_compensation_v1',small)
    model.eval().requires_grad_(False)
    report['base_coverage']=native.audit_parameter_coverage(model)
    report['base507_before_audit']=native.audit_frozen_values(model,hashes)
    beta=1.;source=None;source_hashes=None
    try:
        for pass_number in range(4):
            record={'pass':pass_number,'beta_fp32':beta,'derivatives_used_for_update':pass_number<3,'windows':[]}
            report['fit_passes'].append(record)
            for i,(window,plan) in enumerate(zip(fit,data['fit_windows'])):
                row=score_window(model,window,beta,derivatives=pass_number<3)
                row.update(plan,index=i);record['windows'].append(row)
                if (i+1)%16==0:
                    print(f'[scalar fit] pass{pass_number+1}/4 {i+1}/64 beta={beta:.9g}',flush=True)
                    native.write_json(args.report,report)
            record['summary']=aggregate(record['windows'],'scaled')
            if record['summary']['target_tokens']!=131008:raise ValueError('Fixed fitting pass count differs')
            if pass_number<3:
                g=math.fsum(r['moments']['g_sum'] for r in record['windows'])/131008
                h=math.fsum(r['moments']['h_sum'] for r in record['windows'])/131008
                beta,step=next_beta(beta,g,h);report['solver_history'].append({'update':pass_number+1,**step})
            native.write_json(args.report,report)
        initial=report['fit_passes'][0]['summary']['nll'];final=report['fit_passes'][3]['summary']['nll']
        report['final_train_check']={'initial_nll':initial,'final_nll':final,'met':final<=initial}
        args.out_dir.mkdir(parents=True,exist_ok=False)
        scalar=write_beta(args.out_dir/'beta.bin',beta)
        manifest={'format':'MAMBA2_SCALAR_TEMPERATURE_V1','complete':True,'binding':binding,
            'updates':3,'history':report['solver_history'],'beta_fp32':beta,'files':{'beta.bin':scalar},
            'data':data,'fit_pass_summaries':[r['summary'] for r in report['fit_passes']],
            'final_train_check':report['final_train_check'],'base507_fp16_sha256':hashes,
            'representation':'Native unchanged FP16 backbone/head; cast logits to FP32 and multiply by stored FP32 beta.',
            'storage':{'scalar_payload_bytes':4,'base_logical_data_bytes':base_receipt['logical_candidate_data_bytes'],
                'logical_data_plus_scalar_bytes':base_receipt['logical_candidate_data_bytes']+4,
                'manifest_bytes':0,'new_implementation_source_bytes':Path(__file__).stat().st_size,
                'complete_distribution_built':False,'scope':'Scalar plus measured metadata; no recomposed distribution or changed weight bytes.'}}
        while True:
            native.write_json(args.out_dir/'manifest.json',manifest)
            size=(args.out_dir/'manifest.json').stat().st_size
            if manifest['storage']['manifest_bytes']==size:break
            manifest['storage']['manifest_bytes']=size
        manifest_sha=native.sha(args.out_dir/'manifest.json')
        beta,restored=reload_export(args.out_dir,manifest_sha,binding)
        report['scalar_export']={'directory':str(args.out_dir),'manifest_sha256':manifest_sha,
            'manifest_bytes':(args.out_dir/'manifest.json').stat().st_size,'scalar':scalar,
            'reloaded_beta_fp32':beta,'roundtrip_exact':True,'storage':restored['storage']}
        native.write_json(args.report,report)
        if final>initial:
            report.update(complete=True,stopped_reason='Final TRAIN CE worsened; screen not evaluated.');return
        # The source model is first loaded only after fixed fitting and final scalar reload.
        source=native.load_source_model(ROOT/'models/source').eval().requires_grad_(False)
        report['source_package_receipt']=source._package_receipt
        report['source_coverage']=native.audit_parameter_coverage(source)
        source_hashes={n:native.tensor_sha_fp16(v) for n,v in source.named_parameters()}
        report['source507_before_sha256']=source_hashes
        screen_rows=report['screen']['rows']=[]
        for window,plan in zip(screen,data['screen_windows']):
            row=score_window(model,window,beta,source=source);row.update(plan);screen_rows.append(row)
            native.write_json(args.report,report)
        report['screen'].update(performed=True,summary=paired_summary(screen_rows))
        report['screen']['passed']=report['screen']['summary']['meaningful_improvement_reference']['met']
        native.write_json(args.report,report)
        if not report['screen']['passed']:
            report.update(complete=True,stopped_reason='Seven-window PPL screen failed; no validation loaded.');return
        tokenizer=native.SentencePieceTokenizer(ROOT/'models/source')
        ids,dataset=native.load_wikitext_tokens(tokenizer,'validation',native.WIKITEXT_REVISION)
        windows=native.ppl_windows(ids,2048,None);plan=native.window_plan(windows)
        if dataset!=prior['dataset'] or plan!=prior['window_plan']:raise ValueError('Established validation identities differ')
        beta,_=reload_export(args.out_dir,manifest_sha,binding)
        report['full_validation'].update(performed=True,dataset=dataset,window_plan=plan,rows=[])
        rows=report['full_validation']['rows']
        for i,((_,window),expected) in enumerate(zip(windows,plan)):
            row=score_window(model,window,beta,source=source);row.update(expected);rows.append(row)
            if (i+1)%8==0:native.write_json(args.report,report)
        summary=paired_summary(rows)
        if summary['arms'][ARMS[0]]['target_tokens']!=264764 or len(rows)!=130:raise ValueError('Full validation target count differs')
        for name,key in zip(ARMS,('source','base','scaled')):
            check={**summary['arms'][name],'execution':'prefill','logits_chunk_tokens':64,
                'windows':[{**expected,'nll':row['nll'][key],
                    'ppl':math.exp(row['nll'][key]/row['target_tokens'])} for expected,row in zip(plan,rows)]}
            native.check_arm(check,plan)
        report['full_validation']['summary']=summary
        report['full_validation']['historical_baseline_ppl_drift']={
            'source':summary['arms'][ARMS[0]]['ppl']/prior['results']['source_fp16']['ppl']-1,
            'accepted_all_small':summary['arms'][ARMS[1]]['ppl']/prior['results']['small_e8w5']['ppl']-1}
        report.update(complete=True,stopped_reason='Fixed scalar experiment complete; no promotion or publication.')
    finally:
        report['base507_after_audit']=native.audit_frozen_values(model,hashes)
        if source is not None and source_hashes is not None:
            report['source507_after_audit']=native.audit_frozen_values(source,source_hashes)
        report['reproducibility_after']=reproducibility_snapshot()
        for path,digest in checked.items():
            if native.sha(path)!=digest:raise ValueError(f'Bound scalar input changed:{path}')
        if 'scalar_export' in report:reload_export(args.out_dir,report['scalar_export']['manifest_sha256'],binding)
        report['gpu_memory']=native.gpu_memory_receipt()


def self_test():
    torch.set_num_threads(2);torch.manual_seed(928)
    if torch.cuda.is_initialized():raise ValueError('CPU self-test must not initialize CUDA')
    logits=torch.randn(7,19,dtype=torch.float64);targets=torch.arange(7)
    for value in (fp32(.87),1.,fp32(1.13)):
        ce,g,h,_=scalar_moments(logits,targets,value)
        beta=torch.tensor(value,dtype=torch.float64,requires_grad=True)
        loss=F.cross_entropy(logits*beta,targets,reduction='sum')
        first,=torch.autograd.grad(loss,beta,create_graph=True);second,=torch.autograd.grad(first,beta)
        torch.testing.assert_close(g.sum(),first,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(h.sum(),second,atol=1e-12,rtol=1e-12)
        pieces=[scalar_moments(logits[i:i+3],targets[i:i+3],value) for i in range(0,7,3)]
        for index,reference in [(0,ce),(1,g),(2,h)]:
            assert math.isclose(math.fsum(float(p[index].sum()) for p in pieces)/7,float(reference.mean()),abs_tol=1e-12)
    for bad in (0.,-1.,float('nan'),float('inf'),.79,1.21,True):
        try:beta_value(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid beta accepted')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'beta.bin';value=fp32(.9734567);receipt=write_beta(path,value)
        assert receipt['bytes']==4 and path.read_bytes()==struct.pack('<f',value) and read_beta(path)==value
        path.write_bytes(b'bad')
        try:read_beta(path)
        except ValueError:pass
        else:raise AssertionError('Invalid scalar bytes accepted')
    assert next_beta(1.,10.,1.)[0]==fp32(.95) and next_beta(1.,-10.,1.)[0]==fp32(1.05)
    assert next_beta(BETA_MIN,10.,1.)[0]==BETA_MIN and next_beta(BETA_MAX,-10.,1.)[0]==BETA_MAX
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Module()
            self.backbone.embedding=torch.nn.Embedding(19,4,dtype=torch.float16)
            self.backbone.forward=self.backbone.embedding.forward
            self.lm_head=torch.nn.Linear(4,19,bias=False,dtype=torch.float16)
    toy=Toy().eval().requires_grad_(False)
    row=score_window(toy,torch.arange(8),1.,derivatives=True,source=toy,chunk_tokens=3)
    assert row['chunk_target_counts']==[3,3,1] and row['target_tokens']==7 and row['beta1_native_ce_exact']
    assert row['ce_chunk_sums']['base']==row['ce_chunk_sums']['scaled']==row['ce_chunk_sums']['source']
    assert row['argmax_changes']==0
    return {'complete':True,'passed':True,'cuda_initialized':False,'script_sha256':native.sha(__file__),
        'protocol_sha256':PROTOCOL_SHA,'checks':['General-beta analytic gradient/curvature versus FP64 autograd at3fixed CPU fixture points',
            'Short3+3+1 tail token weighting','Invalid beta rejection','Exact4-byte FP32 serialization and malformed file rejection',
            'Clipped Newton step and FP32 bounds','Native FP16 toy beta1 CE/argmax and paired-source identity'],
        'limitations':'CPU implementation checks only; fixture beta points are not a model-temperature sweep.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--report',type=Path)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/scalar_temperature_v1')
    args=parser.parse_args()
    if args.self_test:
        if args.report is not None:parser.error('Self-test prints receipt; omit report')
        print(json.dumps(self_test(),indent=2));return
    if args.report is None:parser.error('--report required')
    args.report=args.report.resolve();args.out_dir=args.out_dir.resolve()
    if (args.report.parent!=ROOT/'reports' or args.out_dir.parent!=ROOT/'artifacts'
            or args.out_dir.exists() or args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists()):
        raise ValueError('Require fresh report and artifact directories; no overwrite or resume')
    torch.set_num_threads(8);started=time.perf_counter()
    report={'format':'MAMBA2_SCALAR_TEMPERATURE_EXPERIMENT_V1','complete':False,'candidate_advancement':False,
        'publication_performed':False,'checkpoint_selection':False,'solver_updates':3,'fit_passes_required':4,
        'screen':{'performed':False,'passed':None},'full_validation':{'performed':False},
        'script_sha256':native.sha(__file__),'protocol_sha256':PROTOCOL_SHA,
        'limitations':['Only one accepted-base scalar; no model-weight or recurrent-state change.',
            'The7-window screen is small; full validation is conditional and development-informed.',
            'Argmax equality is a numerical control, not MK validation.',
            'No newly composed distribution or compressed-residency claim.']}
    try:run(args,report)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;native.write_json(args.report,report)
    print(json.dumps({'complete':True,'report':str(args.report),'sha256':native.sha(args.report),
        'final_train_check':report['final_train_check'],'screen':{k:v for k,v in report['screen'].items() if k!='rows'},
        'full_validation_performed':report['full_validation']['performed']}),flush=True)


if __name__=='__main__':main()
