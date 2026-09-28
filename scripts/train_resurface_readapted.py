#!/usr/bin/env python3
"""Fixed1536-step, TRAIN-only soft readout repair of the accepted readapted base."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mamba_e8w5 import axis_small_base as base
from mamba_e8w5 import resurface_data as data
from mamba_e8w5 import resurface_native as native
from mamba_e8w5 import resurface_loss as loss_helper
from evaluate_axis_small import verify_export as verify_base_export
from train_norm_compensation import save_checkpoint,cpu_tree,tree_digest,scaler_for

PROTOCOL_SHA='469063147d892282a2bdad8558afca8466d6f365f1efe14ecbba7417e9857dd6'
BASE_MANIFEST_SHA='3ede80b2883d8b8987cb3e2198fd8badc2dbbb5af03c8fe6cac7f5f9f5b62047'
BASE_EVAL_SHA='3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4'
TRAIN_MANIFEST_SHA='dfe076ef18d5b0610e016d41d00a38948f3878b7c67d9cd8f6f46237e09d7e11'
CHECKPOINT_FORMAT='MAMBA2_RESURFACE_READAPTED_TRAINING_CHECKPOINT_V1'
REPORT_FORMAT='MAMBA2_RESURFACE_READAPTED_TRAINING_REPORT_V1'
MANIFEST_FORMAT='MAMBA2_RESURFACE_READAPTED_OVERLAY_V1'
STEPS=1536
PARAMETERS=1154104
CODE_PINS={
 'mamba_e8w5/axis_small_base.py':'4e7026475fc97b005105b27043a15bae83c576934612b1e346576438eef4da70',
 'scripts/evaluate_axis_small.py':'f2c2309f1dee45d32b8c4fcb9114a8d0f594cb56277b3a194a039070c361ac4c',
 'scripts/train_norm_compensation.py':'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
 'mamba_e8w5/resurface_data.py':'e813a382c25e99529eaddedb78aecd122b3c875903831f485c1cdf054ef7760c',
 'mamba_e8w5/resurface_native.py':'2dd08c7ee8958c832f0ae9da7cf5f261dceeff3e528e493c905c7c52df7166d7',
 'mamba_e8w5/resurface_loss.py':'ec045368583b6e49f484341b2d718658bb59e8d11050710377742752c1c67294',
}
HYPERPARAMETERS={
 'steps':1536,'maximum_attempts':1544,'maximum_overflow_retries':8,'seed':2026092803,
 'gate_mode':'soft','layers':56,'tensors':224,'parameters':PARAMETERS,'master_dtype':'float32','forward_dtype':'float16',
 'initialization':{'V_read':0.,'g_read':1.,'router_w':0.,'router_b':-4.},
 'optimizer':'AdamW','mix_lr':1e-4,'router_lr':3e-4,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,
 'gradient_clip_norm':1.,'lr_multiplier':'0.1+0.9*(1+cos(pi*j/1535))/2; j=successful_updates_before',
 'mk_ce_coefficient':1.,'prose_ce_coefficient':.5,'prose_teacher_kl_coefficient':.5,'temperature':1.,
 'prose_closure_coefficient':3.,'closure_budget':.006,'closure_budget_coefficient':10.,'opening_coefficient':0.,
 'prose_stored_tokens':512,'prose_targets':511,'full_vocabulary':256000,'logits_chunk_tokens':64,
 'loss_scaler':{'init_scale':1024.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000},
 'checkpoint_interval':384,'checkpoint_each_block':True,'selection':'final1536 only','resume':False,
 'teacher':'separately loaded identical compressed axis plus final448 small-tensor model; no adapter',
}
STOP=False


def sha(path):return native.file_hash(path)


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def receipt(path):
    path=Path(path);return {'file':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def schedule():
    return torch.randperm(STEPS,generator=torch.Generator(device='cpu').manual_seed(2026092803)).tolist()


def inventory():
    shapes={'V_read':[128,128],'g_read':[128],'router_w':[4096],'router_b':[]}
    return {f'layer{i}.{name}':{'shape':shape,'numel':math.prod(shape)} for i in range(56) for name,shape in shapes.items()}


def lr_factor(j):
    if type(j) is not int or not 0<=j<STEPS:raise ValueError('Invalid successful-update index')
    return .1+.9*(1+math.cos(math.pi*j/(STEPS-1)))/2


def pair_metadata(inputs,j):
    index=inputs['schedule'][j];mk=inputs['mk_tokens'][index]
    return {'schedule_entry':index,'mk_case_id':mk['id'],'mk_targets':mk['answer_target_count'],
        'prose_window_index':inputs['prose_schedule'][j%448],'prose_start':512*((j//448)%4),'prose_targets':511}


def make_pair(inputs,j,device='cpu'):
    meta=pair_metadata(inputs,j);row=inputs['mk_tokens'][meta['schedule_entry']]
    sequence=torch.tensor(row['full_ids'],dtype=torch.long,device=device)[None]
    mask=torch.zeros_like(sequence[:,:-1],dtype=torch.bool)
    mask[:,row['ce_hidden_positions']]=True
    if int(mask.sum())!=meta['mk_targets']:raise ValueError('Full numeric suffix mask count differs')
    start=meta['prose_start'];window=inputs['prose_windows'][meta['prose_window_index'],start:start+512]
    if tuple(window.shape)!=(512,):raise ValueError('Prose segment exceeds prepared2048-token window')
    return {'metadata':meta,'mk_ids':sequence[:,:-1],'mk_targets':sequence[:,1:],'answer_mask':mask,
        'prose_ids':window[None,:-1].to(device),'prose_targets':window[None,1:].to(device)}


def verify_inputs(args):
    if args.train_manifest_sha256!=TRAIN_MANIFEST_SHA:raise ValueError('Literal frozen prepared TRAIN manifestSHA is required')
    pins={ROOT/name:digest for name,digest in CODE_PINS.items()}
    pins.update({ROOT/'docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md':PROTOCOL_SHA,
        ROOT/'reports/axis_small_readaptation_v1_eval.json':BASE_EVAL_SHA,
        ROOT/'artifacts/axis_small_readaptation_v1/manifest.json':BASE_MANIFEST_SHA,
        Path(__file__):sha(__file__)})
    for path,digest in pins.items():
        if digest.startswith('PENDING') or path.is_symlink() or not path.is_file() or sha(path)!=digest:
            raise ValueError(f'Frozen training input differs:{path}')
    context=base.verify_inputs()
    args_base=argparse.Namespace(overlay_dir=ROOT/'artifacts/axis_small_readaptation_v1',
        training_report=ROOT/'reports/axis_small_readaptation_v1_train.json',training_checkpoint=None,
        smoke_report=ROOT/'reports/axis_small_readaptation_v1_smoke.json',
        diagnostic_report=ROOT/'reports/axis_small_diagnostic_v1.json')
    _manifest,values,proof=verify_base_export(args_base,context)
    prior=json.loads((ROOT/'reports/axis_small_readaptation_v1_eval.json').read_text())
    if prior.get('complete') is not True or prior['integrity']!=proof:raise ValueError('Accepted base differs from completed PPL evidence')
    tokenizer=base.runtime.SentencePieceTokenizer(ROOT/'models/source')
    manifest,rows,tokens=data.load_training(args.data_root,args.train_manifest_sha256,tokenizer)
    if manifest['protocol_sha256']!=PROTOCOL_SHA or manifest['row_count']!=STEPS or manifest['proof']['normal_count']!=STEPS:
        raise ValueError('Training data protocol/count differs')
    if manifest['answer_target_tokens']!=sum(row['answer_target_count'] for row in tokens):raise ValueError('Prepared answer exposure count differs')
    train_path=Path(args.data_root)/'train'
    pins[train_path/'manifest.json']=args.train_manifest_sha256
    for name,row in manifest['files'].items():pins[train_path/name]=row['sha256']
    for name,digest in proof['checked_input_sha256'].items():
        path=Path(name)
        if path in pins and pins[path]!=digest:raise ValueError('Contradictory base/code input identity')
        pins[path]=digest
    prose=context['training_windows'];del context['heldout_windows']
    prose_schedule=context['data_manifest']['schedule']
    if tuple(prose.shape)!=(448,2048) or sorted(prose_schedule)!=list(range(448)):
        raise ValueError('Prepared448 prose windows/schedule differ')
    binding={'protocol_sha256':PROTOCOL_SHA,'base_manifest_sha256':BASE_MANIFEST_SHA,'base_evaluation_sha256':BASE_EVAL_SHA,
        'base_integrity':proof,'base_shared_binding':context['binding'],
        'training_data_manifest_sha256':args.train_manifest_sha256,'training_data_files':manifest['files'],
        'prose_data_manifest_sha256':context['binding']['data_manifest_sha256'],
        'prose_training_tokens_file_sha256':context['binding']['training_tokens_file_sha256'],
        'tokenizer_sha256':tokenizer.sha256,'hyperparameters':HYPERPARAMETERS,
        'training_code_sha256':{**CODE_PINS,'scripts/train_resurface_readapted.py':sha(__file__)},
        'checked_input_sha256':{str(path.resolve()):digest for path,digest in pins.items()}}
    result={'axis_context':context,'base_values':values,'base_expected507':proof['resolved507_candidate_fp16_sha256'],
        'base_integrity':proof,'mk_manifest':manifest,'mk_rows':rows,'mk_tokens':tokens,'prose_windows':prose,
        'prose_schedule':prose_schedule,'schedule':schedule(),'binding':binding}
    # Exact per-pair metadata can be audited later without loading any TRAIN text.
    result['pair_plan']=[pair_metadata(result,j) for j in range(STEPS)]
    return result


def final_recheck(inputs):
    base.final_recheck(inputs['axis_context'])
    for path,digest in inputs['binding']['checked_input_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Bound input changed:{path}')


def identities(model):
    return {name:(id(p),p.data_ptr(),p._version) for name,p in model.named_parameters()}


def assert_identity(model,expected):
    if identities(model)!=expected or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen model identity/storage/version/gradient changed')


def load_models_and_bank(inputs):
    models=[];audits={}
    for name in ('student','teacher'):
        model=base.load_model(inputs['axis_context']).eval().requires_grad_(False)
        base.install_small(model,inputs['base_values'],inputs['base_expected507'])
        audits[name+'507']=base.audit_model(model,inputs['base_expected507'])
        models.append(model)
    student,teacher=models
    bank=native.ResurfaceNative(student,'soft',expected_base_hashes=inputs['base_expected507'])
    if len(bank.masters)!=224 or sum(p.numel() for p in bank.parameters())!=PARAMETERS:raise RuntimeError('Adapter capacity differs')
    for name,p in bank.masters.items():
        if not torch.all(p==HYPERPARAMETERS['initialization'][name.rsplit('.',1)[1]]):raise RuntimeError('Cold initialization differs')
    return student,teacher,bank,audits


def optimizer_for(bank):
    mix=[p for name,p in bank.masters.items() if name.endswith(('.V_read','.g_read'))]
    routers=[p for name,p in bank.masters.items() if name.endswith(('.router_w','.router_b'))]
    return torch.optim.AdamW([{'params':mix,'lr':1e-4,'base_lr':1e-4},
        {'params':routers,'lr':3e-4,'base_lr':3e-4}],betas=(.9,.999),eps=1e-8,weight_decay=0.)


def gradient_receipt(bank):
    families={};count=0;nonzero=0;finite=True
    for name,p in bank.masters.items():
        if p.grad is None or p.grad.dtype!=torch.float32:raise RuntimeError(f'Missing FP32 adapter gradient:{name}')
        g=p.grad;ok=bool(torch.isfinite(g).all());nz=int(torch.count_nonzero(g));family=name.rsplit('.',1)[1]
        row=families.setdefault(family,{'tensors':0,'parameters':0,'nonzero':0,'finite':True})
        row['tensors']+=1;row['parameters']+=g.numel();row['nonzero']+=nz;row['finite'] &=ok
        count+=g.numel();nonzero+=nz;finite &=ok
    return {'tensors':len(bank.masters),'parameters':count,'finite':finite,'nonzero':nonzero,'families':families}


def attempt_update(bank,teacher,inputs,j,optimizer,scaler,teacher_identity=None):
    optimizer.zero_grad(set_to_none=True)
    multiplier=lr_factor(j)
    for group in optimizer.param_groups:group['lr']=group['base_lr']*multiplier
    pair=make_pair(inputs,j,'cuda');scale=scaler.get_scale()
    torch.cuda.synchronize();started=time.monotonic()
    hidden,gates=bank.forward_hidden(pair['mk_ids'],use_checkpoint=True)
    mk=loss_helper.staged_loss_backward(hidden,pair['mk_targets'],bank.model.lm_head,gates=gates,
        scaler=scaler,answer_mask=pair['answer_mask'],ce_weight=1.,kl_weight=0.,closure_weight=0.,chunk_tokens=64)
    del hidden,gates
    with torch.no_grad():teacher_hidden=teacher.backbone(pair['prose_ids'])
    hidden,gates=bank.forward_hidden(pair['prose_ids'],use_checkpoint=True)
    prose=loss_helper.staged_loss_backward(hidden,pair['prose_targets'],bank.model.lm_head,gates=gates,
        scaler=scaler,teacher_hidden=teacher_hidden,teacher_head=teacher.lm_head,
        ce_weight=.5,kl_weight=.5,closure_weight=3.,chunk_tokens=64)
    del hidden,gates,teacher_hidden
    scaler.unscale_(optimizer);gradients=gradient_receipt(bank)
    overflow=(not gradients['finite'] or not mk['scaled_hidden_gradient_finite'] or not prose['scaled_hidden_gradient_finite'])
    if overflow:
        masters_before=tree_digest(bank.state_dict());optimizer_before=tree_digest(optimizer.state_dict())
        scaler.update(new_scale=scale*.5)
        if masters_before!=tree_digest(bank.state_dict()) or optimizer_before!=tree_digest(optimizer.state_dict()):
            raise RuntimeError('Overflow mutated masters/optimizer')
        magnitude=None
    else:
        if gradients['nonzero']==0:raise RuntimeError('All adapter gradients are zero')
        magnitude=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update()
    for p in bank.parameters():
        if not torch.isfinite(p).all() or not torch.isfinite(p.half()).all():raise FloatingPointError('Nonfinite master/FP16 cast')
    bank.assert_base_frozen()
    if teacher_identity is not None:assert_identity(teacher,teacher_identity)
    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
    return {**pair['metadata'],'mk':mk,'prose':prose,'gradients':gradients,'overflow':overflow,
        'loss_scale_before':scale,'loss_scale_after':scaler.get_scale(),'gradient_norm_before_clip':magnitude,
        'mix_lr':1e-4*multiplier,'router_lr':3e-4*multiplier,'lr_multiplier':multiplier,
        'overflow_optimizer_and_masters_unchanged':True if overflow else None,'seconds':time.monotonic()-started}


def accounting(history,plan):
    success=overflow=0;mk_success=mk_attempt=0
    for number,row in enumerate(history,1):
        if (success>=STEPS or type(row.get('attempt')) is not int or row['attempt']!=number
                or type(row.get('successful_updates_after')) is not int or type(row.get('overflow')) is not bool
                or any(type(row.get(k)) is not type(v) or row[k]!=v for k,v in plan[success].items())):
            raise ValueError('Fixed pair schedule/same-window retry accounting differs')
        mk_attempt+=row['mk_targets']
        if row['overflow']:overflow+=1
        else:success+=1;mk_success+=row['mk_targets']
        if row['successful_updates_after']!=success:raise ValueError('History success count differs')
    if overflow>8 or len(history)>1544:raise ValueError('Overflow/attempt budget exceeded')
    return {'successful_updates':success,'attempts':len(history),'overflow_retries':overflow,
        'successful_mk_answer_targets':mk_success,'attempted_mk_answer_targets':mk_attempt,
        'successful_prose_targets':success*511,'attempted_prose_targets':len(history)*511}


class TrainingRun:
    def __init__(self,inputs,student,teacher,bank,args,report):
        self.inputs,self.student,self.teacher,self.bank=inputs,student,teacher,bank
        self.args,self.report=args,report;self.optimizer=optimizer_for(bank);self.scaler=scaler_for()
        self.history=[];self.success=0;self.final_checkpoint=None;self.teacher_identity=identities(teacher)

    def journal(self,row):
        with (self.args.work_dir/'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**row},allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())

    def checkpoint(self,reason):
        if reason=='failure':
            counts={'successful_updates':self.success,'attempts':len(self.history),
                'overflow_retries':sum(row['overflow'] for row in self.history),
                'successful_mk_answer_targets':sum(row['mk_targets'] for row in self.history if not row['overflow']),
                'attempted_mk_answer_targets':sum(row['mk_targets'] for row in self.history),
                'successful_prose_targets':self.success*511,'attempted_prose_targets':len(self.history)*511}
        else:counts=accounting(self.history,self.inputs['pair_plan'])
        payload={'format':CHECKPOINT_FORMAT,'binding':self.inputs['binding'],'smoke_report_sha256':self.report['smoke_report_sha256'],
            'smoke_source_sha256':self.report['smoke_source_sha256'],
            'schedule':self.inputs['schedule'],'pair_plan':self.inputs['pair_plan'],'history':self.history,**counts,
            'masters':self.bank.state_dict(),'optimizer':cpu_tree(self.optimizer.state_dict()),'scaler':self.scaler.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'base507_sha256':self.inputs['base_expected507'],'teacher507_sha256':self.inputs['base_expected507'],
            'reason':reason,'resumable':False}
        path=self.args.work_dir/f"checkpoint_attempt{counts['attempts']:04d}_step{self.success:04d}_{reason}.pt"
        result=save_checkpoint(path,payload)
        if reason=='final':self.final_checkpoint=result
        return result

    def run(self):
        self.report.update(schedule=self.inputs['schedule'],pair_plan=self.inputs['pair_plan'],history=self.history)
        try:
            while self.success<STEPS:
                if STOP:raise RuntimeError('Stopped between attempts; no resume/replay')
                self.journal({'event':'attempt_start','attempt':len(self.history)+1,'successful_updates_before':self.success,
                    **self.inputs['pair_plan'][self.success]})
                value=attempt_update(self.bank,self.teacher,self.inputs,self.success,self.optimizer,self.scaler,self.teacher_identity)
                if not value['overflow']:self.success+=1
                row={'attempt':len(self.history)+1,'successful_updates_after':self.success,**value}
                self.history.append(row);self.journal({'event':'attempt_complete',**row})
                counts=accounting(self.history,self.inputs['pair_plan'])
                if not value['overflow'] and self.success%384==0:
                    self.report.setdefault('checkpoints',[]).append(self.checkpoint('final' if self.success==STEPS else 'periodic'))
                self.report.update(**counts,final_checkpoint=self.final_checkpoint,gpu_memory=base.runtime.gpu_memory_receipt())
                if len(self.history)%16==0 or value['overflow'] or self.success==STEPS:
                    write_json(self.args.report,self.report)
                    print(json.dumps({'updates':self.success,'attempts':len(self.history),'overflow':value['overflow'],
                        'mk':value['mk'],'prose':value['prose']},allow_nan=False),flush=True)
        except BaseException:
            self.report['failure_checkpoint']=self.checkpoint('failure');raise


@torch.no_grad()
def export_native_parity(bank,path,binding,ids,expected):
    functional,_=bank.forward_hidden(ids,use_checkpoint=False)
    function_logits=bank.model.lm_head(functional[:,-8:])
    first={'hidden':native.tensor_hash(functional),'logits':native.tensor_hash(function_logits)}
    if not torch.isfinite(functional).all() or not torch.isfinite(function_logits).all():raise FloatingPointError('Nonfinite functional export probe')
    bank.close()
    with native.install_fp16(bank.model,path,expected_binding=binding,expected_base_hashes=expected) as deployed:
        hidden=bank.model.backbone(ids);logits=bank.model.lm_head(hidden[:,-8:])
        second={'hidden':native.tensor_hash(hidden),'logits':native.tensor_hash(logits)}
        if not torch.isfinite(hidden).all() or not torch.isfinite(logits).all() or first!=second:
            raise RuntimeError('Actual reloaded native FP16 adapter differs from training forward')
        deployed.assert_base_frozen()
    return {'input_tokens':ids.numel(),'logit_positions':8,'functional':first,'native_reloaded':second,
        'hidden_bitwise_equal':True,'logits_bitwise_equal':True,'gate_mode':'soft'}


def finalize(run,inputs,args,report):
    counts=accounting(run.history,inputs['pair_plan'])
    if counts['successful_updates']!=STEPS or run.final_checkpoint is None:raise RuntimeError('Final1536 checkpoint missing')
    cp=run.final_checkpoint
    if sha(cp['file'])!=cp['sha256']:raise ValueError('Final checkpoint changed')
    saved=torch.load(cp['file'],map_location='cpu',weights_only=True)
    for key,value in {'format':CHECKPOINT_FORMAT,'binding':inputs['binding'],'reason':'final','resumable':False,
        'schedule':inputs['schedule'],'pair_plan':inputs['pair_plan'],'history':run.history,
        'smoke_report_sha256':report['smoke_report_sha256'],'smoke_source_sha256':report['smoke_source_sha256'],
        'base507_sha256':inputs['base_expected507'],
        'teacher507_sha256':inputs['base_expected507'],**counts}.items():
        if saved.get(key)!=value:raise ValueError(f'Final checkpoint proof differs:{key}')
    stage=args.work_dir/'final_export';stage.mkdir()
    path=stage/'adapter_fp16.pt';export=run.bank.export_fp16(path,binding=inputs['binding'])
    actual=native.read_fp16(path,inputs['binding'])
    if set(saved['masters'])!=set(inventory()) or set(actual['tensors'])!=set(inventory()):raise ValueError('Final224 inventory differs')
    for name,entry in inventory().items():
        master=saved['masters'][name];value=actual['tensors'][name]
        if (master.dtype!=torch.float32 or list(master.shape)!=entry['shape'] or not torch.isfinite(master).all()
                or native.tensor_hash(master.half())!=native.tensor_hash(value)
                or native.tensor_hash(run.bank.masters[name].half())!=native.tensor_hash(value)):
            raise ValueError(f'Final saved master→FP16 export differs:{name}')
    probe=make_pair(inputs,0,'cuda')['mk_ids'][:,:128]
    report['export_parity']=export_native_parity(run.bank,path,inputs['binding'],probe,inputs['base_expected507'])
    report['final_base507_audit']=base.audit_model(run.student,inputs['base_expected507'])
    assert_identity(run.teacher,run.teacher_identity)
    report['final_teacher507_audit']=base.audit_model(run.teacher,inputs['base_expected507'])
    final_recheck(inputs)
    if (sha(args.smoke_report)!=report['smoke_report_sha256']
            or sha(ROOT/'scripts/smoke_resurface_native.py')!=report['smoke_source_sha256']):
        raise ValueError('Passed smoke receipt/source changed')
    manifest={'format':MANIFEST_FORMAT,'complete':True,'gate_mode':'soft','variant':native.VARIANT,
        'binding':inputs['binding'],'files':{'adapter_fp16.pt':{'bytes':export['bytes'],'sha256':export['sha256']}},
        'tensors':inventory(),'fp16_tensor_sha256':export['tensor_sha256'],'base507_sha256':inputs['base_expected507'],
        'training_receipt':{**counts,'checkpoint_sha256':cp['sha256'],'checkpoint_bytes':cp['bytes'],
            'smoke_report_sha256':report['smoke_report_sha256'],'smoke_source_sha256':report['smoke_source_sha256']},
        'storage':{'base_logical_data_bytes':3138928792,'adapter_parameter_count':PARAMETERS,
            'fp16_adapter_payload_bytes':2*PARAMETERS,'adapter_file_bytes':export['bytes'],
            'base_plus_adapter_data_bytes':3138928792+export['bytes'],'manifest_bytes':0,'complete_distribution_built':False}}
    mp=stage/'manifest.json'
    while True:
        write_json(mp,manifest);size=mp.stat().st_size
        if manifest['storage']['manifest_bytes']==size:break
        manifest['storage']['manifest_bytes']=size
    if args.out_dir.exists():raise FileExistsError('Candidate output exists')
    stage.rename(args.out_dir)
    report.update(complete=True,final_step=STEPS,final_checkpoint=cp,manifest=receipt(args.out_dir/'manifest.json'),
        final_export={**export,'file':str(args.out_dir/'adapter_fp16.pt'),'final1536_master_to_fp16_bitwise_equal':True},
        storage=manifest['storage'],**counts)


def self_test():
    assert schedule()==schedule() and sorted(schedule())==list(range(STEPS))
    assert lr_factor(0)==1. and lr_factor(STEPS-1)==.1
    assert len(inventory())==224 and sum(x['numel'] for x in inventory().values())==PARAMETERS
    plan=[{'schedule_entry':index,'mk_case_id':f'toy{index}','mk_targets':3,'prose_window_index':j%448,
           'prose_start':512*((j//448)%4),'prose_targets':511} for j,index in enumerate(schedule())]
    rows=[{'attempt':j+1,'successful_updates_after':j+1,'overflow':False,**item} for j,item in enumerate(plan)]
    assert accounting(rows,plan)['successful_prose_targets']==784896
    retry=[{'attempt':j+1,'successful_updates_after':0,'overflow':True,**plan[0]} for j in range(8)]
    completed=[{**row,'attempt':row['attempt']+8} for row in rows]
    assert accounting(retry+completed,plan)['attempts']==1544
    for bad in ([{**rows[0],'prose_targets':True}], [{**rows[0],'schedule_entry':-1}],retry+[{**retry[-1],'attempt':9}]):
        try:accounting(bad,plan)
        except ValueError:pass
        else:raise AssertionError('Invalid fixed-pair accounting accepted')
    return {'passed':True,'tests':['fixed1536 permutation and cosineendpoints','224/1154104 inventory',
        'exact784896 prose exposures','8identicalpair retries/1544max','strict count and pair validation']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    p.add_argument('--data-root',type=Path,default=ROOT/'training_data/resurface_readapted_v1')
    p.add_argument('--train-manifest-sha256',required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--work-dir',type=Path);p.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/resurface_readapted_v1')
    p.add_argument('--smoke-report',type=Path)
    args=p.parse_args();torch.set_num_threads(8)
    for name in ('data_root','report','work_dir','out_dir','smoke_report'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    if not args.verify_only and (args.work_dir is None or args.smoke_report is None):p.error('Formaltrain requires fresh workdir and passed smokereport')
    if (args.report.exists() or args.report.with_suffix('.json.tmp').exists()
            or (args.work_dir is not None and args.work_dir.exists()) or (not args.verify_only and args.out_dir.exists())):
        raise FileExistsError('Fresh paths only; no overwrite/resume')
    def stop(_signal,_frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    report={'format':REPORT_FORMAT,'complete':False,'mode':'verify_only' if args.verify_only else 'train',
        'pid':os.getpid(),'started_at_unix':time.time(),'script_sha256':sha(__file__),'protocol_sha256':PROTOCOL_SHA,
        'gate_mode':'soft','mk_development_loaded':False,'confirmation_loaded':False,'validation_scored':False,
        'prose_heldout_verified_for_provenance_only':True,
        'selection':'final1536 only','resume':False}
    started=time.monotonic();bank=None
    try:
        report['self_test']=self_test();inputs=verify_inputs(args)
        report.update(binding=inputs['binding'],training_data_manifest=inputs['mk_manifest'],
            schedule=inputs['schedule'],pair_plan=inputs['pair_plan'],adapter_inventory=inventory())
        if args.verify_only:
            final_recheck(inputs)
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise RuntimeError('CPUpreflight must hide CUDA')
            report.update(complete=True,cuda_initialized=False);return
        smoke=json.loads(args.smoke_report.read_text())
        smoke_source=sha(ROOT/'scripts/smoke_resurface_native.py')
        if (smoke.get('format')!='MAMBA2_RESURFACE_NATIVE_SMOKE_V1' or smoke.get('mode')!='gpu_smoke'
                or smoke.get('complete') is not True or smoke.get('passed') is not True
                or smoke.get('script_sha256')!=smoke_source or smoke.get('protocol_sha256')!=PROTOCOL_SHA
                or smoke.get('binding')!=inputs['binding'] or smoke.get('trial_state_discarded') is not True):
            raise ValueError('Passed same-binding discarded smoke required')
        report['smoke_report_sha256']=sha(args.smoke_report);report['smoke_source_sha256']=smoke_source
        args.work_dir.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(2026092803);torch.cuda.manual_seed_all(2026092803)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats();report['environment']=base.runtime.environment_receipt()
        student,teacher,bank,audits=load_models_and_bank(inputs)
        report['initial_base507_audit']=audits['student507'];report['initial_teacher507_audit']=audits['teacher507']
        write_json(args.report,report)
        run=TrainingRun(inputs,student,teacher,bank,args,report);run.run();finalize(run,inputs,args,report)
        report['gpu_memory']=base.runtime.gpu_memory_receipt()
    except BaseException as error:
        report.update(complete=False,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if bank is not None:bank.close()
        report['elapsed_seconds']=time.monotonic()-started;write_json(args.report,report)


if __name__=='__main__':main()
