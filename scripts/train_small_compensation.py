#!/usr/bin/env python3
"""Fixed all393-small-tensor PPL compensation, starting from norm-v1 FP16."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F

from mamba_e8w5 import small_overlay as overlay,norm_overlay
from mamba_e8w5.small_training import SmallMasters,HYPERPARAMETERS,selected_inventory,training_schedule,family_ranges
from mamba_e8w5.norm_training import chunked_loss_backward,gradient_receipt,export_parity,tensor_hash
from mamba_e8w5.runtime import load_quantized_model,load_source_model,environment_receipt,gpu_memory_receipt,MODEL_CONFIG
from scripts.train_norm_compensation import (write_json,save_checkpoint,cpu_tree,tree_digest,parameter_hashes,
    optimizer_for,scaler_for,gradient_comparison,SMOKE_TOLERANCES)

STOP_REQUESTED=False


class RequestedStop(RuntimeError): pass


def request_stop(_signum,_frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def initialization():
    return {'norm_manifest_sha256':overlay.INITIAL_MANIFEST_SHA256,'other_fp16_sha256':overlay.INITIAL_OTHER_SHA256}


def verify_inputs(args):
    # Existing norm verifier checks raw117-file ledger, source, complete norm
    # overlay, all393 keys and the immutable old training implementation.
    raw,norm_manifest,_=norm_overlay.verify_overlay(args.parent_dir,args.initialization_dir)
    if (overlay.sha(args.initialization_dir/'manifest.json')!=overlay.INITIAL_MANIFEST_SHA256
            or overlay.sha(args.initialization_dir/'other_fp16.pt')!=overlay.INITIAL_OTHER_SHA256):
        raise ValueError('Not the declared norm-v1 FP16 initialization')
    data,windows=overlay.verify_training_data(args.data_dir)
    if overlay.sha(args.protocol)!=overlay.PROTOCOL_SHA256: raise ValueError('Protocol differs')
    binding={'source_checkpoint_sha256':overlay.SOURCE_CHECKPOINT_SHA256,'tokenizer_sha256':overlay.TOKENIZER_SHA256,
        'protocol_sha256':overlay.PROTOCOL_SHA256,'small_overlay_source_sha256':overlay.sha(overlay.__file__),
        'trainer_source_sha256':overlay.sha(__file__),
        'training_helper_source_sha256':overlay.sha(ROOT/'mamba_e8w5/small_training.py'),
        'norm_training_source_sha256':overlay.sha(ROOT/'mamba_e8w5/norm_training.py'),
        'norm_trainer_utility_source_sha256':overlay.sha(ROOT/'scripts/train_norm_compensation.py'),
        'norm_overlay_source_sha256':overlay.sha(norm_overlay.__file__),
        'training_data_manifest_sha256':overlay.sha(args.data_dir/'manifest.json'),
        'training_tokens_sha256':data['token_windows_file_sha256'],'hyperparameters':HYPERPARAMETERS,
        'frozen_code_sha256':raw['binding']['code_sha256'],
        'evaluation_source_sha256':overlay.sha(ROOT/'mamba_e8w5/evaluation.py')}
    return raw,norm_manifest,windows,binding


def attempt_update(bank,teacher,window,optimizer,scaler,detail=False):
    optimizer.zero_grad(set_to_none=True)
    ids,targets=window[None,:-1].cuda(),window[None,1:].cuda()
    with torch.no_grad(): teacher_hidden=teacher.backbone(ids)
    hidden=bank.forward(ids,True)
    scale=scaler.get_scale()
    values=chunked_loss_backward(hidden,teacher_hidden,targets,bank.model.lm_head,teacher.lm_head,scaler,64)
    del hidden,teacher_hidden
    scaler.unscale_(optimizer)
    gradients=gradient_receipt(bank)
    ranges=family_ranges(bank,True) if detail else None
    overflow=not gradients['finite'] or not values['scaled_hidden_gradient_finite']
    if not overflow and gradients['nonzero']==0: raise RuntimeError('All393 gradients are zero')
    if overflow:
        masters_before=tree_digest(bank.state_dict()); optimizer_before=tree_digest(optimizer.state_dict())
        scaler.update(new_scale=scale*.5)
        if masters_before!=tree_digest(bank.state_dict()) or optimizer_before!=tree_digest(optimizer.state_dict()):
            raise RuntimeError('Overflow mutated parameters or optimizer state')
        magnitude=None
    else:
        magnitude=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update()
    stability=bank.stability_receipt()
    bank.assert_frozen();optimizer.zero_grad(set_to_none=True)
    return {**values,'overflow':overflow,'loss_scale_before':scale,'loss_scale_after':scaler.get_scale(),
        'gradient_norm_before_clip':magnitude,'gradients':{k:v for k,v in gradients.items() if k!='per_tensor'},
        'gradient_family_ranges':ranges,'stability':stability,
        'overflow_optimizer_and_masters_unchanged':True if overflow else None}


def smoke(bank,teacher,windows,initial_other,args,report):
    ids=windows[0,None,:128].cuda()
    report['initial_parameter_ranges']=family_ranges(bank)
    report['initial_stability']=bank.stability_receipt()
    print('[small smoke] native norm-v1 versus initial functional forward',flush=True)
    with torch.no_grad():
        native=bank.model.backbone(ids);functional=bank.forward(ids,False)
        if not torch.equal(native,functional): raise RuntimeError('Initial norm-v1 hidden mismatch')
        if not torch.equal(bank.model.lm_head(native[:,-1:]),bank.model.lm_head(functional[:,-1:])):
            raise RuntimeError('Initial norm-v1 output mismatch')
    report['initial_forward']={'hidden_bitwise_equal':True,'logits_bitwise_equal':True,'input_tokens':128,'hidden_sha256':tensor_hash(native)}
    del native,functional
    initial=bank.state_dict();grads=[];hidden_hashes=[]
    with torch.no_grad():
        for p in bank.parameters(): p.add_(torch.linspace(-.003,.003,p.numel(),device=p.device).reshape(p.shape))
    print('[small smoke] perturbed all393 checkpoint and seven-family gradients',flush=True)
    for enabled in (False,True):
        for p in bank.parameters():p.grad=None
        h=bank.forward(ids,enabled);hidden_hashes.append(tensor_hash(h))
        tangent=torch.linspace(-1,1,h.shape[-1],device=h.device)
        (1024*(h.float()*tangent).mean()).backward()
        receipt=gradient_receipt(bank);families=family_ranges(bank,True)
        if not receipt['finite'] or receipt['tensors']!=393 or any(v['nonzero']==0 for v in families.values()):
            raise RuntimeError('Missing finite/nonzero gradient family in all-small smoke')
        grads.append({n:p.grad.cpu().clone() for n,p in bank.masters.items()})
        del h
    if hidden_hashes[0]!=hidden_hashes[1]:raise RuntimeError('Checkpoint forward mismatch')
    report['tolerances']=SMOKE_TOLERANCES
    report['checkpoint_parity']=gradient_comparison(*grads,rtol=SMOKE_TOLERANCES['checkpoint_gradient_rtol'],atol=SMOKE_TOLERANCES['checkpoint_gradient_atol'])
    report['checkpoint_parity'].update(hidden_bitwise_equal=True,gradient_coverage=receipt,gradient_family_ranges=families)
    bank.load_state_dict(initial)
    for p in bank.parameters():p.grad=None
    del grads
    torch.cuda.empty_cache()
    print('[small smoke] chunked complete-vocabulary objective and hidden gradient',flush=True)
    with torch.no_grad():sh=bank.forward(ids[:,:5],False);th=teacher.backbone(ids[:,:5])
    targets=windows[0,None,1:6].cuda()
    staged=sh.detach().clone().requires_grad_()
    chunk=chunked_loss_backward(staged,th,targets,bank.model.lm_head,teacher.lm_head,None,3)
    full=sh.detach().clone().requires_grad_();logits=bank.model.lm_head(full).float()
    with torch.no_grad():tp=F.log_softmax(teacher.lm_head(th).float(),-1)
    loss=.5*(F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='sum')+
             F.kl_div(F.log_softmax(logits,-1),tp,log_target=True,reduction='sum'))/5
    loss.backward();difference=abs(chunk['loss']-float(loss.detach()))
    if difference>SMOKE_TOLERANCES['chunk_objective_absolute']:raise RuntimeError('Chunk loss mismatch')
    report['chunked_loss_parity']=gradient_comparison({'hidden':staged.grad},{'hidden':full.grad},
        rtol=SMOKE_TOLERANCES['chunk_hidden_gradient_rtol'],atol=SMOKE_TOLERANCES['chunk_hidden_gradient_atol'])
    report['chunked_loss_parity'].update(tokens=5,chunk_tokens=3,objective_absolute_difference=difference)
    del staged,full,logits,tp,th,sh,loss
    torch.cuda.empty_cache()
    print('[small smoke] full2047-target trial and changed FP16 export',flush=True)
    optimizer,scaler=optimizer_for(bank),scaler_for();trials=[]
    for _ in range(9):
        values=attempt_update(bank,teacher,windows[0],optimizer,scaler,detail=True);trials.append(values)
        if not values['overflow']:break
    if values['overflow']:raise RuntimeError('Smoke overflow budget exceeded')
    if any(v['nonzero']==0 or not v['finite'] for v in values['gradient_family_ranges'].values()):
        raise RuntimeError('Missing full-window gradient family')
    report['full_window_trial']={'attempts':trials,'successful_updates':1,'discard_before_training':True}
    report['final_parameter_ranges']=family_ranges(bank)
    exported=bank.export_other(initial_other);comparison=overlay.compare_small(initial_other,exported)
    if not comparison['actual_changed_tensor_count']:raise RuntimeError('No changed FP16 tensors in trial export')
    directory=args.work_dir/'smoke_export';directory.mkdir()
    path=directory/'other_fp16.pt';torch.save(exported,path)
    report['export_parity']=export_parity(bank,path,ids)
    report['export_parity'].update(tensor_comparison=comparison,bytes=path.stat().st_size,sha256=overlay.sha(path),
                                  archive_byte_delta=path.stat().st_size-(args.initialization_dir/'other_fp16.pt').stat().st_size)
    bank.load_state_dict(initial);bank.assert_frozen();report['trial_state_discarded']=True


class TrainingRun:
    def __init__(self,bank,teacher,windows,binding,args,report,frozen_before):
        self.bank,self.teacher,self.windows=bank,teacher,windows
        self.binding,self.args,self.report=binding,args,report
        self.frozen_before=frozen_before
        self.optimizer,self.scaler=optimizer_for(bank),scaler_for()
        self.successes=self.attempts=self.overflows=0
        self.history=[];self.schedule=training_schedule();self.final_checkpoint=None
        self.journal_path=args.work_dir/'attempts.jsonl'
        self.in_attempt=False

    def journal(self,event):
        with self.journal_path.open('a') as stream:
            stream.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**event},allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())

    def checkpoint(self,reason,*,resumable=True):
        state={'format':'MAMBA2_SMALL_TRAINING_CHECKPOINT_V1','binding':self.binding,
            'parent_manifest_sha256':norm_overlay.PARENT_MANIFEST_SHA256,'initialization':initialization(),
            'smoke_report_sha256':self.report['smoke_report_sha256'],'schedule':self.schedule,
            'successful_updates':self.successes,'attempts':self.attempts,'overflow_retries':self.overflows,
            'history':self.history,'masters':self.bank.state_dict(),'optimizer':cpu_tree(self.optimizer.state_dict()),
            'scaler':self.scaler.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'frozen_model_sha256':self.frozen_before,'resumable':resumable,'reason':reason}
        path=self.args.work_dir/f'checkpoint_attempt{self.attempts:04d}_step{self.successes:04d}_{reason}_{time.time_ns()}.pt'
        receipt=save_checkpoint(path,state)
        if resumable:
            self.final_checkpoint=receipt;write_json(self.args.work_dir/'latest.json',receipt)
        else:self.report['failure_snapshot']=receipt
        return receipt

    def resume(self):
        latest=json.loads((self.args.work_dir/'latest.json').read_text());path=Path(latest['file'])
        if path.parent.resolve()!=self.args.work_dir.resolve() or overlay.sha(path)!=latest['sha256']:
            raise ValueError('Resume checkpoint integrity/path differs')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if (state.get('format')!='MAMBA2_SMALL_TRAINING_CHECKPOINT_V1'
                or any(type(state.get(key)) is not int for key in ('successful_updates','attempts','overflow_retries'))):
            raise ValueError('Resume checkpoint format/counter types differ')
        for key,value in {'binding':self.binding,'parent_manifest_sha256':norm_overlay.PARENT_MANIFEST_SHA256,
                'initialization':initialization(),'schedule':self.schedule,'smoke_report_sha256':self.report['smoke_report_sha256'],
                'frozen_model_sha256':self.frozen_before,'resumable':True}.items():
            if state.get(key)!=value:raise ValueError(f'Resume provenance differs: {key}')
        self.successes,self.attempts,self.overflows=state['successful_updates'],state['attempts'],state['overflow_retries']
        if not 0<=self.successes<=1024 or not 0<=self.overflows<=8 or self.attempts!=self.successes+self.overflows:
            raise ValueError('Resume counters differ')
        self.bank.load_state_dict(state['masters']);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        self.history=state['history'];self.final_checkpoint=latest
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        starts=[];completed=[]
        if self.journal_path.exists():
            entries=[json.loads(line) for line in self.journal_path.read_text().splitlines() if line.strip()]
            starts=[entry for entry in entries if entry.get('event')=='attempt_start']
            completed=[entry for entry in entries if entry.get('event')=='attempt_complete']
        expected=list(range(1,self.attempts+1))
        if ([x['attempt'] for x in starts]!=expected or [x['attempt'] for x in completed]!=expected
                or len(self.history)!=self.attempts):
            raise RuntimeError('Uncheckpointed/in-flight attempt detected; strict policy forbids replay or ambiguous resume')
        for actual,recorded in zip(completed,self.history):
            if any(actual.get(key)!=value for key,value in recorded.items()):raise ValueError('Journal/checkpoint history differs')
        self.report['resume_uncheckpointed_attempts']=0
        self.journal({'event':'resume','from_checkpoint':latest,'uncheckpointed_attempts':0})

    def run(self):
        if self.args.resume:self.resume()
        elif (self.args.work_dir/'latest.json').exists():raise FileExistsError('Use explicit --resume')
        self.report.update(schedule=self.schedule,history=self.history,successful_updates=self.successes,
            attempts=self.attempts,overflow_retries=self.overflows,successful_target_exposures=self.successes*2047,
            attempted_target_exposures=self.attempts*2047,final_checkpoint=self.final_checkpoint,
            resumption_policy='Reject any journal attempt not represented in the exact saved checkpoint; never replay')
        try:
            while self.successes<1024:
                if STOP_REQUESTED:
                    self.checkpoint('termination');raise RequestedStop('Signal requested stop between updates')
                started=time.time();window_id=self.schedule[self.successes]
                self.journal({'event':'attempt_start','attempt':self.attempts+1,'successful_updates_before':self.successes,'window':window_id})
                self.in_attempt=True
                values=attempt_update(self.bank,self.teacher,self.windows[window_id],self.optimizer,self.scaler,detail=self.successes==1023)
                self.attempts+=1
                if values['overflow']:self.overflows+=1
                else:self.successes+=1
                self.in_attempt=False
                entry={'attempt':self.attempts,'successful_updates_after':self.successes,'window':window_id,
                       'seconds':time.time()-started,**values}
                self.history.append(entry);self.journal({'event':'attempt_complete',**entry})
                if self.overflows>8 or self.attempts>1032:raise RuntimeError('Fixed overflow/attempt budget exceeded')
                if (not values['overflow'] and self.successes%32==0) or STOP_REQUESTED:
                    self.checkpoint('final' if self.successes==1024 else 'periodic')
                self.report.update(successful_updates=self.successes,attempts=self.attempts,overflow_retries=self.overflows,
                    successful_target_exposures=self.successes*2047,attempted_target_exposures=self.attempts*2047,
                    final_checkpoint=self.final_checkpoint,gpu_memory=gpu_memory_receipt())
                write_json(self.args.report,self.report)
                if self.successes%16==0 or values['overflow']:
                    print(f'[small train] {self.successes}/1024 attempt {self.attempts} scale {values["loss_scale_after"]:g} '
                          f'overflow={values["overflow"]} loss={values["loss"]:.6f}',flush=True)
            return self.final_checkpoint
        except RequestedStop:raise
        except BaseException:
            self.checkpoint('failure',resumable=False)
            raise


def finalize(run,raw,initial_other,args,report):
    bank=run.bank
    frozen_after=parameter_hashes(bank.model,bank.inventory)
    if frozen_after!=run.frozen_before:raise RuntimeError('Frozen E8/W5 GPU values changed')
    report['frozen_parameter_audit']={'tensor_count':114,'before_sha256':run.frozen_before,'after_sha256':frozen_after,'bitwise_unchanged':True}
    report['final_parameter_ranges']=family_ranges(bank);report['final_stability']=bank.stability_receipt()
    report['final_gradient_family_ranges']=run.history[-1]['gradient_family_ranges']
    directory=args.work_dir/f'final_export_{time.time_ns()}';directory.mkdir()
    other=bank.export_other(initial_other);path=directory/'other_fp16.pt';torch.save(other,path)
    report['export_tensor_comparison']=overlay.compare_small(initial_other,other)
    report['export_parity']=export_parity(bank,path,run.windows[0,None,:128].cuda())
    report['replacement_file']={'bytes':path.stat().st_size,'sha256':overlay.sha(path)}
    report['archive_byte_delta']=path.stat().st_size-raw['files']['other_fp16.pt']['bytes']
    report.update(complete=True,final_step=1024,final_checkpoint=run.final_checkpoint,
                  elapsed_seconds=time.time()-report['started_at_unix'],gpu_memory=gpu_memory_receipt())
    write_json(args.report,report)
    manifest={'format':overlay.FORMAT,'complete':True,'parent_manifest_sha256':norm_overlay.PARENT_MANIFEST_SHA256,
        'initialization':initialization(),'model_config':MODEL_CONFIG,'binding':report['binding'],
        'files':{'other_fp16.pt':report['replacement_file']},'inherited_files':{k:v for k,v in raw['files'].items() if k!='other_fp16.pt'},
        'selected_small_tensors':selected_inventory(),'parameter_mapping':raw['parameter_mapping'],
        'training_receipt':{'final_step':1024,'checkpoint_sha256':run.final_checkpoint['sha256'],
                            'report_sha256':overlay.sha(args.report),'smoke_report_sha256':report['smoke_report_sha256']}}
    write_json(directory/'manifest.json',manifest)
    _,_,receipt=overlay.verify_overlay(args.parent_dir,directory,initialization_dir=args.initialization_dir,data_dir=args.data_dir,
        protocol=args.protocol,training_report=args.report,training_checkpoint=Path(run.final_checkpoint['file']),smoke_report=args.smoke_report)
    write_json(args.work_dir/'final_overlay_integrity.json',receipt)
    args.overlay_dir.parent.mkdir(parents=True,exist_ok=True);directory.rename(args.overlay_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','train'),required=True)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'models/source')
    parser.add_argument('--parent-dir',type=Path,default=ROOT/'artifacts/e8w5_v1')
    parser.add_argument('--initialization-dir',type=Path,default=ROOT/'artifacts/norm_compensation_v1')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'training_data/small_compensation_v1')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/SMALL_TENSOR_COMPENSATION_PROTOCOL.md')
    parser.add_argument('--work-dir',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--smoke-report',type=Path);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--overlay-dir',type=Path,default=ROOT/'artifacts/small_compensation_v1')
    args=parser.parse_args();args.work_dir=args.work_dir.resolve();args.report=args.report.resolve()
    if args.mode=='train' and args.smoke_report is None:parser.error('Train requires --smoke-report')
    if args.mode=='smoke' and args.resume:parser.error('Smoke cannot resume')
    if args.report.exists() and not args.resume:raise FileExistsError('Preserve old report; use a fresh path')
    if args.mode=='train' and args.overlay_dir.exists():raise FileExistsError('Preserve existing candidate')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    lock=args.work_dir/'active_job.json'
    if lock.exists():
        old=json.loads(lock.read_text())
        try:os.kill(old['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Prior job still alive; refusing concurrent work')
        if not args.resume:raise RuntimeError('Existing job requires explicit resume or fresh paths')
        (args.work_dir/f'prior_job_{time.time_ns()}.json').write_bytes(lock.read_bytes())
    if args.report.exists():
        previous=json.loads(args.report.read_text())
        if previous.get('error_type') not in (None,'RequestedStop'):
            raise RuntimeError('Numerically failed run cannot resume without a new authorized protocol')
        args.report.with_name(args.report.stem+f'.before_resume_{time.time_ns()}.json').write_bytes(args.report.read_bytes())
    write_json(lock,{'pid':os.getpid(),'mode':args.mode,'started_at_unix':time.time()})
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    report={'complete':False,'passed':False if args.mode=='smoke' else None,'mode':args.mode,'pid':os.getpid(),
        'started_at_unix':time.time(),'parent_manifest_sha256':norm_overlay.PARENT_MANIFEST_SHA256,
        'initialization':initialization(),'selected_small_tensors':selected_inventory(),'evaluation_data_used':False,'split':'train'}
    write_json(args.report,report)
    try:
        raw,norm_manifest,windows,binding=verify_inputs(args);report['binding']=binding
        if args.mode=='train':
            sr=json.loads(args.smoke_report.read_text())
            if (sr.get('complete') is not True or sr.get('passed') is not True or sr.get('binding')!=binding
                    or sr.get('initialization')!=initialization()):raise ValueError('Successful smoke binding differs')
            report['smoke_report_sha256']=overlay.sha(args.smoke_report)
        torch.set_num_threads(8);torch.manual_seed(20260927);torch.cuda.manual_seed_all(20260927)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        report['environment']=environment_receipt();write_json(args.report,report)
        print(f'[small {args.mode}] inputs verified, PID={os.getpid()}; loading original E8/W5 + norm-v1 initialization',flush=True)
        model=load_quantized_model(args.parent_dir)
        norm_overlay.apply_overlay(model,args.initialization_dir,norm_manifest)
        bank=SmallMasters(model)
        if len(bank.masters)!=393 or sum(p.numel() for p in bank.parameters())!=3580928:raise ValueError('Wrong selected capacity')
        frozen_before=parameter_hashes(model,bank.inventory)
        if len(frozen_before)!=114:raise ValueError('Wrong frozen projection/vocabulary coverage')
        teacher=load_source_model(args.source_dir)
        initial_other=torch.load(args.initialization_dir/'other_fp16.pt',map_location='cpu',weights_only=True)
        if args.mode=='smoke':
            smoke(bank,teacher,windows,initial_other,args,report)
            frozen_after=parameter_hashes(model,bank.inventory)
            if frozen_before!=frozen_after:raise RuntimeError('Frozen E8/W5 values changed in smoke')
            report['frozen_parameter_audit']={'tensor_count':114,'before_sha256':frozen_before,'after_sha256':frozen_after,'bitwise_unchanged':True}
            report.update(complete=True,passed=True,elapsed_seconds=time.time()-report['started_at_unix'],gpu_memory=gpu_memory_receipt())
            write_json(args.report,report)
        else:
            run=TrainingRun(bank,teacher,windows,binding,args,report,frozen_before)
            report['initial_parameter_ranges']=family_ranges(bank);report['initial_stability']=bank.stability_receipt()
            run.run();finalize(run,raw,initial_other,args,report)
        print(json.dumps({'complete':True,'mode':args.mode,'report':str(args.report),'sha256':overlay.sha(args.report)}),flush=True)
    except BaseException as error:
        report.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                      elapsed_seconds=time.time()-report['started_at_unix'])
        if torch.cuda.is_available():report['gpu_memory']=gpu_memory_receipt()
        write_json(args.report,report);raise


if __name__=='__main__':main()
