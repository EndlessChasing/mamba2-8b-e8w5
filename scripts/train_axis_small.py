#!/usr/bin/env python3
"""Fixed448-step small-tensor readaptation on the verified current axis base."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mamba_e8w5 import axis_small_base as base
from mamba_e8w5.small_training import SmallMasters,selected_inventory,family_ranges
from mamba_e8w5.norm_training import chunked_loss_backward,gradient_receipt,install_other,restore_parameters
from mamba_e8w5.small_overlay import compare_small
from mamba_e8w5.runtime import load_source_model,MODEL_CONFIG,environment_receipt,gpu_memory_receipt
from train_small_compensation import attempt_update
from train_norm_compensation import (write_json,save_checkpoint,cpu_tree,tree_digest,parameter_hashes,
                                    scaler_for,gradient_comparison,SMOKE_TOLERANCES)
from prepare_teacher_kl_data import train_schedule

PROTOCOL_SHA='a2509bec3d82748b18e452c9bbb5053a894c96ed5309eeb67d45f7dfa6029282'
FORMAT='MAMBA2_AXIS_SMALL_READAPTATION_V1'
CHECKPOINT_FORMAT='MAMBA2_AXIS_SMALL_TRAINING_CHECKPOINT_V1'
FROZEN={
 'mamba_e8w5/axis_small_base.py':'4e7026475fc97b005105b27043a15bae83c576934612b1e346576438eef4da70',
 'mamba_e8w5/small_training.py':'6e98858460534c4047f7bf75bd62eaf7769d7e46919cd9de6868fd6177a80086',
 'mamba_e8w5/norm_training.py':'25ac3ec6ba167902014018f99fbf8dbbc659e0e4903f80baf28e28a4a3e74087',
 'scripts/train_small_compensation.py':'d92317e3157f374ef872e40f0de5ad81f9dc6ad9732b19f3df616a114fc5ca7a',
 'scripts/train_norm_compensation.py':'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
 'mamba_e8w5/small_overlay.py':'3512c1b232c4039c16a26ffc0fe8c2eac1be6bb9c704ccbcb1d4e8e7788a1286',
 'scripts/prepare_teacher_kl_data.py':'4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13'}
HYPERPARAMETERS={
 'seed':20260928,'successful_updates':448,'maximum_attempts':456,'maximum_overflow_retries':8,
 'epochs':1,'windows':448,'stored_tokens':2048,'targets_per_window':2047,
 'optimizer':'AdamW','learning_rate':3e-5,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,
 'gradient_clip_norm':1.,'ce_coefficient':.5,'teacher_to_student_kl_coefficient':.5,'temperature':1.,
 'logits_chunk_tokens':64,'selected_tensors':393,'selected_parameters':3580928,
 'master_dtype':'float32','forward_dtype':'float16','native_eval_mode':True,'checkpoint_each_block':True,
 'loss_scaler':{'init_scale':1024.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000},
 'initialization':'current accepted trained393 FP16; fresh optimizer/scaler',
 'schedule':'prepared seed20260928 torch.randperm448 once','selection':'final448 only',
 'resume':False,'overflow_policy':'same window retry, no master/optimizer update; maximum8'}
INVENTORY=selected_inventory()
STOP_REQUESTED=False


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_receipt(path):return {'file':str(path),'bytes':Path(path).stat().st_size,'sha256':sha(path)}


def stop_requested(_signum,_frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def optimizer_for(bank):
    return torch.optim.AdamW(bank.parameters(),lr=3e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)


def accounting(history,schedule):
    if schedule!=train_schedule() or any(type(x) is not int for x in schedule):raise ValueError('Fixed448 schedule differs')
    successes=overflows=0
    for i,row in enumerate(history,1):
        if (successes>=448 or type(row.get('attempt')) is not int or row['attempt']!=i
                or type(row.get('successful_updates_after')) is not int or type(row.get('overflow')) is not bool
                or type(row.get('schedule_entry')) is not int or row['schedule_entry']!=schedule[successes]
                or type(row.get('targets')) is not int or row['targets']!=2047):raise ValueError('Attempt accounting differs')
        if row['overflow']:overflows+=1
        else:successes+=1
        if row['successful_updates_after']!=successes:raise ValueError('Successful count differs')
    if overflows>8 or len(history)>456:raise ValueError('Overflow/attempt budget exceeded')
    return {'successful_updates':successes,'attempts':len(history),'overflow_retries':overflows,
            'successful_target_exposures':successes*2047,'attempted_target_exposures':len(history)*2047}


def validate_masters(masters):
    if set(masters)!=set(INVENTORY):raise ValueError('Expected393 master tensors')
    for name,row in INVENTORY.items():
        value=masters[name]
        if (value.dtype!=torch.float32 or list(value.shape)!=row['shape'] or not torch.isfinite(value).all()
                or not torch.isfinite(value.half()).all()):raise ValueError(f'Invalid FP32/FP16 master:{name}')


def verify_inputs():
    for name,digest in FROZEN.items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Frozen training helper changed:{name}')
    if sha(ROOT/'docs/AXIS_SMALL_READAPTATION_PROTOCOL.md')!=PROTOCOL_SHA:raise ValueError('Protocol differs')
    context=base.verify_inputs()
    heldout=context.pop('heldout_windows');del heldout  # Never passed to any fitting function.
    windows=context['training_windows']
    if windows.dtype!=torch.int64 or list(windows.shape)!=[448,2048] or context['data_manifest']['schedule']!=train_schedule():
        raise ValueError('Prepared training schedule/windows differ')
    if len(INVENTORY)!=393 or sum(v['numel'] for v in INVENTORY.values())!=3580928:raise ValueError('Selected capacity differs')
    compare_small(context['old_small'],context['old_small'])
    code={**FROZEN,'scripts/train_axis_small.py':sha(__file__),
          'mamba_e8w5/axis_small_base.py':sha(ROOT/'mamba_e8w5/axis_small_base.py')}
    binding={**context['binding'],'protocol_sha256':PROTOCOL_SHA,'hyperparameters':HYPERPARAMETERS,
             'trainer_source_sha256':sha(__file__),'training_code_sha256':code}
    return context,binding


def final_recheck(context,binding):
    base.final_recheck(context)
    for name,digest in binding['training_code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Training input changed:{name}')


@torch.no_grad()
def export_parity(bank,path,ids):
    expected=bank.forward(ids,False);expected_logits=bank.model.lm_head(expected[:,-8:])
    exported=torch.load(path,map_location='cpu',weights_only=True)
    previous=install_other(bank.model,exported)
    try:
        actual=bank.model.backbone(ids);actual_logits=bank.model.lm_head(actual[:,-8:])
        hashes={'functional_hidden_sha256':tensor_sha(expected),'native_hidden_sha256':tensor_sha(actual),
                'functional_logits_sha256':tensor_sha(expected_logits),'native_logits_sha256':tensor_sha(actual_logits)}
        if (not torch.isfinite(actual).all() or not torch.isfinite(actual_logits).all()
                or hashes['functional_hidden_sha256']!=hashes['native_hidden_sha256']
                or hashes['functional_logits_sha256']!=hashes['native_logits_sha256']):raise RuntimeError('Native export parity failed')
        return {**hashes,'hidden_bitwise_equal':True,'logits_bitwise_equal':True,'input_tokens':ids.numel(),'logit_positions':8}
    finally:restore_parameters(bank.model,previous);bank.assert_frozen()


def self_test():
    schedule=train_schedule();rows=[]
    for i,window in enumerate(schedule):
        rows.append({'attempt':i+1,'successful_updates_after':i+1,'schedule_entry':window,'overflow':False,'targets':2047})
    assert accounting(rows,schedule)['successful_target_exposures']==917056
    retries=[{'attempt':i+1,'successful_updates_after':0,'schedule_entry':schedule[0],'overflow':True,'targets':2047} for i in range(8)]
    completed=[{**row,'attempt':row['attempt']+8} for row in rows]
    assert accounting(retries+completed,schedule)['attempts']==456
    for bad in ([{**rows[0],'targets':2047.0}], [{**rows[0],'successful_updates_after':True}],
                [{**rows[0],'schedule_entry':schedule[1]}],retries+[{**retries[-1],'attempt':9}]):
        try:accounting(bad,schedule)
        except ValueError:pass
        else:raise AssertionError('Malformed accounting accepted')
    return {'passed':True,'checks':['448 successes/917056 targets','8 same-window overflow retries/456 attempts',
                                   'Reject float/bool/mismatched-window/excess-overflow accounting']}


def smoke(bank,teacher,windows,old_small,args,report):
    ids=windows[0,None,:128].cuda();initial=bank.state_dict();initial_digest=tree_digest(initial)
    report['initial_parameter_ranges']=family_ranges(bank);report['initial_stability']=bank.stability_receipt()
    try:
        with torch.no_grad():
            native=bank.model.backbone(ids);functional=bank.forward(ids,False)
            nh,fh=tensor_sha(native),tensor_sha(functional)
            nl,fl=tensor_sha(bank.model.lm_head(native[:,-8:])),tensor_sha(bank.model.lm_head(functional[:,-8:]))
            if nh!=fh or nl!=fl or not torch.isfinite(native).all():raise RuntimeError('Initial native/functional parity failed')
            report['initial_forward']={'hidden_bitwise_equal':True,'logits_bitwise_equal':True,
                'native_hidden_sha256':nh,'functional_hidden_sha256':fh,'native_logits_sha256':nl,'functional_logits_sha256':fl,
                'input_tokens':128,'logit_positions':8}
            for parameter in bank.parameters():
                parameter.add_(torch.linspace(-.003,.003,parameter.numel(),device=parameter.device).reshape(parameter.shape))
        del native,functional
        grads=[];hashes=[];families=[];coverage=[]
        for enabled in (False,True):
            for parameter in bank.parameters():parameter.grad=None
            hidden=bank.forward(ids,enabled);hashes.append(tensor_sha(hidden))
            tangent=torch.linspace(-1,1,hidden.shape[-1],device=hidden.device)
            (1024*(hidden.float()*tangent).mean()).backward()
            item=gradient_receipt(bank);ranges=family_ranges(bank,True)
            if (item['tensors']!=393 or item['parameters']!=3580928 or not item['finite']
                    or len(ranges)!=7 or any(not x['finite'] or not x['nonzero'] for x in ranges.values())):
                raise RuntimeError('Missing finite/nonzero seven-family gradients')
            coverage.append(item);families.append(ranges)
            grads.append({name:p.grad.detach().cpu().clone() for name,p in bank.masters.items()})
            del hidden
        if hashes[0]!=hashes[1]:raise RuntimeError('Checkpoint forward hash differs')
        report['tolerances']=SMOKE_TOLERANCES
        report['checkpoint_parity']=gradient_comparison(*grads,rtol=SMOKE_TOLERANCES['checkpoint_gradient_rtol'],
            atol=SMOKE_TOLERANCES['checkpoint_gradient_atol'])
        report['checkpoint_parity'].update(hidden_sha256=hashes,hidden_bitwise_equal=True,
            gradient_coverage=coverage,gradient_family_ranges=families,perturbation='linspace(-.003,.003) per FP32 master; discarded')
        bank.load_state_dict(initial)
        for parameter in bank.parameters():parameter.grad=None
        del grads;torch.cuda.empty_cache()
        with torch.no_grad():student=bank.forward(ids[:,:5],False);target_hidden=teacher.backbone(ids[:,:5])
        targets=windows[0,None,1:6].cuda();staged=student.detach().clone().requires_grad_()
        chunked=chunked_loss_backward(staged,target_hidden,targets,bank.model.lm_head,teacher.lm_head,None,3)
        direct=student.detach().clone().requires_grad_();logits=bank.model.lm_head(direct).float()
        with torch.no_grad():teacher_logp=F.log_softmax(teacher.lm_head(target_hidden).float(),-1)
        loss=.5*(F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='sum')+
            F.kl_div(F.log_softmax(logits,-1),teacher_logp,reduction='sum',log_target=True))/5
        loss.backward();difference=abs(chunked['loss']-float(loss.detach()))
        if difference>SMOKE_TOLERANCES['chunk_objective_absolute']:raise RuntimeError('Chunked full-vocabulary objective differs')
        report['chunked_loss_parity']=gradient_comparison({'hidden':staged.grad},{'hidden':direct.grad},
            rtol=SMOKE_TOLERANCES['chunk_hidden_gradient_rtol'],atol=SMOKE_TOLERANCES['chunk_hidden_gradient_atol'])
        report['chunked_loss_parity'].update(targets=5,chunk_tokens=3,tail_targets=2,vocabulary=256000,
            objective_absolute_difference=difference,objective='0.5CE+0.5KL teacher||student, T1, sum/total targets')
        del student,target_hidden,targets,staged,direct,logits,teacher_logp,loss
        torch.cuda.empty_cache();optimizer,scaler=optimizer_for(bank),scaler_for();trials=[]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(9):
            trial=attempt_update(bank,teacher,windows[0],optimizer,scaler,detail=True);trials.append(trial)
            if not trial['overflow']:break
        if trial['overflow']:raise RuntimeError('Smoke exceeded8 overflow retries')
        if any(not x['finite'] or not x['nonzero'] for x in trial['gradient_family_ranges'].values()):
            raise RuntimeError('Full-window training trial lacks seven-family gradients')
        report['full_window_trial']={'attempts':trials,'successful_updates':1,'targets':2047,
            'vocabulary':256000,'chunks':32,'last_chunk_targets':63,'discard_before_training':True,
            'gpu_memory':gpu_memory_receipt()}
        report['post_update_parameter_ranges']=family_ranges(bank);report['post_update_stability']=bank.stability_receipt()
        exported=bank.export_other(old_small);comparison=compare_small(old_small,exported)
        directory=args.work_dir/'smoke_export';directory.mkdir();path=directory/'other_fp16.pt';torch.save(exported,path)
        reloaded=torch.load(path,map_location='cpu',weights_only=True)
        if any(tensor_sha(bank.masters[n].half())!=tensor_sha(reloaded[n]) for n in INVENTORY):
            raise RuntimeError('Smoke serialized FP16 master parity differs')
        report['export_parity']={**export_parity(bank,path,ids),'file':file_receipt(path),
                                'tensor_comparison':comparison,'all393_master_rounding_equal':True}
        del optimizer,scaler
    finally:
        bank.load_state_dict(initial)
        for parameter in bank.parameters():parameter.grad=None
        bank.assert_frozen()
        report['trial_state_discarded']=tree_digest(bank.state_dict())==initial_digest
        if not report['trial_state_discarded']:raise RuntimeError('Smoke master reset failed')


class TrainingRun:
    def __init__(self,bank,teacher,windows,binding,args,report,baseline_hashes,teacher_hashes):
        self.bank,self.teacher,self.windows=bank,teacher,windows
        self.binding,self.args,self.report=binding,args,report
        self.baseline_hashes,self.teacher_hashes=baseline_hashes,teacher_hashes
        self.optimizer,self.scaler=optimizer_for(bank),scaler_for()
        self.schedule=train_schedule();self.history=[];self.successes=self.attempts=self.overflows=0
        self.final_checkpoint=None

    def journal(self,event):
        with (self.args.work_dir/'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**event},allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())

    def checkpoint(self,reason):
        state={'format':CHECKPOINT_FORMAT,'binding':self.binding,
               'smoke_report_sha256':self.report['smoke_report_sha256'],
               'diagnostic_report_sha256':self.report['diagnostic_report_sha256'],
               'schedule':self.schedule,'history':self.history,'successful_updates':self.successes,
               'attempts':self.attempts,'overflow_retries':self.overflows,'masters':self.bank.state_dict(),
               'optimizer':cpu_tree(self.optimizer.state_dict()),'scaler':self.scaler.state_dict(),
               'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
               'frozen_model_sha256':self.baseline_hashes,'teacher_model_sha256':self.teacher_hashes,
               'reason':reason,'resumable':False}
        path=self.args.work_dir/f'checkpoint_attempt{self.attempts:03d}_step{self.successes:03d}_{reason}.pt'
        receipt=save_checkpoint(path,state)
        if reason=='final':self.final_checkpoint=receipt
        return receipt

    def run(self):
        self.report.update(schedule=self.schedule,history=self.history,resumption_policy='No resume/replay; fresh paths only')
        try:
            while self.successes<448:
                if STOP_REQUESTED:raise RuntimeError('Requested stop between updates; no replay/resume')
                window=self.schedule[self.successes];started=time.perf_counter()
                self.journal({'event':'attempt_start','attempt':self.attempts+1,
                    'successful_updates_before':self.successes,'schedule_entry':window})
                values=attempt_update(self.bank,self.teacher,self.windows[window],self.optimizer,self.scaler,
                    detail=self.successes==447)
                if values['gradients']['tensors']!=393 or values['gradients']['parameters']!=3580928:
                    raise RuntimeError('Actual selected gradient capacity differs')
                self.attempts+=1
                if values['overflow']:self.overflows+=1
                else:self.successes+=1
                row={'attempt':self.attempts,'successful_updates_after':self.successes,'schedule_entry':window,
                     'seconds':time.perf_counter()-started,**values}
                self.history.append(row);self.journal({'event':'attempt_complete',**row})
                counts=accounting(self.history,self.schedule)
                if not values['overflow'] and self.successes%64==0:
                    self.report.setdefault('checkpoints',[]).append(self.checkpoint('final' if self.successes==448 else 'periodic'))
                self.report.update(**counts,final_checkpoint=self.final_checkpoint,gpu_memory=gpu_memory_receipt())
                if self.attempts%8==0 or values['overflow'] or self.successes==448:write_json(self.args.report,self.report)
                if self.successes%8==0 or values['overflow']:
                    print(json.dumps({'updates':self.successes,'attempt':self.attempts,'ce':values['ce'],
                        'kl':values['teacher_to_student_kl'],'overflow':values['overflow']}),flush=True)
        except BaseException:
            self.report['failure_snapshot']=self.checkpoint('failure');raise


def finalize(run,context,args,report):
    counts=accounting(run.history,run.schedule)
    if counts['successful_updates']!=448 or run.final_checkpoint is None:raise RuntimeError('Final448 checkpoint missing')
    bank=run.bank;validate_masters(bank.masters)
    report['final_underlying507_audit']=base.audit_model(bank.model,context['current_expected507'])
    report['final_fixed114_audit']=base.verify_fixed_large(bank.model,context)
    report['final_teacher507_audit']=base.audit_model(run.teacher,context['source_expected507'])
    cp=run.final_checkpoint
    if sha(cp['file'])!=cp['sha256']:raise RuntimeError('Final checkpoint changed')
    saved=torch.load(cp['file'],map_location='cpu',weights_only=True)
    required={'format':CHECKPOINT_FORMAT,'binding':run.binding,'reason':'final','resumable':False,
              'history':run.history,'schedule':run.schedule,'successful_updates':448,
              'attempts':counts['attempts'],'overflow_retries':counts['overflow_retries'],
              'smoke_report_sha256':report['smoke_report_sha256'],
              'diagnostic_report_sha256':report['diagnostic_report_sha256'],
              'frozen_model_sha256':context['current_expected507'],'teacher_model_sha256':context['source_expected507']}
    if any(saved.get(k)!=v for k,v in required.items()):raise RuntimeError('Final checkpoint provenance/accounting differs')
    validate_masters(saved['masters'])
    stage=args.work_dir/'final_export';stage.mkdir()
    other=bank.export_other(context['old_small']);path=stage/'other_fp16.pt';torch.save(other,path)
    actual=torch.load(path,map_location='cpu',weights_only=True);comparison=compare_small(context['old_small'],actual)
    for name in INVENTORY:
        if (tensor_sha(actual[name])!=tensor_sha(saved['masters'][name].half())
                or tensor_sha(actual[name])!=tensor_sha(bank.masters[name].half())):
            raise RuntimeError(f'Final393 rounded-master readback mismatch:{name}')
    report['export_parity']=export_parity(bank,path,run.windows[0,None,:128].cuda())
    report['post_export_underlying507_audit']=base.audit_model(bank.model,context['current_expected507'])
    report['final_parameter_ranges']=family_ranges(bank);report['final_stability']=bank.stability_receipt()
    report['final_gradient_family_ranges']=run.history[-1]['gradient_family_ranges']
    report['final_export']={'file':file_receipt(path),'checkpoint448_master_to_fp16_bitwise_equal':True,
        'tensor_count':393,'parameter_count':3580928,'comparison':comparison}
    files=context['resolved_files'];inherited={name:row for name,row in files.items() if name!='other_fp16.pt'}
    if len(files)!=117 or len(inherited)!=116:raise RuntimeError('Resolved current117/inherited116 files differ')
    old_bytes=files['other_fp16.pt']['bytes'];new_bytes=path.stat().st_size
    storage={'current_logical_data_bytes':3138928792,'replaced_other_bytes':old_bytes,'replacement_other_bytes':new_bytes,
             'archive_byte_delta':new_bytes-old_bytes,'logical_inference_data_bytes':3138928792+new_bytes-old_bytes,
             'fp16_parameter_payload_bytes':3580928*2,'manifest_bytes':0,'complete_distribution_built':False}
    manifest={'format':FORMAT,'complete':True,'model_config':MODEL_CONFIG,'binding':run.binding,
              'initialization':{'small_manifest_sha256':'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006',
                                'other_fp16_sha256':'15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'},
              'files':{'other_fp16.pt':{'bytes':new_bytes,'sha256':sha(path)}},'inherited_files':inherited,
              'selected_small_tensors':INVENTORY,'frozen_base_hash_ledger':context['current_expected507'],
              'frozen_large_hash_ledger':{k:context['current_expected507'][k] for k in context['fixed_large_keys']},
              'exported_small_fp16_sha256':comparison['fp16_tensor_sha256'],
              'diagnostic_report_sha256':report['diagnostic_report_sha256'],
              'training_receipt':{'successful_updates':448,'attempts':counts['attempts'],'overflow_retries':counts['overflow_retries'],
                'checkpoint_sha256':cp['sha256'],'checkpoint_bytes':cp['bytes'],'smoke_report_sha256':report['smoke_report_sha256']},
              'storage':storage}
    mp=stage/'manifest.json'
    while True:
        write_json(mp,manifest);size=mp.stat().st_size
        if manifest['storage']['manifest_bytes']==size:break
        manifest['storage']['manifest_bytes']=size
    final_recheck(context,run.binding)
    if sha(args.smoke_report)!=report['smoke_report_sha256'] or sha(args.diagnostic_report)!=report['diagnostic_report_sha256']:
        raise RuntimeError('Passed smoke/diagnostic receipt changed')
    if args.overlay_dir.exists():raise FileExistsError('Candidate output already exists')
    stage.rename(args.overlay_dir)
    report['final_export']['file']=file_receipt(args.overlay_dir/'other_fp16.pt')
    report.update(complete=True,final_step=448,final_checkpoint=cp,manifest=file_receipt(args.overlay_dir/'manifest.json'),
                  storage=storage,**counts)


def diagnostic_receipt(path,context):
    data=json.loads(path.read_text())
    if (data.get('format')!='MAMBA2_AXIS_SMALL_DIAGNOSTIC_V1' or data.get('complete') is not True
            or data.get('correctness_passed') is not True or data.get('baseline_repeat_exact') is not True
            or data.get('fixed114_verified') is not True or data.get('binding')!=context['binding']
            or data.get('training_performed') is not False or data.get('validation_loaded') is not False
            or data.get('candidate_advancement') is not False
            or data.get('final_current507_restored_audit',{}).get('unchanged_content_sha256')!=context['current_expected507']):
        raise ValueError('Valid descriptive diagnostic required')
    return file_receipt(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('cpu-preflight','smoke','train'),required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--smoke-report',type=Path);parser.add_argument('--diagnostic-report',type=Path)
    parser.add_argument('--overlay-dir',type=Path,default=ROOT/'artifacts/axis_small_v1')
    args=parser.parse_args();torch.set_num_threads(8)
    args.report=args.report.resolve();args.overlay_dir=args.overlay_dir.resolve()
    if args.mode!='cpu-preflight' and args.work_dir is None:parser.error('GPU modes require a fresh --work-dir')
    if args.mode=='train' and (args.smoke_report is None or args.diagnostic_report is None):parser.error('Train requires smoke and diagnostic reports')
    if args.work_dir:args.work_dir=args.work_dir.resolve()
    if (args.report.exists() or args.report.with_suffix('.json.tmp').exists()
            or (args.work_dir is not None and args.work_dir.exists())
            or (args.mode=='train' and args.overlay_dir.exists())):raise FileExistsError('Fresh paths only; no overwrite/resume')
    if args.work_dir:args.work_dir.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    report={'format':'MAMBA2_AXIS_SMALL_TRAINING_REPORT_V1','complete':False,'mode':args.mode,'pid':os.getpid(),
            'started_at_unix':time.time(),'script_sha256':sha(__file__),'protocol_sha256':PROTOCOL_SHA,
            'training_split':'train','heldout_used_for_fitting':False,'validation_loaded':False,
            'initialization_selected_by_diagnostic':False,'selected_small_tensors':INVENTORY}
    started=time.monotonic()
    try:
        report['cpu_self_test']=self_test();context,binding=verify_inputs();report['binding']=binding
        if args.mode=='cpu-preflight':
            masters={name:tensor.float() for name,tensor in context['old_small'].items()};validate_masters(masters)
            report.update(complete=True,cuda_initialized=torch.cuda.is_initialized(),training_windows=[448,2048],
                selected_tensors=393,selected_parameters=3580928,heldout_tensor_discarded='heldout_windows' not in context)
            if report['cuda_initialized']:raise RuntimeError('CPU preflight initialized CUDA')
            return
        if args.mode=='train':
            prior=json.loads(args.smoke_report.read_text())
            if prior.get('complete') is not True or prior.get('passed') is not True or prior.get('binding')!=binding:
                raise ValueError('Passed smoke with identical binding required')
            report['smoke_report_sha256']=sha(args.smoke_report)
            report['diagnostic_receipt']=diagnostic_receipt(args.diagnostic_report,context)
            report['diagnostic_report_sha256']=report['diagnostic_receipt']['sha256']
        torch.manual_seed(20260928);torch.cuda.manual_seed_all(20260928)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');report['environment']=environment_receipt()
        model=base.load_model(context)
        report['initial_underlying507_audit']=base.audit_model(model,context['current_expected507'])
        large_identity=base.large_identity(model);report['initial_fixed114_audit']=base.verify_fixed_large(model,context,large_identity)
        bank=SmallMasters(model)  # All112 axis/W4/W5/old393 installed before reference capture.
        if len(bank.masters)!=393 or sum(p.numel() for p in bank.parameters())!=3580928:raise RuntimeError('Unexpected trainable capacity')
        if any(tensor_sha(bank.masters[n].half())!=tensor_sha(context['old_small'][n]) for n in INVENTORY):
            raise RuntimeError('Fixed old-trained393 initialization differs')
        teacher=load_source_model(ROOT/'models/source').eval().requires_grad_(False)
        report['initial_teacher507_audit']=base.audit_model(teacher,context['source_expected507'])
        windows=context['training_windows'];write_json(args.report,report)
        if args.mode=='smoke':
            smoke(bank,teacher,windows,context['old_small'],args,report)
            report['final_underlying507_audit']=base.audit_model(model,context['current_expected507'])
            report['final_fixed114_audit']=base.verify_fixed_large(model,context,large_identity)
            report['final_teacher507_audit']=base.audit_model(teacher,context['source_expected507'])
            final_recheck(context,binding);report.update(complete=True,passed=True)
        else:
            run=TrainingRun(bank,teacher,windows,binding,args,report,context['current_expected507'],context['source_expected507'])
            report['initial_parameter_ranges']=family_ranges(bank);report['initial_stability']=bank.stability_receipt()
            run.run();base.verify_fixed_large(model,context,large_identity);finalize(run,context,args,report)
        report['gpu_memory']=gpu_memory_receipt()
    except BaseException as error:
        report.update(complete=False,passed=False,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write_json(args.report,report)


if __name__=='__main__':main()
