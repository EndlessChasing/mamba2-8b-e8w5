#!/usr/bin/env python3
"""Prospective fixed-soft continuation; preserve the failed strict DEV receipt.

Stage dev-audit is CPU-only. Full/confirm require the passed preceding receipt.
No refitting, profile change, scoring tolerance, or automatic stage launch.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
FORMAT='MAMBA2_RESURFACE_SOFT_CONTINUATION_V1'
PROTOCOL_SHA='f6be64321c214bb7b229ae658b24aa75b8a0f3ec17a9df9059c4303920cdbabf'
FROZEN_SHA='7d8ecefb8c28c4a74b9a9f3faed22831849a312cc65781f22404a3200d52c471'
TRAIN_SHA='e64effbceda78b82488fd69608d6338fd2bb6e4b535c1d8eca12ddee789f70b6'
CANDIDATE_PINS={'checkpoint_sha256':'03ed8da1d7c49fde4b70c053719eff19b715393d7b96dc41427f3cfc62af3fba',
 'adapter_sha256':'1e9857feaf80ede6525cce839726883f6230c98a067cb226eb2f095693ea803e',
 'overlay_manifest_sha256':'5660676adb45806052cbaab7f5e50056769c4f5e106d5a49cf148f025bf2b7a8',
 'training_report_sha256':TRAIN_SHA}
KNOWN_ERROR="ValueError('Current DEV MK output/rows differ from completed historical baseline')"
BASE,ACTIVE,RESTORED='current_readapted','active_resurface','restored_current'
PROFILE_ENV=('MAMBA_DETERMINISTIC','CUBLAS_WORKSPACE_CONFIG','TRITON_CACHE_AUTOTUNING',
 'TRITON_AUTOTUNE_BLOCK_SIZE_M','TRITON_AUTOTUNE_BLOCK_SIZE_N','TRITON_AUTOTUNE_BLOCK_SIZE_K',
 'TRITON_AUTOTUNE_BLOCK_SIZE_DSTATE','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()


def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def assert_audit(value,expected,label):
 if value.get('verified_tensor_count')!=507 or value.get('unchanged_content_sha256')!=expected:
  raise ValueError(f'507 content proof differs:{label}')


def terminal_failure(strict,process,strict_sha):
 if (strict.get('format')!='MAMBA2_RESURFACE_READAPTED_EVALUATION_V1' or strict.get('complete') is not False or strict.get('error')!=KNOWN_ERROR
     or strict.get('stage')!='dev' or strict.get('script_sha256')!=FROZEN_SHA
     or 'gate' in strict or strict.get('mk_restoration_exact') is not True):
  raise ValueError('Original strict run did not fail solely at declared historical MK prerequisite')
 if (process.get('status')!='finished' or type(process.get('exit_code')) is not int
     or process['exit_code']!=1 or process.get('report_complete') is not False
     or process.get('report_sha256')!=strict_sha
     or not isinstance(process.get('finished_unix'),(float,int))
     or process['finished_unix']<process.get('started_unix',math.inf)
     or process.get('code_pins',{}).get('scripts/evaluate_resurface_readapted.py')!=FROZEN_SHA):
  raise ValueError('Expected matching terminated failed process receipt')
 if not strict.get('traceback','').rstrip().endswith('ValueError: Current DEV MK output/rows differ from completed historical baseline'):
  raise ValueError('Terminal traceback differs from the sole allowed original failure')


def validate_mk_arm(entry,cases,plan,expected,mk):
 if entry.get('complete') is not True or entry.get('replay_exact') is not True or len(entry.get('rows',[]))!=768:
  raise ValueError('Incomplete768 prompt arm or within-arm replay')
 for row,case in zip(entry['rows'],cases):
  metadata={k:v for k,v in case.items() if k!='prompt'}
  if any(type(row.get(k)) is not type(v) or row[k]!=v for k,v in metadata.items()):
   raise ValueError('Ordered MK input identity differs')
  ids=row.get('generated_ids')
  if (not isinstance(ids,list) or not 1<=len(ids)<=12 or any(type(x) is not int for x in ids)
      or row.get('prediction')!=mk.predict(row['output'])
      or type(row.get('correct')) is not bool or row['correct']!=(row['prediction']==row['answer'])
      or row.get('finite_hidden_every_backbone_call') is not True
      or row.get('cache_bytes')!=122028032 or row.get('prefill_cache')!=row.get('final_cache')):
   raise ValueError('Native greedy matcher/cache/finite proof differs')
  cache=row['prefill_cache']
  if cache!={'layers':56,'tensors':112,'dtype':'torch.float16','finite':True,'bytes':122028032}:
   raise ValueError('FP16 native cache coverage differs')
 summary=mk.summarize(entry['rows'])
 if entry.get('summary')!=summary['summary'] or entry.get('cells')!=summary['cells']:
  raise ValueError('Recomputed MK summary/cells differ')
 by_id={row['id']:row for row in entry['rows']}
 if len(by_id)!=768 or [r['id'] for r in entry.get('replays',[])]!=plan['replay_case_ids']:
  raise ValueError('Incomplete ordered12 replay selection')
 for row in entry['replays']:
  original=by_id[row['id']]
  if row.get('exact_equal') is not True or any(row.get(k)!=original[k] for k in ('generated_ids','output','prediction','correct')):
   raise ValueError('Fresh-cache exact replay evidence differs')
 for side in ('before','after'):assert_audit(entry[f'actual507_{side}'],expected,'MK'+side)
 if entry['identities507_before']!=entry['identities507_after'] or entry.get('identity_storage_version_unchanged') is not True:
  raise ValueError('MK base identity changed')
 return summary


def validate_ppl_arm(entry,windows,plan,expected,scorer):
 if len(entry.get('windows',[]))!=64 or len(windows)!=64 or len(plan)!=64:
  raise ValueError('Expected complete64 DEV prose windows')
 for row,window,identity in zip(entry['windows'],windows,plan):
  if any(row.get(k)!=v for k,v in identity.items()):raise ValueError('DEV prose identity differs')
  scorer.check_row(row,window,identity)
 summary=scorer.frozen.summarize(entry['windows'])
 if summary!=entry.get('summary') or summary['target_tokens']!=131008:
  raise ValueError('Recomputed weighted prose summary differs')
 for side in ('before','after'):assert_audit(entry[f'actual507_{side}'],expected,'PPL'+side)
 if entry['identities507_before']!=entry['identities507_after']:
  raise ValueError('PPL base identity changed')
 return summary


def audit_dev(args,frozen,shared,scorer,mk,tokenizer,data,prior,expected,payload,proof,checked,report):
 if not args.strict_report_sha or not args.strict_process_sha:
  raise ValueError('Terminal strict DEV report/process exact SHA required')
 strict=frozen.read_bound(args.strict_report,args.strict_report_sha,checked)
 process=frozen.read_bound(args.strict_process,args.strict_process_sha,checked)
 terminal_failure(strict,process,args.strict_report_sha)
 if process.get('report_bytes')!=args.strict_report.stat().st_size:
  raise ValueError('Terminal process report length differs')
 command=process.get('cmd',[])
 if (not isinstance(command,list) or any(type(x) is not str for x in command)
     or '--verify-only' in command or '--stage' not in command or '--report' not in command
     or command[command.index('--stage')+1]!='dev'
     or (ROOT/command[command.index('--report')+1]).resolve()!=args.strict_report
     or not any(Path(item).name=='evaluate_resurface_readapted.py' for item in command)):
  raise ValueError('Terminal process command is not the strict DEV GPU run')
 if strict.get('candidate_identity')!=proof['candidate_identity'] or strict.get('export_proof')!=proof:
  raise ValueError('Failed DEV did not evaluate exactly the current actual export proof')
 if strict.get('protocol_sha256')!=frozen.PROTOCOL_SHA or strict.get('gate_mode')!='soft':
  raise ValueError('Original soft training/evaluation policy differs')
 dev_args=argparse.Namespace(**vars(args));dev_args.stage='dev';dev_args.split_manifest_sha='a6d870fe3e400d892d7eb55aab62d8f15f2d61b6a922d6590ab53fcfcfa0961a'
 prepared=frozen.load_stage_data(dev_args,data,tokenizer,prior,checked)
 saved={k:v for k,v in prepared.items() if k not in ('windows','cases')}
 if strict.get('stage_data')!=saved:raise ValueError('Actual prepared DEV identity differs from strict report')
 if set(strict.get('arms',{}))!={BASE,ACTIVE,RESTORED} or set(strict.get('ppl',{}))!={BASE,ACTIVE,RESTORED}:
  raise ValueError('Missing or extra fixed DEV arms')
 base_identity=strict['arms'][BASE]['identities507_before']
 for arm in (BASE,ACTIVE,RESTORED):
  validate_mk_arm(strict['arms'][arm],prepared['cases'],prepared['plan'],expected,mk)
  validate_ppl_arm(strict['ppl'][arm],prepared['windows'],prepared['prose_plan'],expected,scorer)
  for section in ('arms','ppl'):
   if strict[section][arm]['identities507_before']!=base_identity:
    raise ValueError('Base parameter identity differs between tasks/arms')
 if not frozen.mk_exact(strict['arms'][BASE],strict['arms'][RESTORED]):
  raise ValueError('Restored768 complete MK rows differ')
 if (not scorer.exact_repeat(strict['ppl'][BASE],strict['ppl'][RESTORED])
     or not scorer.exact_repeat(strict['ppl'][BASE],prior['development']['current_axis_readapted_small'])):
  raise ValueError('Restored/historical complete64 CE/PPL differs')
 for key in ('initial_current507_audit','final_current507_audit'):
  assert_audit(strict[key],expected,key)
 if strict.get('final_current507_identity')!=base_identity:raise ValueError('Final base identity restoration differs')
 before,after=strict['adapter_before'],strict['adapter_after']
 if before!=after or before.get('tensor_count')!=224 or before.get('tensor_sha256')!=proof['adapter_tensor_sha256'] or before.get('gate_mode')!='soft':
  raise ValueError('Actual224 adapter proof before/after differs')
 if before.get('base_frozen')!={'tensors':507,'parameters':8236999680,'identity_version_gradients_unchanged':True,'actual_content_checked':False}:
  raise ValueError('Adapter frozen-base identity proof differs')
 ledger=strict['checked_input_sha256'];final=strict['final_bound_input_recheck']
 if final!={'file_count':len(ledger),'checked_input_sha256':ledger}:raise ValueError('Final strict bound-input audit incomplete')
 for path,digest in ledger.items():frozen.bind(path,digest,checked)
 comparison=mk.paired_comparison(strict['arms'][BASE]['rows'],strict['arms'][ACTIVE]['rows'])
 gate=frozen.stage_gate('dev',comparison,strict['ppl'][BASE]['summary']['ppl'],strict['ppl'][ACTIVE]['summary']['ppl'])
 history=frozen.read_bound(ROOT/'reports/readapted_mk_baseline_v1.json',frozen.BASELINE_MK_SHA,checked)
 old=history['arms']['current_axis_readapted_small']['rows'];new=strict['arms'][BASE]['rows'];diff=[]
 for a,b in zip(old,new):
  keys=[k for k in a if a[k]!=b[k]]
  if keys:
   if any(k not in ('output','generated_ids','prediction','correct') for k in keys):raise ValueError('Historical mismatch is not generation-only')
   diff.append({'id':a['id'],'changed_fields':keys,'historical':{k:a[k] for k in keys},'current':{k:b[k] for k in keys}})
 if len(diff)!=7 or len(old)!=768:raise ValueError('Known historical7-row mismatch scope differs')
 report.update(gate=gate,mk_comparison=comparison,ppl={arm:strict['ppl'][arm]['summary'] for arm in (BASE,ACTIVE,RESTORED)},
  environment_reference=strict['environment'],
  original_strict_run={'report_sha256':args.strict_report_sha,'process_sha256':args.strict_process_sha,
   'complete':False,'exit_code':1,'error':strict['error'],'preserved_unchanged':True},
  historical_mk_differences=diff,historical_normal_correct=111,same_process_current_normal_correct=112,
  independent_proofs={'all3x768_native_MK_and12_replays':True,'all3x64_weighted_PPL_chunks':True,
   'current_restored_MK_exact':True,'current_restored_historical_PPL_exact':True,
   'all507_and224_contents_identities':True,'final_bound_files_rechecked':True},
  complete=True,mode='cpu_terminal_dev_reanalysis',cuda_initialized=False)


def common(args,report):
 if sha(ROOT/'scripts/evaluate_resurface_readapted.py')!=FROZEN_SHA:raise ValueError('Frozen evaluator changed')
 frozen=importlib.import_module('evaluate_resurface_readapted');checked={}
 frozen.bind(ROOT/'docs/RESURFACE_SOFT_CONTINUATION_PROTOCOL.md',PROTOCOL_SHA,checked)
 frozen.bind(ROOT/'scripts/evaluate_resurface_readapted.py',FROZEN_SHA,checked);frozen.bind(__file__,sha(__file__),checked)
 frozen.bind(ROOT/'docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md',frozen.PROTOCOL_SHA,checked)
 for name,digest in frozen.CODE_PINS.items():frozen.bind(ROOT/name,digest,checked)
 shared=importlib.import_module('mamba_e8w5.axis_small_base');scorer=importlib.import_module('evaluate_input_axis_residual')
 for pins in (shared.CODE,scorer.frozen.CODE):
  for name,digest in pins.items():
   if Path(name).suffix=='.py':frozen.bind(ROOT/name,digest,checked)
 context,values,expected,prior=frozen.prepare_current(args,shared,checked)
 data=importlib.import_module('mamba_e8w5.resurface_data');mk=importlib.import_module('evaluate_readapted_mk')
 manifest=frozen.read_bound(args.dataset_root/'train/manifest.json',frozen.TRAIN_MANIFEST_SHA,checked)
 if args.training_report_sha!=TRAIN_SHA or args.train_manifest_sha!=frozen.TRAIN_MANIFEST_SHA:
  raise ValueError('Exactly the previously trained final1536 candidate is required')
 payload,proof=frozen.verify_training_export(args,shared,expected,manifest,checked,prior)
 for key,value in CANDIDATE_PINS.items():
  if proof['candidate_identity'].get(key)!=value:raise ValueError('Continuation selected another candidate')
 report.update(candidate_identity=proof['candidate_identity'],export_proof=proof,checked_input_sha256=checked)
 return frozen,shared,scorer,mk,data,context,values,expected,prior,payload,proof,checked


def previous_stage(args,identity,frozen,checked):
 if args.prior_report is None or not args.prior_report_sha:raise ValueError('Passed previous stage receipt SHA required')
 previous=frozen.read_bound(args.prior_report,args.prior_report_sha,checked)
 required='dev-audit' if args.stage=='full' else 'full'
 if (previous.get('format')!=FORMAT or previous.get('complete') is not True or previous.get('stage')!=required
     or previous.get('gate',{}).get('passed') is not True or previous.get('candidate_identity')!=identity
     or previous.get('protocol_sha256')!=PROTOCOL_SHA or previous.get('script_sha256')!=sha(__file__)):
  raise ValueError('Prior stage gate/candidate/source binding differs')
 return {'file':str(args.prior_report),'sha256':args.prior_report_sha,'stage':required,'gate':previous['gate'],
  'environment_reference':previous['environment_reference']}


def run(args,report):
 frozen,shared,scorer,mk,data,context,values,expected,prior,payload,proof,checked=common(args,report)
 tokenizer=shared.runtime.SentencePieceTokenizer(ROOT/'models/source')
 if args.stage=='dev-audit':
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('DEV reanalysis is CPU-only')
  audit_dev(args,frozen,shared,scorer,mk,tokenizer,data,prior,expected,payload,proof,checked,report)
  for path,digest in checked.items():frozen.bind(path,digest,{})
  report['final_bound_input_recheck']={'file_count':len(checked),'checked_input_sha256':checked}
  return
 report['prior_stage']=previous_stage(args,proof['candidate_identity'],frozen,checked)
 # This is the first opening of raw/token full/confirm inputs; prior gate is already checked.
 prepared=frozen.load_stage_data(args,data,tokenizer,prior,checked)
 report['stage_data']={k:v for k,v in prepared.items() if k not in ('windows','cases')}
 if args.verify_only:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU preflight must hideCUDA')
  for path,digest in checked.items():frozen.bind(path,digest,{})
  report.update(complete=True,mode='verify_only',cuda_initialized=False);return
 if any(os.environ.get(name) is not None for name in PROFILE_ENV) or torch.are_deterministic_algorithms_enabled():
  raise ValueError('Do not enable the unselected deterministic/alternate numerical profile')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 torch.cuda.reset_peak_memory_stats()
 model=shared.load_model(context).eval().requires_grad_(False);report['direct_old_axis507_audit']=model._axis_load_audit
 report['installed_readapted393_audit']=shared.install_small(model,values,expected);identities=mk.identities(model)
 report['initial_current507_audit']=shared.audit_model(model,expected);report['environment']=shared.runtime.environment_receipt()
 if report['environment']!=report['prior_stage']['environment_reference']:
  raise ValueError('Native package/GPU/precision environment changed from strict DEV')
 report['environment_reference']=report['environment']
 profile_module=importlib.import_module('diagnose_output_calibration_v2')
 report['numerical_before']=profile_module.reproducibility_snapshot();report['arms']={};bank=None
 native=importlib.import_module('mamba_e8w5.resurface_native');evaluation=importlib.import_module('mamba_e8w5.evaluation')
 try:
  for arm in (BASE,ACTIVE,RESTORED):
   if arm==ACTIVE:
    bank=native.install_fp16(model,args.adapter,expected_binding=payload['binding'],expected_base_hashes=expected)
    report['adapter_before']=frozen.adapter_audit(bank,native,proof['adapter_tensor_sha256'])
   elif arm==RESTORED:
    report['adapter_after']=frozen.adapter_audit(bank,native,proof['adapter_tensor_sha256'],report['adapter_before'])
    bank.close();bank=None
   if args.stage=='full':frozen.score_ppl(model,scorer,mk,shared,expected,prepared['windows'],prepared['plan'],arm,args,report,True)
   else:mk.score_arm(model,tokenizer,evaluation,shared,expected,prepared['cases'],prepared['plan'],arm,args,report)
   if mk.identities(model)!=identities:raise ValueError('Base507 identity changed')
   write(args.report,report)
  if args.stage=='full':
   if (not scorer.exact_repeat(report['ppl'][BASE],report['ppl'][RESTORED]) or not scorer.exact_repeat(report['ppl'][BASE],prior['full_validation']['arms']['current_axis_readapted_small'])):
    raise ValueError('Historical or restored full130 PPL differs')
   report.update(ppl_restoration_exact=True,historical_current_ppl_exact=True)
   report['gate']=frozen.stage_gate('full',current_ppl=report['ppl'][BASE]['summary']['ppl'],active_ppl=report['ppl'][ACTIVE]['summary']['ppl'])
  else:
   if not frozen.mk_exact(report['arms'][BASE],report['arms'][RESTORED]):raise ValueError('Full confirmation baseline restoration differs')
   report['mk_restoration_exact']=True;report['mk_comparison']=mk.paired_comparison(report['arms'][BASE]['rows'],report['arms'][ACTIVE]['rows'])
   report['gate']=frozen.stage_gate('confirm',report['mk_comparison'])
  report.update(complete=True,mode='native_fixed_soft_continuation')
 finally:
  if bank is not None:bank.close()
  report['final_current507_audit']=shared.audit_model(model,expected);report['final_current507_identity']=mk.identities(model)
  if report['final_current507_identity']!=identities:raise ValueError('Final restored base identity differs')
  report['numerical_after']=profile_module.reproducibility_snapshot();report['gpu_memory']=shared.runtime.gpu_memory_receipt()
  settings=lambda snap:{k:v for k,v in snap.items() if k not in ('already_loaded_autotuner_metadata','scope')}
  if settings(report['numerical_before'])!=settings(report['numerical_after']):
   raise ValueError('Numerical settings changed within continuation stage')
  for path,digest in checked.items():frozen.bind(path,digest,{})
  report['final_bound_input_recheck']={'file_count':len(checked),'checked_input_sha256':checked}
  del model;gc.collect()


def self_test():
 strict={'format':'MAMBA2_RESURFACE_READAPTED_EVALUATION_V1','complete':False,'error':KNOWN_ERROR,'stage':'dev','script_sha256':FROZEN_SHA,'mk_restoration_exact':True,
  'traceback':'Traceback\nValueError: Current DEV MK output/rows differ from completed historical baseline\n'}
 process={'status':'finished','exit_code':1,'report_complete':False,'report_sha256':'x','finished_unix':2.,'started_unix':1.,
  'code_pins':{'scripts/evaluate_resurface_readapted.py':FROZEN_SHA}}
 terminal_failure(strict,process,'x')
 for key,value in [('format','another'),('complete',True),('error','ValueError(another failure)'),('mk_restoration_exact',False),('gate',{'passed':True})]:
  bad={**strict,key:value}
  try:terminal_failure(bad,process,'x')
  except ValueError:pass
  else:raise AssertionError('Invalid original failure accepted')
 for key,value in [('status','running'),('exit_code',0),('exit_code',True),('report_complete',True),('report_sha256','different')]:
  try:terminal_failure(strict,{**process,key:value},'x')
  except ValueError:pass
  else:raise AssertionError('Invalid terminal process accepted')
 # Independent toy receipt exercises the production replay validator with768
 # fixed cells; actual tokenizer/data are verified separately by dev-audit.
 mk=importlib.import_module('evaluate_readapted_mk');frozen=importlib.import_module('evaluate_resurface_readapted')
 evaluation=importlib.import_module('mamba_e8w5.evaluation')
 cases=evaluation.synthetic_mk_cases('validation',samples_per_cell=64)
 for i,row in enumerate(cases):row.update(prompt_tokens=1,prompt_token_sha256_int64le=str(i))
 rows=[];cache={'layers':56,'tensors':112,'dtype':'torch.float16','finite':True,'bytes':122028032}
 for case in cases:
  output=case['answer'] if case['condition']=='normal' else ''
  prediction=mk.predict(output)
  rows.append({**{k:v for k,v in case.items() if k!='prompt'},'output':output,'generated_ids':[1],
   'prediction':prediction,'correct':prediction==case['answer'],'finite_hidden_every_backbone_call':True,
   'cache_bytes':122028032,'prefill_cache':cache,'final_cache':cache})
 expected={str(i):str(i) for i in range(507)};content={'verified_tensor_count':507,'unchanged_content_sha256':expected}
 replayids=[x['id'] for x in rows if x['id'].endswith(('-s0','-s0-removed'))]
 replays=[{'id':row['id'],'exact_equal':True,**{k:row[k] for k in ('generated_ids','output','prediction','correct')}}
  for row in rows if row['id'] in replayids]
 entry={'complete':True,'replay_exact':True,'rows':rows,'replays':replays,**mk.summarize(rows),
  'actual507_before':content,'actual507_after':content,'identities507_before':{},'identities507_after':{},
  'identity_storage_version_unchanged':True}
 validate_mk_arm(entry,cases,{'replay_case_ids':replayids},expected,mk)
 bad=copy.deepcopy(entry);bad['replays'][0]['exact_equal']=False
 try:validate_mk_arm(bad,cases,{'replay_case_ids':replayids},expected,mk)
 except ValueError:pass
 else:raise AssertionError('False fresh-cache replay accepted')
 assert frozen.mk_exact(entry,copy.deepcopy(entry))
 bad=copy.deepcopy(entry);bad['rows'][0]['generated_ids']=[2]
 assert not frozen.mk_exact(entry,bad)
 if torch.cuda.is_initialized():raise AssertionError('Selftest initialized CUDA')
 return {'passed':True,'tests':['Accept exactly declared original failure','Reject completed/different-failure/missing-restoration/oldpassed-gate receipts','Reject running/success/boolean-code/wrong-hash process receipts','Reject768-row fresh-cache replay corruption','Reject restored generated-ID mismatch']}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('dev-audit','full','confirm'))
 p.add_argument('--report',type=Path,required=True);p.add_argument('--self-test',action='store_true');p.add_argument('--verify-only',action='store_true')
 p.add_argument('--adapter',type=Path,default=ROOT/'artifacts/resurface_readapted_v1/adapter_fp16.pt')
 p.add_argument('--training-report',type=Path,default=ROOT/'reports/resurface_readapted_v1_train.json');p.add_argument('--training-report-sha',default=TRAIN_SHA)
 p.add_argument('--smoke-report',type=Path,default=ROOT/'reports/resurface_readapted_v1_smoke.json')
 p.add_argument('--dataset-root',type=Path,default=ROOT/'training_data/resurface_readapted_v1')
 p.add_argument('--train-manifest-sha',default='dfe076ef18d5b0610e016d41d00a38948f3878b7c67d9cd8f6f46237e09d7e11')
 p.add_argument('--split-manifest-sha');p.add_argument('--prior-report',type=Path);p.add_argument('--prior-report-sha')
 p.add_argument('--strict-report',type=Path,default=ROOT/'reports/resurface_readapted_v1_dev_eval.json');p.add_argument('--strict-report-sha')
 p.add_argument('--strict-process',type=Path,default=ROOT/'reports/resurface_readapted_v1_dev_eval_process.json');p.add_argument('--strict-process-sha')
 args=p.parse_args();torch.set_num_threads(8)
 for key in ('report','adapter','training_report','smoke_report','dataset_root','prior_report','strict_report','strict_process'):
  if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve())
 if not args.self_test and args.stage is None:p.error('Stage required')
 if args.report.parent!=ROOT/'reports' or args.report.exists() or args.report.with_suffix('.json.tmp').exists():raise ValueError('Fresh report required')
 if (args.self_test or args.stage=='dev-audit' or args.verify_only) and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU work must hideCUDA')
 report={'format':FORMAT,'complete':False,'stage':args.stage,'script_sha256':sha(__file__),'protocol_sha256':PROTOCOL_SHA,
  'original_training_protocol_sha256':'469063147d892282a2bdad8558afca8466d6f365f1efe14ecbba7417e9857dd6',
  'gate_mode':'soft','original_strict_protocol_pass_claimed':False,'post_observation_prerequisite_amendment':True,
  'fitting_performed':False,'publication_performed':False,'automatic_next_stage':False}
 started=time.monotonic()
 try:
  report['self_test']=self_test()
  if args.self_test:report.update(complete=True,mode='self_test',cuda_initialized=False)
  else:run(args,report)
 except BaseException as error:
  report.update(complete=False,error=repr(error),traceback=traceback.format_exc());raise
 finally:
  report['elapsed_seconds']=time.monotonic()-started;write(args.report,report)
 print(json.dumps({'complete':report['complete'],'stage':args.stage,'gate':report.get('gate')}),flush=True)


if __name__=='__main__':main()
