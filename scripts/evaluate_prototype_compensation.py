#!/usr/bin/env python3
"""Independently verify and evaluate serialized learned E8-derived prototypes.

CPU --verify-only checks final128 checkpoint/table identities and provenance.
Quality uses three native model arms: original FP16 source, fixed all-small
E8/W5 baseline, and independently reloaded/decoded prototype candidate. No
training bank forward, checkpoint selection, MK, test split or publication.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5 import small_overlay
from mamba_e8w5.codec import read_e8, load_reference_primitives
from mamba_e8w5.learned_e8_codebook import read_prototypes, decode_e8_with_prototypes, FILE_BYTES
from mamba_e8w5.runtime import (MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256,
    SentencePieceTokenizer, load_source_model, load_quantized_model, token_digest,
    environment_receipt, gpu_memory_receipt)
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from evaluate_small_compensation import (sha, write_json, window_plan, check_arm,
    audit_parameter_coverage, audit_selected_values, audit_frozen_values)
from audit_decoded import audit_loaded_model, tensor_sha_fp16

PROTOCOL_SHA = 'e50295425dcd2f5a09b5fea4254b9c6f6f38744bc73583f93621ac1ce942cb9d'
DATA_SHA = 'a485424d1b7894eb1d5a6361ab4c25fdb4f6e5bedc56d7a15a7e4add2c454bcd'
TOKEN_FILE_SHA = 'b569cffab846e1088c9648c70c233fe54a0686fce86f028f551656f5bf244021'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
BASELINE_SHA = '957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206'
HELPER_SHA = '2102d406da33482303f0dd9a6446b1c070fdf28ecade70e9ab8da9cdfd0202be'
FOUNDATION_SHA = 'b244b93dabb9a85487f4745588e7c0dd922274768223da94bf368aecfe10843f'
LABELS = tuple(f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj'))
ARMS = ('source_fp16', 'best_small_e8w5', 'prototype_e8w5')
TARGETS = 264764
TRAIN_CODE_PATHS = {'scripts/train_prototype_compensation.py', 'mamba_e8w5/prototype_training.py',
                    'mamba_e8w5/learned_e8_codebook.py'}
FROZEN = {
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
    'mamba_e8w5/evaluation.py': 'ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089',
    'mamba_e8w5/small_overlay.py': '3512c1b232c4039c16a26ffc0fe8c2eac1be6bb9c704ccbcb1d4e8e7788a1286',
    'mamba_e8w5/norm_training.py': '25ac3ec6ba167902014018f99fbf8dbbc659e0e4903f80baf28e28a4a3e74087',
    'scripts/train_norm_compensation.py': 'da1818af295738cfa8a012b35ff0d1f007c410f27988d9bdaf88425fbedc47f2',
}


def expected_schedule():
    generator = torch.Generator(device='cpu').manual_seed(20260927)
    return [{'epoch': epoch, 'position_in_epoch': position, 'window': index}
        for epoch in range(4) for position,index in enumerate(torch.randperm(32,generator=generator).tolist())]


def verify_accounting(state):
    schedule = expected_schedule()
    actual_schedule = state.get('schedule')
    if (not isinstance(actual_schedule,list) or actual_schedule != schedule
            or any(not isinstance(row,dict) or set(row)!={'epoch','position_in_epoch','window'}
                   or any(type(value)is not int for value in row.values()) for row in actual_schedule)
            or type(state.get('successful_updates'))is not int or state['successful_updates'] != 128
            or type(state.get('attempts'))is not int or type(state.get('overflow_retries'))is not int):
        raise ValueError('Only the declared final128 schedule is eligible')
    successes = overflows = 0
    history = state.get('history', [])
    for number, row in enumerate(history, 1):
        if (successes >= 128 or type(row.get('attempt')) is not int or row['attempt'] != number
                or type(row.get('overflow')) is not bool or row.get('schedule_entry') != schedule[successes]
                or not isinstance(row.get('schedule_entry'),dict)
                or any(type(value)is not int for value in row['schedule_entry'].values())
                or type(row.get('targets'))is not int or row['targets'] != 2047
                or type(row.get('successful_updates_after'))is not int):
            raise ValueError('Attempt history differs from the declared schedule')
        overflows += int(row['overflow'])
        successes += int(not row['overflow'])
        if row.get('successful_updates_after') != successes:
            raise ValueError('Incorrect successful-update history')
    if (successes != 128 or overflows > 8 or len(history) != 128+overflows
            or state.get('attempts') != len(history) or state.get('overflow_retries') != overflows):
        raise ValueError('Final update/overflow accounting differs')
    return {'successful_updates': 128, 'attempts': len(history), 'overflow_retries': overflows,
            'successful_target_exposures': 262016, 'attempted_target_exposures': len(history)*2047}


def verify_checkpoint_tables(state, tables, binding, frozen_hashes, smoke_sha):
    if (state.get('format') != 'MAMBA2_PROTOTYPE_TRAINING_CHECKPOINT_V1'
            or state.get('binding') != binding or state.get('parent_manifest_sha256') != PARENT_SHA
            or state.get('small_manifest_sha256') != SMALL_SHA or state.get('smoke_report_sha256') != smoke_sha
            or state.get('frozen_model_sha256') != frozen_hashes or state.get('resumable') is not False
            or state.get('reason') != 'final'):
        raise ValueError('Final checkpoint provenance differs')
    counts = verify_accounting(state)
    if set(state.get('masters', {})) != set(LABELS) or set(tables) != set(LABELS):
        raise ValueError('Exactly112 final prototype tables/masters are required')
    hashes = {}
    for label in LABELS:
        master, table = state['masters'][label], tables[label]
        if (not isinstance(master, torch.Tensor) or master.dtype != torch.float32
                or tuple(master.shape) != (256,8) or not torch.isfinite(master).all()
                or table.dtype != torch.float16 or tuple(table.shape) != (256,8)):
            raise ValueError(f'Invalid final master/table: {label}')
        rounded = master.half()
        if not torch.isfinite(rounded).all() or tensor_sha_fp16(rounded) != tensor_sha_fp16(table):
            raise ValueError(f'Final master.half() differs from serialized table: {label}')
        hashes[label] = tensor_sha_fp16(table)
    return {**counts, 'verified_tables': 112, 'verified_parameters': 229376,
            'rounded_master_equals_serialized_fp16_sha256': hashes}


def checked_file(directory, name, receipt):
    if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected safe flat file name')
    path = Path(directory)/name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != receipt['bytes'] or sha(path) != receipt['sha256']:
        raise ValueError(f'File identity differs: {name}')
    return path


def verify_training_data(directory):
    path = Path(directory)/'manifest.json'
    if sha(path) != DATA_SHA:
        raise ValueError('Fresh training-data manifest differs')
    data = json.loads(path.read_text())
    if (data.get('format') != 'MAMBA2_PROTOTYPE_TRAIN_WINDOWS_V1' or data.get('complete') is not True
            or data.get('split') != 'train' or data.get('evaluation_data_used') is not False
            or data.get('protocol_sha256') != PROTOCOL_SHA or data.get('nwin') != 32 or data.get('seqlen') != 2048):
        raise ValueError('Training-data scope differs')
    token_path = checked_file(directory, 'training_tokens.pt',
        {'bytes': data['token_windows_file_bytes'], 'sha256': TOKEN_FILE_SHA})
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    if (tokens.dtype != torch.int64 or tuple(tokens.shape) != (32,2048)
            or tokens.min() < 0 or tokens.max() >= 256000
            or token_digest(tokens.flatten().numpy()) != data['selected_tokens_sha256_int64le']):
        raise ValueError('Training-token tensor identity differs')
    excluded = [start for starts in data['excluded_starts'].values() for start in starts]
    eligible = [start for start in range(0,2533678-2048+1,2048)
                if not any(start < old+2048 and old < start+2048 for old in excluded)]
    starts = [eligible[i*(len(eligible)-1)//31] for i in range(32)]
    if len(eligible) != 871 or starts != data['starts']:
        raise ValueError('Disjoint training-window selection differs')
    for row, start, window in zip(data['windows'], starts, tokens):
        if row != {'start': start, 'token_sha256_int64le': token_digest(window.numpy())}:
            raise ValueError('Training-window content differs')
    return data


def verify_export(args):
    """CPU-only independent ledger/master validation; no training bank imported."""
    pins = {args.protocol: PROTOCOL_SHA, args.baseline_report: BASELINE_SHA,
        args.parent_dir/'manifest.json': PARENT_SHA, args.small_dir/'manifest.json': SMALL_SHA,
        args.small_dir/'other_fp16.pt': SMALL_VALUES_SHA,
        ROOT/'scripts/evaluate_small_compensation.py': HELPER_SHA,
        ROOT/'mamba_e8w5/learned_e8_codebook.py': FOUNDATION_SHA}
    pins.update({ROOT/name: digest for name,digest in FROZEN.items()})
    for path,digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f'Pinned input/code differs: {path}')
    baseline = json.loads(args.baseline_report.read_text())
    if baseline.get('complete') is not True or sha(ROOT/'scripts/audit_decoded.py') != baseline['loaded_tensor_audit_source_sha256']:
        raise ValueError('Baseline/audit helper differs')
    parent, small, base_receipt = small_overlay.verify_overlay(args.parent_dir, args.small_dir,
        initialization_dir=args.initialization_dir)
    data = verify_training_data(args.data_dir)
    manifest_path = args.overlay_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'MAMBA2_PROTOTYPE_COMPENSATION_V1' or manifest.get('complete') is not True
            or manifest.get('parent_manifest_sha256') != PARENT_SHA or manifest.get('small_manifest_sha256') != SMALL_SHA
            or manifest.get('small_values_sha256') != SMALL_VALUES_SHA or manifest.get('model_config') != MODEL_CONFIG):
        raise ValueError('Invalid prototype overlay or baseline identity')
    binding = manifest['binding']
    expected = {'parent_manifest_sha256': PARENT_SHA, 'small_manifest_sha256': SMALL_SHA,
        'small_values_sha256': SMALL_VALUES_SHA, 'source_checkpoint_sha256': SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256': TOKENIZER_SHA256, 'baseline_report_sha256': BASELINE_SHA,
        'protocol_sha256': PROTOCOL_SHA, 'training_data_manifest_sha256': DATA_SHA,
        'training_tokens_sha256': TOKEN_FILE_SHA, 'frozen_code_sha256': FROZEN}
    for key,value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f'Prototype binding differs: {key}')
    code = binding['code_sha256']
    if set(code) != TRAIN_CODE_PATHS:
        raise ValueError('Training code binding coverage differs')
    for name,digest in code.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Training source differs: {name}')
    hp = binding['hyperparameters']
    required_hp = {'successful_updates':128,'maximum_overflow_retries':8,'maximum_attempts':136,
        'epochs':4,'windows_per_epoch':32,'stored_window_tokens':2048,'targets_per_update':2047,
        'seed':20260927,'optimizer':'AdamW','learning_rate':.001,'betas':[.9,.999],'epsilon':1e-8,
        'weight_decay':0.,'gradient_clip_norm':1.,'ce_coefficient':.5,'teacher_to_student_kl_coefficient':.5,
        'temperature':1.,'logits_chunk_tokens':64,'resume':False,'checkpoint_every_successful_updates':32,
        'loss_scaler':{'init_scale':1024.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000}}
    if any(hp.get(key) != value for key,value in required_hp.items()):
        raise ValueError('Fixed optimization recipe differs')
    expected_inventory = {label:{'file':label+'.proto','shape':[256,8],'dtype':'float16','numel':2048} for label in LABELS}
    expected_names = {label+'.proto' for label in LABELS}
    if manifest.get('tables_inventory') != expected_inventory or set(manifest.get('files',{})) != expected_names:
        raise ValueError('Expected exactly112 declared prototype files')
    if {p.name for p in args.overlay_dir.iterdir()} != expected_names|{'manifest.json'}:
        raise ValueError('Unlisted prototype overlay file')
    tables = {}
    for label in LABELS:
        name = label+'.proto'
        entry = manifest['files'][name]
        path = checked_file(args.overlay_dir, name, entry)
        if entry['bytes'] != FILE_BYTES or entry['header_bytes'] != 32 or entry['payload_bytes'] != 4096:
            raise ValueError('Prototype file capacity differs')
        table = read_prototypes(path)
        if entry['payload_sha256'] != tensor_sha_fp16(table):
            raise ValueError('Table payload digest differs')
        tables[label] = table
    frozen_hashes = {entry['name']:entry['decoded_fp16_sha256']
                     for entry in baseline['parent_loaded_tensor_audit']['tensors']}
    frozen_hashes.update(baseline['small_loaded_tensor_audit']['small']['loaded_and_export_fp16_sha256'])
    if len(frozen_hashes) != 507 or manifest.get('frozen_base_hash_ledger') != frozen_hashes:
        raise ValueError('Frozen baseline507 hash ledger differs')
    receipt = manifest['training_receipt']
    for path,key in ((args.training_report,'report_sha256'), (args.smoke_report,'smoke_report_sha256'),
                     (args.training_checkpoint,'checkpoint_sha256')):
        if sha(path) != receipt[key]:
            raise ValueError(f'Final training receipt differs: {key}')
    train = json.loads(args.training_report.read_text())
    smoke = json.loads(args.smoke_report.read_text())
    for report,mode in ((train,'train'), (smoke,'smoke')):
        if (report.get('format') != 'MAMBA2_PROTOTYPE_COMPENSATION_TRAIN_V1' or report.get('complete') is not True
                or report.get('mode') != mode or report.get('binding') != binding
                or report.get('split') != 'train' or report.get('evaluation_data_used') is not False):
            raise ValueError(f'{mode} provenance/scope differs')
    if smoke.get('passed') is not True or smoke.get('trial_state_discarded') is not True:
        raise ValueError('Required smoke did not pass/discard trial state')
    if (type(train.get('final_step'))is not int or train['final_step'] != 128
            or train.get('final_export',{}).get('files') != manifest['files']
            or train.get('smoke_report_sha256') != receipt['smoke_report_sha256']
            or train.get('final_export',{}).get('checkpoint128_master_to_fp16_bitwise_equal') is not True
            or train.get('final_checkpoint',{}).get('sha256') != receipt['checkpoint_sha256']
            or train['final_checkpoint']['bytes'] != args.training_checkpoint.stat().st_size):
        raise ValueError('Final export/training receipt differs')
    state = torch.load(args.training_checkpoint,map_location='cpu',weights_only=True)
    rounding = verify_checkpoint_tables(state,tables,binding,frozen_hashes,receipt['smoke_report_sha256'])
    if train.get('history') != state['history'] or train.get('schedule') != state['schedule']:
        raise ValueError('Training report/checkpoint schedule differs')
    for key in ('successful_updates','attempts','overflow_retries'):
        if (type(train.get(key))is not int or type(receipt.get(key))is not int
                or train[key] != rounding[key] or receipt[key] != rounding[key]):
            raise ValueError('Final training accounting differs')
    for key in ('successful_target_exposures','attempted_target_exposures'):
        if train.get(key) != rounding[key]:
            raise ValueError('Training target exposure count differs')
    if train.get('frozen_parameter_audit',{}).get('after_sha256') != frozen_hashes:
        raise ValueError('Final training frozen hashes differ')
    table_bytes = sum(entry['bytes'] for entry in manifest['files'].values())
    logical = base_receipt['logical_candidate_data_bytes']+table_bytes
    storage = {'base_logical_data_bytes':base_receipt['logical_candidate_data_bytes'],
        'table_payload_bytes':458752,'table_file_bytes':table_bytes,
        'logical_inference_data_bytes':logical,'prototype_manifest_bytes':manifest_path.stat().st_size,
        'physical_parent_small_prototype_directory_bytes':sum(p.stat().st_size
            for directory in (args.parent_dir,args.small_dir,args.overlay_dir) for p in directory.iterdir()),
        'scope':'Raw resolved model/overlay files and measured manifests; tokenizer/software/reports/training/source excluded.'}
    for key in ('base_logical_data_bytes','table_payload_bytes','table_file_bytes','logical_inference_data_bytes'):
        if manifest['storage'].get(key) != storage[key]:
            raise ValueError(f'Manifest storage accounting differs: {key}')
    if table_bytes != 462336 or manifest['storage'].get('manifest_bytes') != manifest_path.stat().st_size:
        raise ValueError('Actual table/manifest capacity differs')
    checked_paths = {str(path.resolve()):digest for path,digest in pins.items()}
    checked_paths.update({str((ROOT/name).resolve()):digest for name,digest in code.items()})
    checked_paths.update({str((args.overlay_dir/name).resolve()):entry['sha256']
                          for name,entry in manifest['files'].items()})
    for path in (manifest_path,args.training_report,args.smoke_report,args.training_checkpoint,
                 args.data_dir/'manifest.json',args.data_dir/'training_tokens.pt',Path(__file__)):
        checked_paths[str(path.resolve())] = sha(path)
    result = {'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
        'prototype_manifest_sha256':sha(manifest_path),'binding':binding,'training_receipt':receipt,
        'independent_final128_table_audit':rounding,'verified_file_count':112,
        'frozen_baseline507_hashes':frozen_hashes,'data_manifest_sha256':DATA_SHA,
        'storage':storage,'checked_input_sha256':checked_paths}
    return parent,small,manifest,baseline,result


def compare_results(results, plan):
    if set(results) != set(ARMS):
        raise ValueError('Exactly three native full-validation arms are required')
    for result in results.values():
        check_arm(result,plan)
    source,base,candidate = (results[name] for name in ARMS)
    paired = []
    for i,row in enumerate(plan):
        values = {name:results[name]['windows'][i] for name in ARMS}
        paired.append({**row,'nll':{name:value['nll']for name,value in values.items()},
            'ppl':{name:value['ppl']for name,value in values.items()},
            'candidate_minus_baseline_mean_nll':(values[ARMS[2]]['nll']-values[ARMS[1]]['nll'])/row['target_tokens'],
            'candidate_minus_source_mean_nll':(values[ARMS[2]]['nll']-values[ARMS[0]]['nll'])/row['target_tokens']})
    comparison = {'ppl':{name:results[name]['ppl']for name in ARMS},
        'candidate_vs_baseline_ppl_relative_change':candidate['ppl']/base['ppl']-1,
        'candidate_vs_source_ppl_relative_change':candidate['ppl']/source['ppl']-1,
        'baseline_vs_source_ppl_relative_change':base['ppl']/source['ppl']-1,
        'candidate_minus_baseline_mean_nll':(candidate['nll']-base['nll'])/TARGETS,
        'meaningful_improvement_reference':{'required_reduction':.01,'met':candidate['ppl']<=.99*base['ppl']},
        'source_plus5_percent_reference':{'maximum_ppl':1.05*source['ppl'],'met':candidate['ppl']<=1.05*source['ppl']},
        'window_counts':{'improved':sum(row['candidate_minus_baseline_mean_nll']<0 for row in paired),
            'unchanged':sum(row['candidate_minus_baseline_mean_nll']==0 for row in paired),
            'worsened':sum(row['candidate_minus_baseline_mean_nll']>0 for row in paired)}}
    return comparison,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('overlay-dir','training-report','training-checkpoint','smoke-report','report'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    for key,default in {'source-dir':'models/source','parent-dir':'artifacts/e8w5_v1',
        'small-dir':'artifacts/small_compensation_v1','initialization-dir':'artifacts/norm_compensation_v1',
        'data-dir':'training_data/prototype_compensation_v1','protocol':'docs/PROTOTYPE_COMPENSATION_PROTOCOL.md',
        'baseline-report':'reports/small_compensation_v1_eval.json'}.items():
        parser.add_argument('--'+key,type=Path,default=Path(default))
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    started=time.perf_counter()
    parent,small,manifest,prior,receipt=verify_export(args)
    report={'format':'MAMBA2_PROTOTYPE_COMPENSATION_EVALUATION_V1','complete':False,
        'mode':'cpu_integrity_only' if args.verify_only else 'three_arm_full_validation',
        'script_sha256':sha(__file__),'imported_window_audit_helper_sha256':HELPER_SHA,
        'integrity':receipt,'arm_order':list(ARMS),'results':{},'same_process':True,
        'quality_forward':'native model only; independent serialized table decode; no training bank forward',
        'limitations':['Validation has informed development; this is not untouched test evidence.',
            'No MK/test/training/checkpoint selection/publication is performed here.',
            'Prototype files add462336bytes plus manifest; this is not same-byte compression.',
            'Evaluation expands weights to FP16; timings are diagnostics, not compressed-runtime throughput.']}
    write_json(args.report,report)
    if args.verify_only:
        report.update(complete=True,elapsed_seconds=time.perf_counter()-started)
        write_json(args.report,report)
        print(json.dumps({'complete':True,'mode':report['mode'],'report_sha256':sha(args.report)}),flush=True)
        return
    report['environment']=environment_receipt()
    tokenizer=SentencePieceTokenizer(args.source_dir)
    ids,dataset=load_wikitext_tokens(tokenizer,'validation',WIKITEXT_REVISION)
    windows=ppl_windows(ids,2048,None);plan=window_plan(windows)
    if dataset!=prior['dataset'] or plan!=prior['window_plan']:
        raise ValueError('Frozen validation dataset/window hashes differ')
    report.update(dataset=dataset,window_plan=plan,
        protocol={'target_tokens':TARGETS,'windows':130,'maximum_targets_per_window':2048,
            'execution':'prefill','logits_chunk_tokens':64,'fresh_state_per_window':True})
    write_json(args.report,report);torch.cuda.reset_peak_memory_stats()

    def score(name,model):
        print(f'[prototype evaluation] Scoring {name}',flush=True)
        result=evaluate_ppl(model,windows,execution='prefill',logits_chunk=64)
        check_arm(result,plan);report['results'][name]=result;write_json(args.report,report)

    source=load_source_model(args.source_dir)
    report['source_package_receipt']=source._package_receipt
    report['source_coverage']=audit_parameter_coverage(source)
    score('source_fp16',source)
    del source;gc.collect();torch.cuda.empty_cache()
    model=load_quantized_model(args.parent_dir)
    report['parent_package_receipt']=model._package_receipt
    report['parent507_loaded_audit']=audit_loaded_model(model,parent,args.parent_dir)
    small_overlay.apply_overlay(model,args.small_dir,small)
    report['small_overlay_application']=model._small_overlay_receipt
    small_values=torch.load(args.small_dir/'other_fp16.pt',map_location='cpu',weights_only=True)
    report['baseline393_export_audit']=audit_selected_values(model,small_values)
    report['baseline507_actual_content_audit']=audit_frozen_values(model,receipt['frozen_baseline507_hashes'])
    report['baseline_coverage']=audit_parameter_coverage(model)
    score('best_small_e8w5',model)
    projection_names={parent['matrices'][label]['source_key'] for label in LABELS}
    unchanged={name:value for name,value in dict(model.named_parameters()).items() if name not in projection_names}
    if len(unchanged)!=395:raise ValueError('Expected393 small plus2 W5 unchanged tensors')
    expected_unchanged={name:receipt['frozen_baseline507_hashes'][name] for name in unchanged}
    codebook,_=load_reference_primitives();codebook=codebook.cuda()
    decoded_hashes={}
    for position,label in enumerate(LABELS,1):
        path=checked_file(args.overlay_dir,label+'.proto',manifest['files'][label+'.proto'])
        table=read_prototypes(path,device='cpu')
        original_path=checked_file(args.parent_dir,label+'.e8',parent['files'][label+'.e8'])
        payload=read_e8(original_path)
        value=decode_e8_with_prototypes(payload,codebook,table,device='cuda')
        name=parent['matrices'][label]['source_key'];owner,leaf=name.rsplit('.',1)
        if list(value.shape)!=parent['matrices'][label]['shape'] or value.dtype!=torch.float16:
            raise ValueError(f'Independent prototype decode geometry differs: {label}')
        digest=tensor_sha_fp16(value)
        setattr(model.get_submodule(owner),leaf,torch.nn.Parameter(value,requires_grad=False))
        if tensor_sha_fp16(model.get_parameter(name))!=digest:
            raise ValueError(f'Installed native projection differs: {label}')
        decoded_hashes[label]={'source_key':name,'shape':list(value.shape),'decoded_fp16_sha256':digest,
            'table_sha256':sha(path),'original_e8_sha256':parent['files'][label+'.e8']['sha256']}
        if position%16==0 or position==112:print(f'[prototype evaluation] Independent native decode {position}/112',flush=True)
    del codebook,payload,value;gc.collect();torch.cuda.empty_cache()
    if any(model.get_parameter(name) is not value for name,value in unchanged.items()):
        raise ValueError('Candidate replaced a frozen small/vocabulary tensor object')
    report['candidate112_decoded_fp16']=decoded_hashes
    report['candidate395_unchanged_content_audit']=audit_frozen_values(model,expected_unchanged)
    report['candidate_coverage']=audit_parameter_coverage(model)
    score('prototype_e8w5',model)
    report['comparison'],report['paired_windows']=compare_results(report['results'],plan)
    for path,digest in receipt['checked_input_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Bound input changed during evaluation: {path}')
    report.update(complete=True,elapsed_seconds=time.perf_counter()-started,gpu_memory=gpu_memory_receipt())
    write_json(args.report,report)
    print(json.dumps({'complete':True,'comparison':report['comparison'],'report_sha256':sha(args.report)}),flush=True)


if __name__=='__main__':main()
