#!/usr/bin/env python3
"""Posthoc fit-versus-fresh TRAIN diagnostic for one fixed final rank4 export.

This never trains, selects checkpoints, computes validation/test metrics or
advances a candidate. Source/all-small/final-candidate CE and teacher KL are
measured on the original256 fitting windows and64 new disjoint TRAIN windows.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate_low_rank_compensation as native

FINAL_EVALUATION_SHA = 'efb4264bc65efded5fa9b05f9ea689542f9feaf23d2108a1f7c335bee6c67979'
FINAL_MANIFEST_SHA = '6549fac5e2531ff73bfcca54e4615ae35389f2a4e8f39b32fac3c583515f018a'
EVALUATOR_SHA = 'f0d1285276d1b4ea3bb132b98cb412a9718289b715d70ee1fa961c08bc3b0bd7'
SEQLEN = 2048
TARGETS_PER_WINDOW = 2047
SET_ORDER = ('fit_train256','fresh_train64')
ARMS = ('source_fp16','best_small_e8w5','low_rank_e8w5')
DIAGNOSTIC_PROTOCOL = {
    'posthoc':True,'candidate':'unchanged final1024 serialized rank4 export',
    'purpose':'describe fitting versus new TRAIN generalization; no selection or candidate advancement',
    'split':'train','fit_windows':256,'fresh_windows':64,'stored_window_tokens':2048,
    'targets_per_window':2047,'fresh_state_per_window':True,'batch_windows':1,
    'exclusions':'calibration32+small256+crossmoment48+prototype32+current_low_rank256',
    'fresh_selection':'from583 remaining eligible2048-token grid blocks; eligible[floor(i*582/63)], i=0..63',
    'arm_pass_order':['source_with_all_small','source_with_final_low_rank'],
    'set_order':list(SET_ORDER),'temperature':1.0,'logits_chunk_tokens':64,
    'head_dtype':'native FP16 GEMM','loss_dtype':'FP32','kl_direction':'teacher_source || student',
    'kl_reduction':'sum over the full vocabulary and targets, divide by total targets',
    'teacher_repeat':'exact per-chunk CE sums and aggregate CE on identical windows across both passes',
    'precision':'TF32 disabled; highest FP32 matmul precision; autocast disabled',
    'checkpoint_sweep':False,'retraining':False,'validation_or_test_computation':False,
    'candidate_advancement':False,
}


def protocol_digest():
    return hashlib.sha256(json.dumps(DIAGNOSTIC_PROTOCOL,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def select_fresh_starts(data):
    """Posthoc deterministic interval arithmetic; no loss/content-based selection."""
    excluded = {name:list(starts) for name,starts in data['excluded_starts'].items()}
    excluded['current_low_rank_training'] = list(data['starts'])
    counts = {'original_calibration':32,'all_small_training':256,'crossmoment_fit':32,
        'crossmoment_heldout':16,'prototype_training':32,'current_low_rank_training':256}
    if {name:len(values) for name,values in excluded.items()} != counts:
        raise ValueError('Prior TRAIN interval family counts differ')
    if any(type(s) is not int or s < 0 or s+SEQLEN > 2533678
           for values in excluded.values() for s in values):
        raise ValueError('Invalid excluded TRAIN interval')
    prior = [s for values in excluded.values() for s in values]
    grid = list(range(0,2533678-SEQLEN+1,SEQLEN))
    eligible = [s for s in grid if all(s+SEQLEN <= old or old+SEQLEN <= s for old in prior)]
    if len(eligible) != 583:
        raise ValueError('Expected583 fresh TRAIN grid blocks')
    starts = [eligible[i*582//63] for i in range(64)]
    internal = sum(max(0,a+SEQLEN-b) for a,b in zip(starts,starts[1:]))
    overlaps = {name:sum(max(0,min(s+SEQLEN,old+SEQLEN)-max(s,old))
        for s in starts for old in values) for name,values in excluded.items()}
    if (len(set(starts)) != 64 or internal or any(overlaps.values()) or starts[0] != 12288
            or starts[-1] != 2521088 or min(b-a for a,b in zip(starts,starts[1:])) != 32768):
        raise ValueError('Fresh windows overlap fitting or each other')
    return starts, {'grid_blocks':len(grid),'eligible_blocks':583,'selected_blocks':64,
        'internal_overlap_tokens':internal,'previous_family_overlap_tokens':overlaps,
        'all_previous_fitting_overlap_tokens':sum(overlaps.values()),
        'minimum_start_gap':min(b-a for a,b in zip(starts,starts[1:]))}


def verify_inputs():
    """Rehash the already verified final export's ledger, without checkpoint search."""
    report_path = ROOT/'reports/low_rank_compensation_v1_eval.json'
    manifest_path = ROOT/'artifacts/low_rank_compensation_v1/manifest.json'
    if (native.sha(report_path) != FINAL_EVALUATION_SHA
            or native.sha(manifest_path) != FINAL_MANIFEST_SHA
            or native.sha(ROOT/'scripts/evaluate_low_rank_compensation.py') != EVALUATOR_SHA):
        raise ValueError('Fixed final-model diagnostic identity differs')
    prior = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (prior.get('complete') is not True or prior.get('mode') != 'three_arm_full_validation'
            or prior.get('script_sha256') != EVALUATOR_SHA
            or prior['integrity']['low_rank_manifest_sha256'] != FINAL_MANIFEST_SHA
            or manifest.get('complete') is not True
            or manifest['binding'] != prior['integrity']['binding']
            or manifest['training_receipt'] != prior['integrity']['training_receipt']):
        raise ValueError('Completed final-model provenance differs')
    # Rebase the pinned report's repository-relative input paths on this checkout.
    old_roots = {Path(p).parents[1] for p,digest in prior['integrity']['checked_input_sha256'].items()
                 if Path(p).name == 'evaluate_low_rank_compensation.py' and digest == EVALUATOR_SHA}
    if len(old_roots) != 1:
        raise ValueError('Cannot identify the pinned source repository root')
    old_root = next(iter(old_roots))
    checked = {}
    for name,digest in prior['integrity']['checked_input_sha256'].items():
        relative = Path(name).relative_to(old_root)
        path = ROOT/relative
        if path.is_symlink() or not path.is_file() or native.sha(path) != digest:
            raise ValueError(f'Final-model input changed: {relative}')
        checked[str(path)] = digest
    checked[str(report_path)] = FINAL_EVALUATION_SHA
    checked[str(Path(__file__).resolve())] = native.sha(__file__)
    if manifest['factors_inventory'] != native.expected_inventory():
        raise ValueError('Final factor geometry differs')
    folder = manifest_path.parent
    if {p.name for p in folder.iterdir()} != set(manifest['files'])|{'manifest.json'}:
        raise ValueError('Unexpected final factor file inventory')
    data = native.verify_training_data(ROOT/'training_data/low_rank_compensation_v1')
    baseline = prior['baseline507_actual_content_audit']['unchanged_content_sha256']
    candidate = prior['candidate507_post_evaluation_actual_content_audit']['unchanged_content_sha256']
    if (len(baseline) != 507 or len(candidate) != 507
            or baseline != prior['integrity']['frozen_baseline507_hashes']
            or baseline != manifest['frozen_base_hash_ledger']):
        raise ValueError('Pinned actual507 tensor coverage differs')
    expected = dict(baseline)
    if set(prior['candidate112_merged_fp16']) != set(native.LABELS):
        raise ValueError('Pinned final112 merged tensor coverage differs')
    for label,entry in native.expected_inventory().items():
        row = prior['candidate112_merged_fp16'][label]
        if row['source_key'] != entry['source_key'] or row['factor_file_sha256'] != manifest['files'][entry['file']]['sha256']:
            raise ValueError('Pinned final factor-to-native mapping differs')
        expected[entry['source_key']] = row['merged_fp16_sha256']
    if expected != candidate:
        raise ValueError('Final candidate changes tensors outside its112 projections')
    return prior,manifest,data,baseline,candidate,checked


def prepare_sets(data):
    tokenizer = native.SentencePieceTokenizer(ROOT/'models/source')
    ids,dataset = native.load_wikitext_tokens(tokenizer,'train',native.WIKITEXT_REVISION)
    if dataset != data['dataset'] or tuple(ids.shape) != (2533678,) or ids.dtype != torch.int64:
        raise ValueError('Original TRAIN token stream identity differs')
    fit = torch.load(ROOT/'training_data/low_rank_compensation_v1/training_tokens.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(window,ids[start:start+SEQLEN]) for start,window in zip(data['starts'],fit)):
        raise ValueError('Stored fitting windows differ from original TRAIN stream')
    fresh_starts,overlap = select_fresh_starts(data)
    fresh = torch.stack([ids[start:start+SEQLEN] for start in fresh_starts])
    values = {'fit_train256':fit,'fresh_train64':fresh}
    starts = {'fit_train256':data['starts'],'fresh_train64':fresh_starts}
    plans = {name:[{'index':i,'start':start,'stored_tokens':SEQLEN,'target_tokens':TARGETS_PER_WINDOW,
        'token_sha256_int64le':native.token_digest(window.numpy())}
        for i,(start,window) in enumerate(zip(starts[name],values[name]))] for name in SET_ORDER}
    if [len(values[name]) for name in SET_ORDER] != [256,64]:
        raise ValueError('Diagnostic set sizes differ')
    metadata = {'dataset':dataset,'selection_uses_model_outputs':False,'fresh_overlap':overlap,
        'sets':{name:{'window_count':len(values[name]),'target_tokens':len(values[name])*TARGETS_PER_WINDOW,
            'stored_token_tensor_sha256_int64le':native.token_digest(values[name].flatten().numpy()),
            'windows':plans[name]} for name in SET_ORDER}}
    return values,plans,metadata


@torch.inference_mode()
def score_window(source,student,window,chunk_tokens=64):
    """Native FP16 head GEMMs; full-vocabulary FP32 CE and KL(T=1)."""
    device = source.backbone.embedding.weight.device
    if student.backbone.embedding.weight.device != device:
        raise ValueError('Source and student must share a device')
    ids,targets = window[None,:-1].to(device),window[None,1:].to(device)
    with torch.autocast(device_type=device.type,enabled=False):
        teacher_hidden = source.backbone(ids)
        student_hidden = student.backbone(ids)
        if teacher_hidden.dtype != torch.float16 or student_hidden.dtype != torch.float16:
            raise ValueError('Expected native FP16 hidden states')
        if not torch.isfinite(teacher_hidden).all() or not torch.isfinite(student_hidden).all():
            raise FloatingPointError('Nonfinite diagnostic hidden state')
        teacher_chunks,student_chunks,kl_chunks = [],[],[]
        for first in range(0,targets.shape[1],chunk_tokens):
            end = min(first+chunk_tokens,targets.shape[1])
            teacher_logits = source.lm_head(teacher_hidden[:,first:end]).float()
            student_logits = student.lm_head(student_hidden[:,first:end]).float()
            if teacher_logits.shape != student_logits.shape or not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
                raise FloatingPointError('Invalid diagnostic vocabulary logits')
            truth = targets[:,first:end].flatten()
            teacher_ce = F.cross_entropy(teacher_logits.flatten(0,1),truth,reduction='sum')
            student_ce = F.cross_entropy(student_logits.flatten(0,1),truth,reduction='sum')
            teacher_logp = F.log_softmax(teacher_logits,dim=-1)
            student_logp = F.log_softmax(student_logits,dim=-1)
            kl = F.kl_div(student_logp,teacher_logp,reduction='sum',log_target=True)
            if not all(torch.isfinite(value) for value in (teacher_ce,student_ce,kl)):
                raise FloatingPointError('Nonfinite diagnostic CE/KL')
            teacher_chunks.append(float(teacher_ce))
            student_chunks.append(float(student_ce))
            kl_chunks.append(float(kl))
            del teacher_logits,student_logits,teacher_logp,student_logp,teacher_ce,student_ce,kl
    count = targets.numel()
    teacher_nll,student_nll,kl_sum = map(math.fsum,(teacher_chunks,student_chunks,kl_chunks))
    return {'target_tokens':count,'source_nll':teacher_nll,'source_mean_nll':teacher_nll/count,
        'source_ppl':math.exp(teacher_nll/count),'student_nll':student_nll,'student_mean_nll':student_nll/count,
        'student_ppl':math.exp(student_nll/count),'teacher_to_student_kl_sum':kl_sum,
        'teacher_to_student_kl_mean':kl_sum/count,'teacher_ce_chunk_sums':teacher_chunks,
        'student_ce_chunk_sums':student_chunks,'teacher_kl_chunk_sums':kl_chunks}


def equal_float_bytes(left,right):
    return len(left) == len(right) and all(struct.pack('<d',a) == struct.pack('<d',b) for a,b in zip(left,right))


def aggregate(rows):
    count = sum(row['target_tokens'] for row in rows)
    if not count:
        raise ValueError('Cannot aggregate an empty diagnostic set')
    teacher = math.fsum(row['source_nll'] for row in rows)
    student = math.fsum(row['student_nll'] for row in rows)
    kl = math.fsum(row['teacher_to_student_kl_sum'] for row in rows)
    return {'windows':len(rows),'target_tokens':count,
        'source_nll':teacher,'source_mean_nll':teacher/count,'source_ppl':math.exp(teacher/count),
        'student_nll':student,'student_mean_nll':student/count,'student_ppl':math.exp(student/count),
        'teacher_to_student_kl_sum':kl,'teacher_to_student_kl_mean':kl/count}


def compare_passes(passes,plans):
    summaries,paired = {},{}
    for name in SET_ORDER:
        base,candidate = passes['best_small_e8w5'][name],passes['low_rank_e8w5'][name]
        if len(base) != len(plans[name]) or len(candidate) != len(plans[name]):
            raise ValueError('Incomplete diagnostic window coverage')
        paired[name] = []
        for first,second,plan in zip(base,candidate,plans[name]):
            if any(first.get(key) != value or second.get(key) != value for key,value in plan.items()):
                raise ValueError('Paired diagnostic tokens/counts differ')
            if (not equal_float_bytes(first['teacher_ce_chunk_sums'],second['teacher_ce_chunk_sums'])
                    or not equal_float_bytes([first['source_nll']],[second['source_nll']])):
                raise ValueError('Repeated source CE differs on paired TRAIN window')
            paired[name].append({**plan,'mean_nll':{'source_fp16':first['source_mean_nll'],
                'best_small_e8w5':first['student_mean_nll'],'low_rank_e8w5':second['student_mean_nll']},
                'ppl':{'source_fp16':first['source_ppl'],'best_small_e8w5':first['student_ppl'],
                       'low_rank_e8w5':second['student_ppl']},
                'teacher_to_model_kl_mean':{'source_fp16':0.0,'best_small_e8w5':first['teacher_to_student_kl_mean'],
                    'low_rank_e8w5':second['teacher_to_student_kl_mean']},
                'candidate_minus_baseline_mean_nll':second['student_mean_nll']-first['student_mean_nll'],
                'candidate_minus_baseline_teacher_kl_mean':second['teacher_to_student_kl_mean']-first['teacher_to_student_kl_mean'],
                'source_ce_exact_repeat':True})
        a,b = aggregate(base),aggregate(candidate)
        summaries[name] = {'window_count':len(base),'target_tokens':a['target_tokens'],
            'nll':{'source_fp16':a['source_nll'],'best_small_e8w5':a['student_nll'],'low_rank_e8w5':b['student_nll']},
            'mean_nll':{'source_fp16':a['source_mean_nll'],'best_small_e8w5':a['student_mean_nll'],'low_rank_e8w5':b['student_mean_nll']},
            'ppl':{'source_fp16':a['source_ppl'],'best_small_e8w5':a['student_ppl'],'low_rank_e8w5':b['student_ppl']},
            'teacher_to_model_kl_mean':{'source_fp16':0.0,'best_small_e8w5':a['teacher_to_student_kl_mean'],
                'low_rank_e8w5':b['teacher_to_student_kl_mean']},
            'candidate_vs_baseline_ppl_relative_change':b['student_ppl']/a['student_ppl']-1,
            'candidate_minus_baseline_mean_nll':b['student_mean_nll']-a['student_mean_nll'],
            'candidate_minus_baseline_teacher_kl_mean':b['teacher_to_student_kl_mean']-a['teacher_to_student_kl_mean'],
            'window_counts':{'nll_improved':sum(row['candidate_minus_baseline_mean_nll'] < 0 for row in paired[name]),
                'nll_worsened':sum(row['candidate_minus_baseline_mean_nll'] > 0 for row in paired[name]),
                'kl_improved':sum(row['candidate_minus_baseline_teacher_kl_mean'] < 0 for row in paired[name]),
                'kl_worsened':sum(row['candidate_minus_baseline_teacher_kl_mean'] > 0 for row in paired[name])}}
    fit,fresh = (summaries[name] for name in SET_ORDER)
    gap = {'fresh_minus_fit_mean_nll':{arm:fresh['mean_nll'][arm]-fit['mean_nll'][arm] for arm in ARMS},
        'fresh_minus_fit_teacher_kl_mean':{arm:fresh['teacher_to_model_kl_mean'][arm]-fit['teacher_to_model_kl_mean'][arm] for arm in ARMS},
        'candidate_delta_nll_fresh_minus_fit':fresh['candidate_minus_baseline_mean_nll']-fit['candidate_minus_baseline_mean_nll'],
        'candidate_delta_kl_fresh_minus_fit':fresh['candidate_minus_baseline_teacher_kl_mean']-fit['candidate_minus_baseline_teacher_kl_mean'],
        'interpretation_limit':'Posthoc descriptive differences on different TRAIN content; no causal proof or candidate-selection gate.'}
    return summaries,paired,gap


def model_identity(model):
    return {name:(id(value),value.data_ptr(),value._version) for name,value in model.named_parameters()}


def run(args,report):
    prior,manifest,data,baseline_hashes,candidate_hashes,checked = verify_inputs()
    sets,plans,metadata = prepare_sets(data)
    report.update(binding={'final_evaluation_sha256':FINAL_EVALUATION_SHA,'final_manifest_sha256':FINAL_MANIFEST_SHA,
        'training_binding':manifest['binding'],'training_receipt':manifest['training_receipt'],
        'diagnostic_protocol_sha256':protocol_digest(),'checked_input_sha256':checked},data=metadata,passes={})
    native.write_json(args.report,report)
    if args.prepare_only:
        report.update(complete=True,mode='cpu_data_and_identity_only',cuda_initialized=torch.cuda.is_initialized())
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    report['environment'] = native.environment_receipt()
    source = native.load_source_model(ROOT/'models/source')
    source.eval().requires_grad_(False)
    if source._package_receipt != prior['source_package_receipt']:
        raise ValueError('Source checkpoint/FP16 loader receipt differs')
    report['source_package_receipt'] = source._package_receipt
    report['source_coverage'] = native.audit_parameter_coverage(source)
    source_identity = model_identity(source)
    model = native.load_quantized_model(ROOT/'artifacts/e8w5_v1')
    small = json.loads((ROOT/'artifacts/small_compensation_v1/manifest.json').read_text())
    native.small_overlay.apply_overlay(model,ROOT/'artifacts/small_compensation_v1',small)
    model.eval().requires_grad_(False)
    if source.lm_head.weight.shape[0] != 256000 or model.lm_head.weight.shape[0] != 256000:
        raise ValueError('Diagnostic must score the complete256000-token vocabulary')
    report['baseline507_actual_audit'] = native.audit_frozen_values(model,baseline_hashes)
    report['baseline_coverage'] = native.audit_parameter_coverage(model)
    baseline_identity = model_identity(model)

    def score_pass(arm):
        report['passes'][arm] = {}
        for name in SET_ORDER:
            rows = report['passes'][arm][name] = []
            for index,(window,plan) in enumerate(zip(sets[name],plans[name])):
                row = {**plan,**score_window(source,model,window)}
                if row['target_tokens'] != TARGETS_PER_WINDOW:
                    raise ValueError('TRAIN window target count differs')
                if arm == 'low_rank_e8w5':
                    reference = report['passes']['best_small_e8w5'][name][index]
                    if (not equal_float_bytes(row['teacher_ce_chunk_sums'],reference['teacher_ce_chunk_sums'])
                            or not equal_float_bytes([row['source_nll']],[reference['source_nll']])):
                        report['source_ce_repeat_failure'] = {'set':name,'index':index,
                            'reference_chunk_sums':reference['teacher_ce_chunk_sums'],
                            'actual_chunk_sums':row['teacher_ce_chunk_sums'],
                            'reference_nll':reference['source_nll'],'actual_nll':row['source_nll']}
                        raise ValueError(f'Source CE did not exactly repeat: {name}/{index}')
                    row['source_ce_exact_repeat'] = True
                rows.append(row)
                if (index+1)%8 == 0 or index+1 == len(sets[name]):
                    print(f'[generalization] {arm} {name} {index+1}/{len(sets[name])} '
                          f'CE={row["student_mean_nll"]:.6f} KL={row["teacher_to_student_kl_mean"]:.6f}',flush=True)
                    native.write_json(args.report,report)
            report.setdefault('pass_aggregates',{}).setdefault(arm,{})[name] = aggregate(rows)
            native.write_json(args.report,report)

    score_pass('best_small_e8w5')
    if model_identity(model) != baseline_identity:
        raise ValueError('Baseline scoring changed parameter objects/versions')
    projection_names = {entry['source_key'] for entry in native.expected_inventory().values()}
    unchanged = {name:value for name,value in model.named_parameters() if name not in projection_names}
    if len(unchanged) != 395:
        raise ValueError('Expected395 inherited nonprojection parameters')
    merged = {}
    for label,entry in native.expected_inventory().items():
        path = native.checked_file(ROOT/'artifacts/low_rank_compensation_v1',entry['file'],manifest['files'][entry['file']])
        B,A = native.read_factors(path,expected_shape=entry['shape'],expected_rank=4)
        name = entry['source_key']
        base = model.get_parameter(name)
        if native.tensor_sha_fp16(base) != baseline_hashes[name]:
            raise ValueError(f'Baseline projection changed: {label}')
        value = native.merge_native_projection(base,B,A)
        digest = native.tensor_sha_fp16(value)
        if digest != candidate_hashes[name]:
            raise ValueError(f'Independent native merge differs from already evaluated final model: {label}')
        owner,leaf = name.rsplit('.',1)
        setattr(model.get_submodule(owner),leaf,torch.nn.Parameter(value,requires_grad=False))
        merged[label] = digest
    del value,base,B,A
    gc.collect()
    torch.cuda.empty_cache()
    report['candidate112_merged_fp16_sha256'] = merged
    report['candidate507_actual_audit'] = native.audit_frozen_values(model,candidate_hashes)
    report['candidate_coverage'] = native.audit_parameter_coverage(model)
    if any(model.get_parameter(name) is not value for name,value in unchanged.items()):
        raise ValueError('Candidate replaced a frozen nonprojection object')
    score_pass('low_rank_e8w5')
    report['candidate507_post_diagnostic_audit'] = native.audit_frozen_values(model,candidate_hashes)
    if model_identity(source) != source_identity:
        raise ValueError('Source parameter identity/version changed during diagnostic')
    report['summaries'],report['paired_windows'],report['generalization_gap'] = compare_passes(report['passes'],plans)
    report['source_ce_exact_repeat_all320_windows'] = True
    report['gpu_memory'] = native.gpu_memory_receipt()
    for path,digest in checked.items():
        if native.sha(path) != digest:
            raise ValueError(f'Bound diagnostic input changed: {path}')
    report['complete'] = True


def self_test():
    """One bounded CPU check of selection, CE/KL direction, tail and pairing."""
    if torch.cuda.is_initialized():
        raise ValueError('CPU self-test must precede CUDA initialization')
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.embedding = torch.nn.Embedding(19,4,dtype=torch.float16)
            self.backbone.forward = self.backbone.embedding.forward
            self.lm_head = torch.nn.Linear(4,19,bias=False,dtype=torch.float16)
    torch.manual_seed(928)
    source = Toy().eval().requires_grad_(False)
    student = copy.deepcopy(source)
    with torch.no_grad():
        student.lm_head.weight.add_(torch.linspace(-.2,.1,76).reshape(19,4).half())
    window = torch.arange(8)
    row = score_window(source,student,window,chunk_tokens=3)
    with torch.no_grad():
        truth = window[None,1:].flatten()
        source_logits = source.lm_head(source.backbone(window[None,:-1])).float()
        logits = student.lm_head(student.backbone(window[None,:-1])).float()
        ce = F.cross_entropy(logits.flatten(0,1),truth,reduction='sum')
        kl = F.kl_div(F.log_softmax(logits,-1),F.log_softmax(source_logits,-1),log_target=True,reduction='sum')
    if (row['target_tokens'] != 7 or len(row['teacher_ce_chunk_sums']) != 3
            or not math.isclose(row['student_nll'],float(ce),rel_tol=1e-6,abs_tol=1e-6)
            or not math.isclose(row['teacher_to_student_kl_sum'],float(kl),rel_tol=1e-5,abs_tol=1e-6)):
        raise AssertionError('Chunk/tail CE or KL direction/reduction differs')
    repeat = score_window(source,student,window,chunk_tokens=3)
    if not equal_float_bytes(row['teacher_ce_chunk_sums'],repeat['teacher_ce_chunk_sums']):
        raise AssertionError('Teacher CE repeat check failed')
    changed = list(repeat['teacher_ce_chunk_sums'])
    changed[0] = math.nextafter(changed[0],math.inf)
    if equal_float_bytes(row['teacher_ce_chunk_sums'],changed):
        raise AssertionError('Teacher CE exact comparison missed an altered scalar')
    data_path = ROOT/'training_data/low_rank_compensation_v1/manifest.json'
    if not data_path.exists():
        data_path = ROOT/'reports/low_rank_compensation_v1_data_manifest.json'
    if native.sha(data_path) != native.DATA_SHA:
        raise ValueError('Self-test requires the pinned existing fit-data manifest')
    starts,overlap = select_fresh_starts(json.loads(data_path.read_text()))
    print(json.dumps({'passed':True,'cpu_only':True,'cuda_initialized':torch.cuda.is_initialized(),
        'fresh_windows':len(starts),'fresh_overlap':overlap,'fresh_first_start':starts[0],
        'fresh_last_start':starts[-1],'toy_targets':7,'toy_chunks':3,
        'checks':['full-vocabulary teacher-to-student KL','FP32 CE','short chunk weighting',
                  'exact teacher scalar repeat and mutation rejection','actual pinned583→64 interval selection']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.self_test:
        if args.report is not None or args.prepare_only:
            parser.error('--self-test cannot create a diagnostic report')
        self_test()
        return
    if args.report is None:
        parser.error('--report is required')
    args.report = args.report.resolve()
    if args.report.exists() or args.report.with_suffix(args.report.suffix+'.tmp').exists():
        raise FileExistsError(args.report)
    for relative in ('models','artifacts','training_data','calibration'):
        if ROOT/relative == args.report or ROOT/relative in args.report.parents:
            raise ValueError('Diagnostic report must not be inside immutable input directories')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    report = {'format':'MAMBA2_LOW_RANK_POSTHOC_GENERALIZATION_V1','complete':False,
        'posthoc':True,'candidate_advancement':False,'mode':'posthoc_train_diagnostic',
        'script_sha256':native.sha(__file__),'protocol':DIAGNOSTIC_PROTOCOL,
        'limitations':['This diagnostic follows a failed full-validation result; it is not a new acceptance test.',
            'Different TRAIN subsets can differ in difficulty; gaps are descriptive, not causal proof.',
            'Only one fixed final candidate is inspected; no checkpoint selection, training or promotion.',
            'Source-to-source KL is zero by definition; reported student KL uses the full vocabulary at T=1.',
            'Fresh TRAIN windows become observed diagnostic data after this run.',
            'No validation/test/MK computation; dense FP16 inference does not measure compressed runtime.']}
    native.write_json(args.report,report)
    try:
        run(args,report)
    except BaseException as error:
        report.update(complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        native.write_json(args.report,report)
    print(json.dumps({'complete':report['complete'],'report':str(args.report),
        'summaries':report.get('summaries'),'generalization_gap':report.get('generalization_gap'),
        'report_sha256':native.sha(args.report)}),flush=True)


if __name__ == '__main__':
    main()
