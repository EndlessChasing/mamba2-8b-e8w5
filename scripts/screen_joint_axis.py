#!/usr/bin/env python3
"""Six fixed TRAIN matrices: exact greedy replay, then same-byte joint search."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import tempfile
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5 import codec
from mamba_e8w5.joint_axis import JointAxisResidualCodebook, joint_ldlq_adapter, ROW_BATCH
from mamba_e8w5.input_axis_residual import read_axis_e8, sha
from mamba_e8w5.runtime import MODEL_CONFIG, load_source_state, checkpoint_path

FORMAT = 'MAMBA2_JOINT_AXIS_SCREEN_V1'
PROTOCOL_SHA = '73c175d7a81886cc7102640966d0b2b8ff47734234d586f7c584f222e0a03bcd'
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
CALIBRATION_SHA = '70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'
INPUT_SHA = '0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85'
OUTPUT_SHA = '2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f'
EVALUATION_SHA = '23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4'
LABELS = tuple(f'layer{i}.{part}_proj' for i in (0, 18, 55) for part in ('in', 'out'))
RECIPE = {'residual_bits':4, 'index_bits':20, 'values_per_index':8, 'tune_iters':2,
          'damping':.01, 'scale_override':.9, 'feedback':True, 'buffer_width':128,
          'seed_rule':'1000 + 2*layer + part', 'row_batch':4096, 'shift_order':list(range(16))}


def tensor_sha(tensor):
    h = hashlib.sha256()
    chunks = tensor.reshape(1) if tensor.ndim == 0 else tensor
    for start in range(0, len(chunks), 128):
        h.update(chunks[start:start+128].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(pending, path)


def metadata_identity(path, layout):
    raw = Path(path).read_bytes(); head = layout['header_bytes']
    offset = head+layout['base_uint16_bytes']+layout['axis_nibble_bytes']
    result = {'raw_bytes':len(raw), 'header_bytes':head,
              'header_sha256':hashlib.sha256(raw[:head]).hexdigest(),
              'tail_sha256':hashlib.sha256(raw[offset:]).hexdigest(), 'tail_bytes':len(raw)-offset,
              'amplitude_float64_hex':struct.pack('<d',layout['axis_residual_amplitude']).hex()}
    for key in ('balance_fp16_bytes','input_sign_bytes','output_sign_bytes','scale_fp32_bytes'):
        size=layout[key];result[key+'_sha256']=hashlib.sha256(raw[offset:offset+size]).hexdigest();offset+=size
    assert offset == len(raw)
    return result


def verify_inputs():
    def read(path):return json.loads(path.read_text())
    parent_dir=ROOT/'artifacts/e8w5_v1';small_dir=ROOT/'artifacts/small_compensation_v1'
    input_dir=ROOT/'artifacts/input_axis_residual_v1';output_dir=ROOT/'artifacts/output_axis_residual_v1'
    pins={ROOT/'docs/JOINT_AXIS_SELECTION_PROTOCOL.md':PROTOCOL_SHA,
          parent_dir/'manifest.json':PARENT_SHA,small_dir/'manifest.json':SMALL_SHA,
          input_dir/'manifest.json':INPUT_SHA,output_dir/'manifest.json':OUTPUT_SHA,
          ROOT/'calibration/v1/manifest.json':CALIBRATION_SHA,
          ROOT/'reports/output_axis_residual_v1_eval.json':EVALUATION_SHA,
          checkpoint_path(ROOT/'models/source'):SOURCE_SHA,
          ROOT/'mamba_e8w5/input_axis_residual.py':'baff8c56c875258830ac00076f880105b7933a173f57ba0bf38e27cbf1436c06'}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError(f'Pinned input changed: {path}')
    parent=read(parent_dir/'manifest.json');small=read(small_dir/'manifest.json')
    inputs=read(input_dir/'manifest.json');outputs=read(output_dir/'manifest.json')
    cal=read(ROOT/'calibration/v1/manifest.json');prior=read(ROOT/'reports/output_axis_residual_v1_eval.json')
    if (any(x.get('complete') is not True for x in (parent,small,inputs,outputs,cal,prior))
            or cal['calibration_split']!='train' or cal['evaluation_data_used'] is not False
            or cal['source_checkpoint_sha256']!=SOURCE_SHA or cal['model_config']!=MODEL_CONFIG
            or parent['binding']['hessian_manifest_sha256']!=CALIBRATION_SHA
            or prior['primary_arm']!='axis_in_out_w4_w5' or not prior['full_validation']['performed']
            or prior['integrity']['overlay_manifest_sha256']!=OUTPUT_SHA
            or prior['integrity']['input_manifest_sha256']!=INPUT_SHA):
        raise ValueError('Incomplete/mismatched source/calibration/current model chain')
    for name,digest in parent['binding']['code_sha256'].items():pins[ROOT/'mamba_e8w5'/name]=digest
    for name,digest in parent['binding']['quip_sha256'].items():pins[ROOT/'third_party/quip-sharp'/name]=digest
    tokenizer=parent['binding']['source']['files'][1]
    pins[ROOT/'models/source'/tokenizer['path']]=tokenizer['sha256']
    expected=prior['integrity']['resolved507_fp16_sha256_by_arm']['axis_in_out_w4_w5']
    projection_keys={f'backbone.layers.{i}.mixer.{part}_proj.weight' for i in range(56) for part in ('in','out')}
    inherited395={k:v for k,v in expected.items() if k not in projection_keys}
    if len(expected)!=507 or len(inherited395)!=395:raise ValueError('Current507/395 coverage differs')
    inherited={k:v for k,v in outputs['inherited_files'].items() if not k.endswith('.e8')}
    if set(inherited)!={'embedding.uniform','lm_head.uniform','other_fp16.pt','config.json','e8_codebook.bin'}:
        raise ValueError('Expected five inherited data files')
    for name,row in inherited.items():
        origin=row['origin'];directory=ROOT/'artifacts'/({'parent':'e8w5_v1'}.get(origin,origin))
        path=directory/name
        if path.stat().st_size!=row['bytes']:raise ValueError('Inherited file length differs')
        pins[path]=row['sha256']
    entries={}
    for label in LABELS:
        layer=int(label.split('.')[0][5:]);part=label.split('.')[1]
        manifest,directory=(inputs,input_dir) if part=='in_proj' else (outputs,output_dir)
        entry=manifest['matrices'][label];hp=ROOT/'calibration/v1'/cal['matrices'][label]['file']
        shape=[18560,4096] if part=='in_proj' else [4096,8192];key=f'backbone.layers.{layer}.mixer.{part}.weight'
        if (entry['shape']!=shape or entry['source_key']!=key or entry['source_dtype']!='torch.bfloat16'
                or entry['seed']!=1000+2*layer+(part=='out_proj')
                or entry['hessian_sha256']!=parent['matrices'][label]['hessian_sha256']
                or entry['hessian_sha256']!=cal['matrices'][label]['sha256']
                or cal['matrices'][label]['shape']!=[shape[1],shape[1]]
                or expected[key]!=entry['decoded_fp16_sha256']):raise ValueError(f'Matrix provenance differs: {label}')
        pins[hp]=entry['hessian_sha256'];pins[directory/entry['file']]=entry['sha256']
        entries[label]={'baseline':entry,'baseline_path':directory/entry['file'],'hessian_path':hp}
    for name in ('mamba_e8w5/joint_axis.py','scripts/screen_joint_axis.py'):pins[ROOT/name]=sha(ROOT/name)
    for path,digest in pins.items():
        # The checkpoint was already checked once above; avoid rereading 16GB in one preflight.
        if path==checkpoint_path(ROOT/'models/source'):continue
        if sha(path)!=digest:raise ValueError(f'Bound input changed: {path}')
    source=load_source_state(ROOT/'models/source')
    for label,item in entries.items():
        entry=item['baseline'];original=source[entry['source_key']]
        if original.dtype!=torch.bfloat16 or list(original.shape)!=entry['shape'] or tensor_sha(original)!=entry['source_tensor_sha256']:
            raise ValueError('Actual original BF16 tensor differs')
        disk,layout=read_axis_e8(item['baseline_path'],entry['shape'])
        if layout!=entry['layout']:raise ValueError('Actual baseline layout differs')
        item['metadata_identity']=metadata_identity(item['baseline_path'],layout)
        del disk
    del source
    binding={'protocol_sha256':PROTOCOL_SHA,'parent_manifest_sha256':PARENT_SHA,'small_manifest_sha256':SMALL_SHA,
             'input_manifest_sha256':INPUT_SHA,'output_manifest_sha256':OUTPUT_SHA,
             'current_evaluation_sha256':EVALUATION_SHA,'source_checkpoint_sha256':SOURCE_SHA,
             'calibration_manifest_sha256':CALIBRATION_SHA,'tokenizer_sha256':tokenizer['sha256'],
             'quip_sha256':parent['binding']['quip_sha256'],'builder_source_sha256':sha(__file__),
             'selector_source_sha256':sha(ROOT/'mamba_e8w5/joint_axis.py'),
             'checked_input_sha256':{str(p):v for p,v in pins.items()}}
    return entries,inherited,inherited395,binding


def screen_gate(rows):
    if set(rows)!=set(LABELS):raise ValueError('All six matrices required')
    reductions=[]
    for label in LABELS:
        old,new=rows[label]['greedy'],rows[label]['joint']
        if any(not math.isfinite(x) or x<0 for x in (old,new)):raise ValueError('Invalid original-H error')
        if old==0 and new>0:raise ValueError('Positive joint error against zero baseline')
        reductions.append(0. if old==0 else 1-new/old)
    median=statistics.median(reductions);count=sum(x>=.02 for x in reductions);worst=min(reductions)
    return {'reductions':dict(zip(LABELS,reductions)),'median_reduction':median,
            'improve_at_least_2pct_count':count,'min_reduction':worst,
            'thresholds':{'median':.05,'count_at_least_2pct':4,'minimum':-.001},
            'passed':median>=.05 and count>=4 and worst>=-.001}


def self_test():
    class TinyBase:
        def __init__(self,grid):self.grid=grid
        def quantize(self,x,return_idx=True,**kwargs):
            idx=(x[:,None,:]-self.grid[None,:,:]).square().sum(-1).argmin(-1)
            return (self.grid[idx],idx) if return_idx else self.grid[idx]
    gen=torch.Generator().manual_seed(17)
    grid=torch.randint(-4,5,(7,8),generator=gen).float()/4
    base=TinyBase(grid);x=torch.randint(-12,13,(4101,8),generator=gen).float()/8
    joint=JointAxisResidualCodebook(base,.25);actual,codes=joint.quantize(x)
    # Independent NumPy exhaustive small base x16 codebook oracle.
    all_codes=np.arange(len(grid)*16);points=grid.numpy()[all_codes>>4].copy()
    for j,c in enumerate(all_codes):points[j,(c&15)//2]+=.25*(1-2*(c&1))
    for start in range(0,len(x),100):
        q=x[start:start+100].numpy();oracle=((q[:,None,:]-points[None,:,:])**2).sum(-1).min(-1)
        observed=((q-actual[start:start+100].numpy())**2).sum(-1)
        assert np.array_equal(oracle,observed)
    assert torch.equal(actual,codec.decode_codes(codes,base,.25))
    assert torch.equal(actual,JointAxisResidualCodebook(base,.25).quantize(x,return_idx=False))
    assert torch.equal(actual,torch.cat([JointAxisResidualCodebook(base,.25).quantize(z,False) for z in x.split(4096)]))
    ties=JointAxisResidualCodebook(TinyBase(torch.zeros(1,8)),.25)
    _,tie_codes=ties.quantize(torch.zeros(3,8));assert torch.equal(tie_codes,torch.zeros(3,dtype=torch.int64))
    for bad in (torch.ones(3,7),torch.empty(0,8),torch.ones(2,8,dtype=torch.float16),torch.full((1,8),float('nan'))):
        try:joint.quantize(bad)
        except ValueError:pass
        else:raise AssertionError('Malformed query accepted')
    for amp in (0.,-1.,float('nan'),float('inf')):
        try:JointAxisResidualCodebook(base,amp)
        except ValueError:pass
        else:raise AssertionError('Malformed amplitude accepted')
    cb,_=codec.load_reference_primitives();m,n=16,128;count=m*n//8
    high=torch.tensor([0,1,32768,65535],dtype=torch.int32).repeat(count//4)
    low=torch.arange(count,dtype=torch.int32)%16
    payload={'indices':((high<<4)|low).reshape(m,n//8),'balance':torch.ones(n,dtype=torch.float16),
             'input_sign':torch.ones(n,dtype=torch.int8),'output_sign':torch.ones(m,dtype=torch.int8),
             'scale':torch.tensor(.5),'axis_residual_amplitude':.25}
    direct=cb.grid[high.long()].clone()
    for i in range(count):direct[i,int(low[i])//2]+=.25*(1-2*(int(low[i])%2))
    assert torch.equal(codec.decode_codes(payload['indices'],cb,.25).reshape(-1,8),direct)
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'tiny.e8';codec.write_e8(p,payload,{'shape':[m,n]});disk,layout=read_axis_e8(p,[m,n])
        raw=p.read_bytes();off=layout['header_bytes']
        assert np.array_equal(np.frombuffer(raw[off:off+count*2],dtype='<u2'),high.numpy())
        nib=np.frombuffer(raw[off+count*2:off+count*2+count//2],dtype='u1')
        assert np.array_equal(nib&15,low.numpy()[::2]) and np.array_equal(nib>>4,low.numpy()[1::2])
        assert tensor_sha(codec.decode_e8(disk,cb,'cpu'))==tensor_sha(codec.decode_e8(payload,cb,'cpu'))
        for invalid in (raw[:-1],raw+b'X',b'BADMAGIC'+raw[8:]):
            bad=Path(folder)/'bad.e8';bad.write_bytes(invalid)
            try:read_axis_e8(bad,[m,n])
            except (ValueError,AssertionError):pass
            else:raise AssertionError('Malformed file accepted')
    assert not torch.cuda.is_initialized()
    return {'passed':True,'checks':['small exhaustive NumPy codebook oracle','incumbent ties',
            '4096-row boundary and return modes','malformed inputs/amplitudes',
            'actual E8 uint16 boundaries/all16 nibbles','two-plane bytes/native FP16 readback/malformed length'],
            'grid_part_shape':list(cb.grid_part.shape),'maximum_score_matrix_bytes':4096*16*len(cb.grid_part)*4,
            'score_matrix_scope':'one base.round score tensor; temporary and allocator peaks measured on GPU'}


def run(args,report):
    entries,inherited,inherited395,binding=verify_inputs()
    report.update(binding=binding,inherited_files=inherited,inherited395_fp16_sha256=inherited395,
                  model_config=MODEL_CONFIG,recipe=RECIPE,greedy_replay={},matrices={},files={})
    args.out_dir.mkdir(exist_ok=False)
    for folder in ('greedy','joint'):(args.out_dir/folder).mkdir()
    manifest=args.out_dir/'manifest.json'
    def save():write_json(manifest,report);write_json(args.report,report)
    save();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
                           'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32}
    source=load_source_state(ROOT/'models/source');cb,ldlq=codec.load_reference_primitives();cb=cb.cuda()
    if cb.grid_packed_abs.cpu().numpy().astype('<i4').tobytes()!=(ROOT/'artifacts/e8w5_v1/e8_codebook.bin').read_bytes():
        raise ValueError('Shared base codebook identity differs')
    with torch.inference_mode():
        for stage in ('greedy','joint'):
            if stage=='joint' and len(report['greedy_replay'])!=6:raise ValueError('All greedy replays must pass first')
            report['stage']=stage;save()
            for label in LABELS:
                started=time.monotonic();item=entries[label];baseline=item['baseline'];key=baseline['source_key']
                original=source[key];hp=item['hessian_path']
                if tensor_sha(original)!=baseline['source_tensor_sha256'] or sha(hp)!=baseline['hessian_sha256']:
                    raise ValueError('Source/H changed before matrix')
                h=torch.load(hp,map_location='cuda',weights_only=True);w=original.to('cuda')
                if h.dtype!=torch.float32 or tuple(h.shape)!=(w.shape[1],w.shape[1]) or not torch.isfinite(h).all():
                    raise ValueError('Invalid original TRAIN Hessian')
                torch.cuda.reset_peak_memory_stats();selectors=[]
                algorithm=ldlq if stage=='greedy' else joint_ldlq_adapter(ldlq,selectors)
                restored,info,payload=codec.vector_quantize(w,h,cb,algorithm,baseline['seed'],
                    damping=.01,scale_override=.9,tune_iters=2,use_feedback=True,residual_bits=4)
                path=args.out_dir/stage/(label+'.e8');pending=path.with_suffix('.e8.part')
                with pending.open('xb'):pass
                codec.write_e8(pending,payload,info);disk,layout=read_axis_e8(pending,baseline['shape'])
                if not torch.equal(disk['indices'],payload['indices']):raise ValueError('Stored indices differ')
                decoded=codec.decode_e8(disk,cb,'cuda')
                if not torch.isfinite(decoded).all() or not torch.isfinite(restored).all():raise ValueError('Nonfinite FP16 restoration')
                digest=tensor_sha(decoded)
                if digest!=tensor_sha(restored):raise ValueError('Independent disk FP16 decode differs')
                metadata=metadata_identity(pending,layout)
                if metadata!=item['metadata_identity']:raise ValueError('Stored metadata/raw size changed')
                file_sha=sha(pending)
                if stage=='greedy':
                    original_disk,_=read_axis_e8(item['baseline_path'],baseline['shape'])
                    if (not torch.equal(disk['indices'],original_disk['indices']) or file_sha!=baseline['sha256']
                            or digest!=baseline['decoded_fp16_sha256']):raise ValueError('Exact greedy replay failed')
                    del original_disk
                delta=decoded.float()-w.float()
                j=float(((delta@h)*delta).sum());den=float(((w.float()@h)*w.float()).sum())
                if not math.isfinite(j) or j<0 or not math.isfinite(den) or den<=0:raise ValueError('Invalid final-FP16 original-H error')
                entry={**info,'file':str(path.relative_to(args.out_dir)),'source_key':key,'source_dtype':str(original.dtype),
                       'source_tensor_sha256':baseline['source_tensor_sha256'],'hessian_sha256':baseline['hessian_sha256'],
                       'original_file_sha256':baseline['sha256'],'original_decoded_fp16_sha256':baseline['decoded_fp16_sha256'],
                       'bytes':pending.stat().st_size,'sha256':file_sha,'decoded_fp16_sha256':digest,
                       'original_h_squared_error':j,'original_h_denominator':den,'metadata_identity':metadata,
                       'metadata_equal':True,'raw_length_equal':True,'packed_indices_equal':True,
                       'disk_roundtrip_fp16_equal':True,'layout':layout,'elapsed_seconds':time.monotonic()-started,
                       'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
                if stage=='greedy':entry.update(replay_indices_equal=True,replay_file_equal=True,replay_fp16_equal=True)
                else:
                    old=report['greedy_replay'][label]
                    if den!=old['original_h_denominator']:raise ValueError('Original-H denominator changed')
                    if old['original_h_squared_error']==0 and j>0:raise ValueError('Joint worsened zero baseline')
                    entry['relative_squared_error_reduction']=0. if old['original_h_squared_error']==0 else 1-j/old['original_h_squared_error']
                    if len(selectors)!=1:raise ValueError('Expected one LDLQ callback')
                    entry['selector_stats']=selectors[0]
                os.replace(pending,path)
                report['greedy_replay' if stage=='greedy' else 'matrices'][label]=entry
                report['files'][entry['file']]={k:entry[k] for k in ('bytes','sha256','decoded_fp16_sha256')}
                save();print(json.dumps({'stage':stage,'label':label,'seconds':entry['elapsed_seconds'],'J':j}),flush=True)
                del h,w,restored,payload,disk,decoded,delta;torch.cuda.empty_cache()
    for path,digest in binding['checked_input_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Bound input changed during screen: {path}')
    pairs={label:{'greedy':report['greedy_replay'][label]['original_h_squared_error'],
                  'joint':report['matrices'][label]['original_h_squared_error']} for label in LABELS}
    report['gate']=screen_gate(pairs)
    report['storage']={'greedy_replay_bytes':sum(v['bytes'] for v in report['greedy_replay'].values()),
                       'joint_pilot_bytes':sum(v['bytes'] for v in report['matrices'].values()),
                       'physical_data_bytes':sum(v['bytes'] for v in report['files'].values()),
                       'raw_replacement_delta_bytes':0,'full_candidate_built':False,
                       'current_resolved_raw_data_bytes':3138928792}
    report.update(stage='complete',complete=True);save()
    report['manifest_sha256']=sha(manifest);report['manifest_bytes']=manifest.stat().st_size


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-preflight',action='store_true')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--report',type=Path)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/joint_axis_screen_v1');args=parser.parse_args()
    torch.set_num_threads(8)
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    args.report=(args.report or ROOT/'reports'/('joint_axis_screen_v1_cpu.json' if args.cpu_preflight else 'joint_axis_screen_v1.json')).resolve()
    args.out_dir=args.out_dir.resolve()
    if args.report.parent!=ROOT/'reports' or args.out_dir.parent!=ROOT/'artifacts':raise ValueError('Project report/artifact paths required')
    if args.report.exists() or args.report.with_suffix('.json.tmp').exists() or (not args.cpu_preflight and args.out_dir.exists()):
        raise FileExistsError('Fresh paths required; no overwrite/resume')
    report={'format':FORMAT,'complete':False,'script_sha256':sha(__file__),'labels':list(LABELS),
            'stage':'cpu_preflight' if args.cpu_preflight else 'initializing','ppl_evaluated':False,
            'candidate_advancement':False};started=time.monotonic()
    try:
        if args.cpu_preflight:
            report['self_test']=self_test();entries,inherited,inherited395,binding=verify_inputs()
            report.update(binding=binding,hessian_count=len(entries),inherited_files=inherited,
                          inherited395_fp16_sha256=inherited395,cuda_initialized=torch.cuda.is_initialized())
            assert not report['cuda_initialized'];report['complete']=True
        else:run(args,report)
    except BaseException as error:
        report.update(complete=False,error=repr(error),traceback=traceback.format_exc())
        if not args.cpu_preflight and (args.out_dir/'manifest.json').exists():write_json(args.out_dir/'manifest.json',report)
        raise
    finally:report['elapsed_seconds']=time.monotonic()-started;write_json(args.report,report)


if __name__=='__main__':main()
