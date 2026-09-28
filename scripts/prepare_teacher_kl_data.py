#!/usr/bin/env python3
"""CPU-only fixed448-train/64-heldout preparation for teacher-only distillation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from mamba_e8w5.calibration import load_wikitext_tokens, WIKITEXT_REVISION
from mamba_e8w5.runtime import (SentencePieceTokenizer, SOURCE_CHECKPOINT_SHA256,
    TOKENIZER_SHA256, sha256_file, token_digest)

SEQLEN = 2048
TRAIN_WINDOWS = 448
HELDOUT_WINDOWS = 64
TRAIN_TOKENS = 2533678
SEED = 20260928
FORMAT = 'MAMBA2_TEACHER_KL_TRAIN_WINDOWS_V1'
PROTOCOL_SHA = 'c80fe02fe1dafb24b7a983bf347205c709a7af297cc325ed8761a29ecbff5c6a'
BOUND = {
    'calibration/v1/manifest.json':'70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab',
    'training_data/small_compensation_v1/manifest.json':'88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709',
    'reports/projection_crossmoment_pilot_v1.json':'034a161df40fa4dac5be99b25545f9087da6695accc414a852fafe0e47228f87',
    'training_data/prototype_compensation_v1/manifest.json':'a485424d1b7894eb1d5a6361ab4c25fdb4f6e5bedc56d7a15a7e4add2c454bcd',
    'training_data/low_rank_compensation_v1/manifest.json':'e168e1d90ea7cd9c14c8210f13b5e6b569612c82dcaf832100c250cb72d24bef',
    'reports/low_rank_generalization_v1.json':'2f75a73ad675f563b035f02ee3fab3522f011785eff3459b5c0caf098347da25',
}
CODE_BOUND = {
    'mamba_e8w5/runtime.py':'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/calibration.py':'b5a7356a2ab769538940aed218653a092a7055aa26a9ec6f62d3783b9ac78ee5',
}
EXPECTED_COUNTS = {'original_calibration':32,'all_small_training':256,'crossmoment_fit':32,
    'crossmoment_heldout':16,'prototype_training':32,'low_rank_training':256,'generalization_fresh':64}
UNUSED_STARTS = [362496,718848,1083392,1439744,1802240,2158592,2510848]


def dataset_of(document):
    return document['dataset'] if 'dataset' in document else document['data']['dataset']


def previous_intervals(documents):
    cal,small,pilot,prototype,low_rank,diagnostic = documents
    if any(document.get('complete') is not True for document in documents):
        raise ValueError('Prior interval provenance is incomplete')
    if (diagnostic.get('posthoc') is not True or diagnostic.get('candidate_advancement') is not False
            or [row['start'] for row in diagnostic['data']['sets']['fit_train256']['windows']] != low_rank['starts']):
        raise ValueError('Posthoc diagnostic fitting identity differs')
    excluded = {'original_calibration':cal['starts'],'all_small_training':small['starts'],
        'crossmoment_fit':[row['start'] for row in pilot['fit_windows']],
        'crossmoment_heldout':[row['start'] for row in pilot['heldout_windows']],
        'prototype_training':prototype['starts'],'low_rank_training':low_rank['starts'],
        'generalization_fresh':[row['start'] for row in diagnostic['data']['sets']['fresh_train64']['windows']]}
    for name,starts in excluded.items():
        if (not isinstance(starts,list) or len(starts) != EXPECTED_COUNTS[name]
                or len(set(starts)) != len(starts)
                or any(type(s) is not int or s < 0 or s+SEQLEN > TRAIN_TOKENS for s in starts)):
            raise ValueError(f'Invalid prior intervals: {name}')
    return excluded


def _eligible(excluded):
    if {name:len(starts) for name,starts in excluded.items()} != EXPECTED_COUNTS:
        raise ValueError('Prior interval family counts differ')
    intervals = [s for starts in excluded.values() for s in starts]
    grid = list(range(0,TRAIN_TOKENS-SEQLEN+1,SEQLEN))
    eligible = [s for s in grid if all(s+SEQLEN <= old or old+SEQLEN <= s for old in intervals)]
    if len(grid) != 1237 or len(eligible) != 519:
        raise ValueError('Expected1237 grid blocks and519 previously unused blocks')
    return eligible


def select_starts(excluded):
    """Reserve first, then train; fixed spread indices, no model outputs read."""
    eligible = _eligible(excluded)
    heldout = [eligible[i*518//63] for i in range(64)]
    heldout_set = set(heldout)
    remaining = [s for s in eligible if s not in heldout_set]
    if len(remaining) != 455:
        raise ValueError('Expected455 blocks after reservation')
    training = [remaining[i*454//447] for i in range(448)]
    training_set = set(training)
    unused = [s for s in remaining if s not in training_set]
    if (len(heldout_set) != 64 or len(training_set) != 448 or training_set & heldout_set
            or heldout[0] != 14336 or heldout[-1] != 2516992
            or min(b-a for a,b in zip(heldout,heldout[1:])) != 30720
            or training[0] != 18432 or training[-1] != 2514944
            or min(b-a for a,b in zip(training,training[1:])) != 2048
            or unused != UNUSED_STARTS):
        raise ValueError('Fixed train/heldout/unused selection differs')
    previous = {family:sum(max(0,min(s+SEQLEN,old+SEQLEN)-max(s,old))
        for s in training+heldout for old in starts) for family,starts in excluded.items()}
    cross = sum(max(0,min(s+SEQLEN,other+SEQLEN)-max(s,other)) for s in training for other in heldout)
    internal = {name:sum(max(0,a+SEQLEN-b) for a,b in zip(starts,starts[1:]))
        for name,starts in (('training',training),('heldout',heldout))}
    if any(previous.values()) or cross or any(internal.values()):
        raise ValueError('Selected stored token intervals overlap')
    return training,heldout,unused,{'grid_blocks':1237,'excluded_grid_blocks':718,'eligible_blocks':519,
        'reserved_blocks':64,'remaining_after_reservation':455,'training_blocks':448,'unused_blocks':7,
        'training_heldout_overlap_tokens':cross,'internal_overlap_tokens':internal,
        'previous_family_overlap_tokens':previous,'all_previous_overlap_tokens':sum(previous.values()),
        'minimum_training_start_gap':2048,'minimum_heldout_start_gap':30720}


def train_schedule():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    return torch.randperm(TRAIN_WINDOWS,generator=generator,device='cpu').tolist()


def verify_pins(protocol=None):
    protocol = ROOT/'docs/TEACHER_KL_COMPENSATION_PROTOCOL.md' if protocol is None else Path(protocol)
    if not isinstance(PROTOCOL_SHA,str) or len(PROTOCOL_SHA) != 64 or sha256_file(protocol) != PROTOCOL_SHA:
        raise ValueError('Teacher-KL protocol has not been frozen or differs')
    for name,digest in {**BOUND,**CODE_BOUND}.items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f'Pinned provenance differs: {name}')
    return [json.loads((ROOT/name).read_text()) for name in BOUND]


def _tensor_file(directory,name,count,digest,size):
    path = Path(directory)/name
    if (path.is_symlink() or not path.is_file() or type(size) is not int
            or path.stat().st_size != size or sha256_file(path) != digest):
        raise ValueError(f'Token file identity differs: {name}')
    value = torch.load(path,map_location='cpu',weights_only=True)
    if (not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.int64
            or tuple(value.shape) != (count,SEQLEN) or value.min() < 0 or value.max() >= 256000):
        raise ValueError(f'Token file geometry/content differs: {name}')
    return value


def verify_data(directory,expected_manifest_sha256=None):
    """Return (manifest,train,heldout), CPU only; heldout is never a fit schedule."""
    directory = Path(directory)
    path = directory/'manifest.json'
    if expected_manifest_sha256 is not None and sha256_file(path) != expected_manifest_sha256:
        raise ValueError('Prepared teacher-KL manifest differs')
    data = json.loads(path.read_text())
    documents = verify_pins()
    excluded = previous_intervals(documents)
    training,heldout,unused,overlap = select_starts(excluded)
    if (data.get('format') != FORMAT or data.get('complete') is not True or data.get('split') != 'train'
            or data.get('evaluation_data_used') is not False or data.get('heldout_used_for_fitting') is not False
            or data.get('protocol_sha256') != PROTOCOL_SHA or data.get('excluded_manifests_sha256') != BOUND
            or data.get('preparation_source_sha256') != sha256_file(__file__)
            or data.get('calibration_loader_source_sha256') != CODE_BOUND['mamba_e8w5/calibration.py']
            or data.get('runtime_source_sha256') != CODE_BOUND['mamba_e8w5/runtime.py']
            or data.get('source_checkpoint_sha256') != SOURCE_CHECKPOINT_SHA256 or data.get('tokenizer_sha256') != TOKENIZER_SHA256
            or data.get('excluded_starts') != excluded or data.get('eligible_starts') != _eligible(excluded)
            or data.get('training_starts') != training or data.get('heldout_starts') != heldout
            or data.get('unused_starts') != unused or data.get('overlap') != overlap
            or data.get('schedule_seed') != SEED or data.get('schedule') != train_schedule()
            or any(type(index) is not int for index in data.get('schedule',[]))
            or any(data.get('dataset') != dataset_of(document) for document in documents)):
        raise ValueError('Teacher-KL data declaration/provenance differs')
    if {p.name for p in directory.iterdir()} != {'manifest.json','training_tokens.pt','heldout_tokens.pt'}:
        raise ValueError('Unexpected prepared-data file inventory')
    tensors = {}
    for name,starts,count in (('training',training,448),('heldout',heldout,64)):
        if data.get(name+'_tokens_file') != name+'_tokens.pt':
            raise ValueError('Unexpected token file name')
        tokens = _tensor_file(directory,name+'_tokens.pt',count,data[name+'_tokens_file_sha256'],data[name+'_tokens_file_bytes'])
        if (token_digest(tokens.flatten().numpy()) != data[name+'_tokens_sha256_int64le']
                or len(data[name+'_windows']) != count):
            raise ValueError('Selected token content/coverage differs')
        for start,window,row in zip(starts,tokens,data[name+'_windows']):
            if row != {'start':start,'stored_tokens':2048,'targets':2047,'token_sha256_int64le':token_digest(window.numpy())}:
                raise ValueError('Per-window token identity differs')
        tensors[name] = tokens
    if (data.get('training_target_exposures') != 917056 or data.get('heldout_targets') != 131008
            or data.get('training_window_count') != 448 or data.get('heldout_window_count') != 64
            or data.get('seqlen') != 2048):
        raise ValueError('Data exposure/count accounting differs')
    return data,tensors['training'],tensors['heldout']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'models/source')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'training_data/teacher_kl_compensation_v1')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/TEACHER_KL_COMPENSATION_PROTOCOL.md')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError("Use CUDA_VISIBLE_DEVICES='' for CPU-only data preparation")
    torch.set_num_threads(2)
    if args.out_dir.exists():
        raise FileExistsError('Prepared data must use a fresh directory')
    documents = verify_pins(args.protocol)
    excluded = previous_intervals(documents)
    training,heldout,unused,overlap = select_starts(excluded)
    source_sha = sha256_file(__file__)
    tokenizer = SentencePieceTokenizer(args.source_dir)
    ids,dataset = load_wikitext_tokens(tokenizer,'train',WIKITEXT_REVISION)
    if (ids.dtype != torch.int64 or ids.device.type != 'cpu' or tuple(ids.shape) != (TRAIN_TOKENS,)
            or any(dataset != dataset_of(document) for document in documents)
            or dataset['tokenizer_sha256'] != TOKENIZER_SHA256 or dataset['split'] != 'train'
            or documents[0]['source_checkpoint_sha256'] != SOURCE_CHECKPOINT_SHA256):
        raise ValueError('Original TRAIN token stream/source identity differs')
    data = {'format':FORMAT,'complete':True,'split':'train','evaluation_data_used':False,
        'heldout_used_for_fitting':False,'selection_uses_model_outputs':False,'dataset':dataset,
        'source_checkpoint_sha256':SOURCE_CHECKPOINT_SHA256,'tokenizer_sha256':TOKENIZER_SHA256,
        'seqlen':2048,'training_window_count':448,'heldout_window_count':64,
        'excluded_manifests_sha256':BOUND,'excluded_starts':excluded,'eligible_starts':_eligible(excluded),
        'training_starts':training,'heldout_starts':heldout,'unused_starts':unused,'overlap':overlap,
        'selection':'519 eligible; heldout=eligible[floor(i*518/63)],i0..63; remove64; training=remaining[floor(i*454/447)],i0..447;7unused',
        'schedule_seed':SEED,'schedule':train_schedule(),
        'schedule_generator':'one CPU torch.randperm(448), separate from factor initialization',
        'training_stored_tokens':917504,'training_target_exposures':917056,
        'heldout_stored_tokens':131072,'heldout_targets':131008,
        'preparation_source_sha256':source_sha,'protocol_sha256':PROTOCOL_SHA,
        'calibration_loader_source_sha256':CODE_BOUND['mamba_e8w5/calibration.py'],
        'runtime_source_sha256':CODE_BOUND['mamba_e8w5/runtime.py'],
        'environment':{'python':sys.version,'torch':torch.__version__,'CUDA_VISIBLE_DEVICES':'','cuda_initialized':False}}
    args.out_dir.mkdir(parents=True,exist_ok=False)
    for name,starts in (('training',training),('heldout',heldout)):
        tokens = torch.stack([ids[start:start+SEQLEN] for start in starts])
        path = args.out_dir/(name+'_tokens.pt')
        with path.open('xb') as stream:
            torch.save(tokens,stream)
            stream.flush()
            os.fsync(stream.fileno())
        restored = torch.load(path,map_location='cpu',weights_only=True)
        if restored.dtype != torch.int64 or restored.shape != tokens.shape or not torch.equal(tokens,restored):
            raise ValueError('Token file readback differs')
        data.update({name+'_tokens_file':path.name,name+'_tokens_file_bytes':path.stat().st_size,
            name+'_tokens_file_sha256':sha256_file(path),name+'_tokens_sha256_int64le':token_digest(tokens.flatten().numpy()),
            name+'_windows':[{'start':start,'stored_tokens':2048,'targets':2047,
                'token_sha256_int64le':token_digest(window.numpy())} for start,window in zip(starts,tokens)]})
    if torch.cuda.is_initialized():
        raise ValueError('CPU data preparation initialized CUDA')
    verify_pins(args.protocol)
    if sha256_file(__file__) != source_sha:
        raise ValueError('Data preparer changed during execution')
    data['disk_roundtrip_equal'] = True
    partial = args.out_dir/'manifest.json.partial'
    with partial.open('x') as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    manifest_path = args.out_dir/'manifest.json'
    os.link(partial,manifest_path)
    partial.unlink()
    verify_data(args.out_dir,sha256_file(manifest_path))
    print(json.dumps({'complete':True,'manifest_sha256':sha256_file(manifest_path),
        'training_tokens_file_sha256':data['training_tokens_file_sha256'],
        'heldout_tokens_file_sha256':data['heldout_tokens_file_sha256'],
        'training_tokens_sha256_int64le':data['training_tokens_sha256_int64le'],
        'heldout_tokens_sha256_int64le':data['heldout_tokens_sha256_int64le'],
        'overlap':overlap,'training_target_exposures':917056,'heldout_targets':131008,
        'cuda_initialized':False}),flush=True)


if __name__ == '__main__':
    main()
