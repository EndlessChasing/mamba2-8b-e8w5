#!/usr/bin/env python3
"""Prepare train-only blocks without old calibration overlap; no GPU or Hessians."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba_e8w5.calibration import load_wikitext_tokens,WIKITEXT_REVISION
from mamba_e8w5.runtime import SentencePieceTokenizer,token_digest,sha256_file
from mamba_e8w5.small_training import eligible_window_starts

CALIBRATION_SHA='70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT/'models/source')
    p.add_argument('--calibration-manifest',type=Path,default=ROOT/'calibration/v1/manifest.json')
    p.add_argument('--out-dir',type=Path,default=ROOT/'training_data/small_compensation_v1')
    args=p.parse_args()
    if args.out_dir.exists(): raise FileExistsError('Preserving existing data directory')
    if sha256_file(args.calibration_manifest)!=CALIBRATION_SHA: raise ValueError('Calibration exclusion identity differs')
    previous=json.loads(args.calibration_manifest.read_text())
    tokenizer=SentencePieceTokenizer(args.source_dir)
    ids,dataset=load_wikitext_tokens(tokenizer,'train',WIKITEXT_REVISION)
    if dataset!=previous['dataset']: raise ValueError('Pinned train token stream differs from original calibration source')
    starts,overlap=eligible_window_starts(len(ids),previous['starts'])
    if len(ids)!=2533678 or {key:overlap[key] for key in ('grid_blocks','excluded_grid_blocks','eligible_blocks','selected_blocks')} != {
            'grid_blocks':1237,'excluded_grid_blocks':62,'eligible_blocks':1175,'selected_blocks':256}:
        raise ValueError('Declared grid/exclusion counts differ')
    windows=torch.stack([ids[start:start+2048] for start in starts])
    args.out_dir.mkdir(parents=True)
    path=args.out_dir/'training_tokens.pt'
    torch.save(windows,path)
    report={'format':'MAMBA2_SMALL_TRAIN_WINDOWS_V1','complete':True,
        'dataset':dataset,'split':'train','evaluation_data_used':False,'nwin':256,'seqlen':2048,
        'selection':'2048-token grid; exclude any overlap with original32 calibration intervals; eligible[floor(i*(N-1)/255)]',
        'starts':starts,'original_calibration_manifest_sha256':CALIBRATION_SHA,
        'excluded_original_calibration_starts':previous['starts'],'overlap':overlap,
        'selected_tokens_sha256_int64le':token_digest(windows.flatten().numpy()),
        'token_windows_file':path.name,'token_windows_file_sha256':sha256_file(path),'token_windows_file_bytes':path.stat().st_size,
        'unique_stored_window_tokens':windows.numel(),'unique_input_positions_per_pass':256*2047,
        'targets_per_pass':256*2047,'four_pass_target_exposures':256*2047*4,
        'preparation_source_sha256':sha256_file(__file__),
        'selection_source_sha256':sha256_file(ROOT/'mamba_e8w5/small_training.py'),
        'calibration_loader_source_sha256':sha256_file(ROOT/'mamba_e8w5/calibration.py'),
        'runtime_source_sha256':sha256_file(ROOT/'mamba_e8w5/runtime.py')}
    (args.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'manifest_sha256':sha256_file(args.out_dir/'manifest.json'),
                      'overlap':overlap,'token_windows_file_sha256':report['token_windows_file_sha256']}),flush=True)


if __name__=='__main__': main()
