#!/usr/bin/env python3
"""Prepare32 fresh TRAIN windows, disjoint from earlier project fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba_e8w5.calibration import load_wikitext_tokens,WIKITEXT_REVISION
from mamba_e8w5.runtime import SentencePieceTokenizer,sha256_file,token_digest

BOUND={
 'calibration/v1/manifest.json':'70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab',
 'training_data/small_compensation_v1/manifest.json':'88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709',
 'reports/projection_crossmoment_pilot_v1.json':'034a161df40fa4dac5be99b25545f9087da6695accc414a852fafe0e47228f87'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=Path('models/source'))
    p.add_argument('--out-dir',type=Path,default=Path('training_data/prototype_compensation_v1'))
    p.add_argument('--protocol',type=Path,default=Path('docs/PROTOTYPE_COMPENSATION_PROTOCOL.md'))
    args=p.parse_args()
    if args.out_dir.exists():raise FileExistsError('Prepared data directory must be new')
    for path,digest in BOUND.items():
        if sha256_file(path)!=digest:raise ValueError(f'Previous fitting provenance differs: {path}')
    cal,small,pilot=[json.loads(Path(path).read_text()) for path in BOUND]
    ids,dataset=load_wikitext_tokens(SentencePieceTokenizer(args.source_dir),'train',WIKITEXT_REVISION)
    if dataset!=small['dataset'] or dataset!=pilot['dataset'] or len(ids)!=2533678:
        raise ValueError('Pinned TRAIN corpus differs')
    excluded={'original_calibration':cal['starts'],'all_small_training':small['starts'],
        'crossmoment_fit':[w['start'] for w in pilot['fit_windows']],
        'crossmoment_heldout':[w['start'] for w in pilot['heldout_windows']]}
    if [len(v) for v in excluded.values()]!=[32,256,32,16]:raise ValueError('Unexpected previous window count')
    intervals=[s for starts in excluded.values() for s in starts]
    grid=list(range(0,len(ids)-2048+1,2048))
    eligible=[s for s in grid if all(s+2048<=old or old+2048<=s for old in intervals)]
    if len(grid)!=1237 or len(eligible)!=871:raise ValueError('Fresh eligible grid differs')
    starts=[eligible[i*870//31] for i in range(32)]
    if len(set(starts))!=32:raise ValueError('Repeated selected interval')
    windows=torch.stack([ids[start:start+2048] for start in starts])
    args.out_dir.mkdir(parents=True)
    path=args.out_dir/'training_tokens.pt'
    torch.save(windows,path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if not torch.equal(windows,restored):raise ValueError('Training window readback differs')
    manifest={'format':'MAMBA2_PROTOTYPE_TRAIN_WINDOWS_V1','complete':True,'split':'train',
        'evaluation_data_used':False,'dataset':dataset,'nwin':32,'seqlen':2048,'starts':starts,
        'selection':'2048-token grid excluding all prior intervals; eligible[floor(i*870/31)],i=0..31',
        'excluded_starts':excluded,'excluded_manifests_sha256':BOUND,
        'overlap':{'grid_blocks':1237,'excluded_grid_blocks':366,'eligible_blocks':871,
            'selected_blocks':32,'internal_overlap_tokens':0,'all_previous_fitting_overlap_tokens':0},
        'unique_stored_window_tokens':65536,'targets_per_pass':65504,'four_pass_target_exposures':262016,
        'token_windows_file':'training_tokens.pt','token_windows_file_bytes':path.stat().st_size,
        'token_windows_file_sha256':sha256_file(path),'selected_tokens_sha256_int64le':token_digest(windows.flatten().numpy()),
        'windows':[{'start':s,'token_sha256_int64le':token_digest(w.numpy())} for s,w in zip(starts,windows)],
        'preparation_source_sha256':sha256_file(__file__),
        'calibration_loader_source_sha256':sha256_file(ROOT/'mamba_e8w5/calibration.py'),
        'runtime_source_sha256':sha256_file(ROOT/'mamba_e8w5/runtime.py'),
        'protocol_sha256':sha256_file(args.protocol)}
    mp=args.out_dir/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'manifest_sha256':sha256_file(mp),'starts':starts,
        'tokens_sha256':manifest['token_windows_file_sha256'],'overlap':manifest['overlap']}),flush=True)


if __name__=='__main__':main()
