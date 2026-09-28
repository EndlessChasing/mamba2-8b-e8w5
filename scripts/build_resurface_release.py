#!/usr/bin/env python3
"""Compose, package and restore the measured v0.2.0 soft-Resurface candidate.

CPU only. Existing weight files are hardlinked read-only by convention, never
rewritten. E8HUF001 raw members preserve every byte without a new axis coder.
No original checkpoint, training data, GPU evaluation or publication is needed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import package_release as release
from mamba_e8w5.huffman import Reader, sha256_file as sha, verify_container
from mamba_e8w5.runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, TOKENIZER_SHA256, TOKENIZER_FILENAME

FORMAT = 'MAMBA2_AXIS_RESURFACE_RESOLVED_V1'
TAG = 'v0.2.0-resurface'
BUFFER = 8 << 20
PINS = {
    'reports/axis_small_readaptation_v1_eval.json': '3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4',
    'reports/axis_small_readaptation_v1_manifest.json': '3ede80b2883d8b8987cb3e2198fd8badc2dbbb5af03c8fe6cac7f5f9f5b62047',
    'reports/resurface_readapted_v1_manifest.json': '5660676adb45806052cbaab7f5e50056769c4f5e106d5a49cf148f025bf2b7a8',
    'reports/input_axis_residual_v1_manifest.json': '0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85',
    'reports/output_axis_residual_v1_manifest.json': '2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f',
    'reports/resurface_soft_continuation_v1_full_eval.json': '06a71c11fc0a12a52add6e7bf5d28b8a8eb9f832b2ec1cb846d9acaf30afe961',
    'reports/resurface_soft_continuation_v1_confirm_eval.json': '306ae9e8ed5e78756f7c8ea39c8db40dbebf3895ced3c5c279722be5100a344b',
    'reports/resurface_soft_continuation_v1_full_independent_audit.json': '14a9bb85a95dcdd6429e12409c73336765e546a9ae09b43b2c0bd28bdcc2d515',
    'reports/resurface_soft_continuation_v1_confirm_independent_audit.json': '37e3c9e0e5498335fcbf0a30fbc0cfaee34418c80cf5ea0a1cfd99adad6b56da',
}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes(encoded(value)); os.replace(temporary, path)


def check_file(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError('File identity differs: '+str(path))


def expected_names():
    return {f'layer{i}.{part}.e8' for i in range(56) for part in ('in_proj', 'out_proj')} | {
        'embedding.uniform', 'lm_head.uniform', 'other_fp16.pt', 'config.json', 'e8_codebook.bin',
        'adapter_fp16.pt', TOKENIZER_FILENAME}


def audit_raw(directory):
    raw = release.validate_raw(directory)
    if raw.get('format') != FORMAT or set(raw['files']) != expected_names():
        raise ValueError('Not the self-contained119-file axis/soft-Resurface package')
    if raw.get('inference_entrypoint') != 'mamba_e8w5.release_runtime.load_model' or raw['adapter']['gate_mode'] != 'soft':
        raise ValueError('Active soft-adapter runtime is required')
    if sum(e['bytes'] for n,e in raw['files'].items() if n != TOKENIZER_FILENAME) != 3141468439:
        raise ValueError('Resolved actual model byte count differs')
    return raw


def compose(workspace, target):
    """Fresh raw manifest; no inherited metrics or stale original-small state."""
    import torch
    from mamba_e8w5.resurface_native import read_fp16, tensor_hash
    from mamba_e8w5.release_runtime import REQUIRED_CODE, validate_manifest
    if target.exists(): raise FileExistsError(target)
    reports = {}
    for name, digest in PINS.items():
        path = ROOT/name
        if sha(path) != digest: raise ValueError('Pinned evidence differs: '+name)
        reports[name] = json.loads(path.read_text())
    small = reports['reports/axis_small_readaptation_v1_manifest.json']
    overlay = reports['reports/resurface_readapted_v1_manifest.json']
    full = reports['reports/resurface_soft_continuation_v1_full_eval.json']
    confirm = reports['reports/resurface_soft_continuation_v1_confirm_eval.json']
    source_score = reports['reports/axis_small_readaptation_v1_eval.json']['full_validation']['arms']['source_fp16']['summary']
    for record, stage in ((full,'full'), (confirm,'confirm')):
        if not record['complete'] or record['stage'] != stage or not record['gate']['passed'] or record['gate_mode'] != 'soft':
            raise ValueError('Measured soft candidate did not complete both quality gates')
    if full['candidate_identity'] != confirm['candidate_identity']:
        raise ValueError('PPL and MK did not measure the same adapter')
    if full['candidate_identity']['overlay_manifest_sha256'] != PINS['reports/resurface_readapted_v1_manifest.json']:
        raise ValueError('Evaluated adapter manifest identity differs')
    expected = {**small['frozen_large_hash_ledger'], **small['exported_small_fp16_sha256']}
    if (len(expected) != 507 or overlay['base507_sha256'] != expected or
            full['initial_current507_audit']['unchanged_content_sha256'] != expected or
            confirm['initial_current507_audit']['unchanged_content_sha256'] != expected):
        raise ValueError('Actual evaluated507 tensor identities differ')
    paths, files = {}, {}
    for name, entry in small['inherited_files'].items():
        # Resolve only the recorded relative artifact path under this workspace.
        saved = Path(entry['path']); parts = saved.parts
        relative = Path(*parts[parts.index('artifacts'):])
        paths[name] = workspace/relative
        files[name] = {k:entry[k] for k in ('bytes','sha256')}
    for name, folder, ledger in (
            ('other_fp16.pt','axis_small_readaptation_v1',small['files']),
            ('adapter_fp16.pt','resurface_readapted_v1',overlay['files'])):
        paths[name] = workspace/'artifacts'/folder/name
        files[name] = {k:ledger[name][k] for k in ('bytes','sha256')}
        manifest = workspace/'artifacts'/folder/'manifest.json'
        expected_sha = PINS['reports/'+folder+'_manifest.json']
        if sha(manifest) != expected_sha: raise ValueError('Actual overlay manifest differs: '+folder)
    paths[TOKENIZER_FILENAME] = workspace/'models/source'/TOKENIZER_FILENAME
    files[TOKENIZER_FILENAME] = {'bytes':paths[TOKENIZER_FILENAME].stat().st_size,'sha256':TOKENIZER_SHA256}
    if set(paths) != expected_names(): raise ValueError('Resolved file coverage differs')
    for name,path in paths.items(): check_file(path,files[name])
    values = torch.load(paths['other_fp16.pt'],map_location='cpu',weights_only=True)
    if set(values) != set(small['exported_small_fp16_sha256']) or len(values) != 393:
        raise ValueError('Actual393 small tensor inventory differs')
    for name,value in values.items():
        if value.dtype != torch.float16 or not torch.isfinite(value).all() or tensor_hash(value) != expected[name]:
            raise ValueError('Actual small tensor differs: '+name)
    payload = read_fp16(paths['adapter_fp16.pt'], overlay['binding'])
    adapter_hashes = {name:tensor_hash(t) for name,t in payload['tensors'].items()}
    if (payload['gate_mode'] != 'soft' or len(adapter_hashes) != 224 or
            adapter_hashes != overlay['fp16_tensor_sha256'] or adapter_hashes != full['adapter_before']['tensor_sha256'] or
            adapter_hashes != confirm['adapter_before']['tensor_sha256']):
        raise ValueError('Actual224 adapter tensors differ from measured candidate')
    matrices = {}
    for part,report_name in (('in_proj','input_axis_residual_v1_manifest'),('out_proj','output_axis_residual_v1_manifest')):
        for label, row in reports['reports/'+report_name+'.json']['matrices'].items():
            name = row['file']; key = row['source_key']
            if files[name] != {k:row[k] for k in ('bytes','sha256')} or expected[key] != row['decoded_fp16_sha256']:
                raise ValueError('Projection identity differs: '+label)
            matrices[label] = {k:row[k] for k in ('file','source_key','shape','index_bits','values_per_index',
                'axis_residual_amplitude','decoded_fp16_sha256')}
    vocabularies = {label:{'file':file,'source_key':key,'shape':[256000,4096],
        'bits':bits,'decoded_fp16_sha256':expected[key]} for label,file,key,bits in (
        ('embedding','embedding.uniform','backbone.embedding.weight',4),('lm_head','lm_head.uniform','lm_head.weight',5))}
    code = {name:sha(ROOT/name) for name in sorted(REQUIRED_CODE)}
    # Previously evaluated core decoder source must remain unchanged.
    old_code = full['checked_input_sha256']
    for name in ('mamba_e8w5/runtime.py','mamba_e8w5/codec.py','mamba_e8w5/resurface_native.py'):
        matches = {digest for path,digest in old_code.items() if path.endswith('/'+name)}
        if matches != {code[name]}: raise ValueError('Frozen decoder source differs: '+name)
    quality = {'full_ppl':full['ppl']['active_resurface']['summary'],
        'paired_base_ppl':full['ppl']['current_readapted']['summary'],
        'historical_source_full_ppl':source_score,
        'source_relative_ppl_change_percent':100*(full['ppl']['active_resurface']['summary']['ppl']/source_score['ppl']-1),
        'source_comparison_scope':'Aligned historical source evaluation; source was not rerun in the soft-adapter full-PPL process.',
        'confirmation':{arm:confirm['arms'][arm]['summary'] for arm in ('current_readapted','active_resurface')},
        'confirmation_comparison':confirm['mk_comparison'],'confirmation_gate':confirm['gate'],
        'same_enabled_soft_adapter':True,'validation_informed_development':True,
        'original_strict_protocol_pass_claimed':False,
        'post_observation_prerequisite_amendment':full['post_observation_prerequisite_amendment'],
        'reports_sha256':{name:digest for name,digest in PINS.items() if 'eval' in name or 'audit' in name}}
    raw = {'format':FORMAT,'complete':True,'release_tag':TAG,'model_config':MODEL_CONFIG,
        'source_checkpoint_sha256':SOURCE_CHECKPOINT_SHA256,'tokenizer_sha256':TOKENIZER_SHA256,
        'total_parameter_count':8236999680,'base_parameter_tensors':507,
        'inference_entrypoint':'mamba_e8w5.release_runtime.load_model',
        'files':files,'matrices':matrices,'vocabularies':vocabularies,'base507_fp16_sha256':expected,
        'adapter':{'file':'adapter_fp16.pt','gate_mode':'soft','variant':payload['variant'],
            'geometry':payload['geometry'],'binding':payload['binding'],'tensor_sha256':adapter_hashes,
            'parameters':1154104,'payload_bytes':2308208,**files['adapter_fp16.pt']},
        'binding':{'code_sha256':code,'evidence_sha256':PINS,'builder_source_sha256':sha(__file__),
            'candidate_identity':full['candidate_identity']},'quality':quality,
        'storage':{'base_model_data_bytes':3138928792,'adapter_file_bytes':2539647,
            'model_plus_adapter_data_bytes':3141468439,'tokenizer_bytes':files[TOKENIZER_FILENAME]['bytes'],
            'all_payload_bytes':sum(e['bytes'] for e in files.values()),
            'scope':'119 exact files; includes tokenizer, excludes this manifest and outer distribution assets.'}}
    validate_manifest(raw)
    target.mkdir(parents=True)
    for name,path in paths.items():
        os.link(path,target/name)  # No chmod/write to any linked source inode.
    write(target/'manifest.json',raw)
    audit_raw(target)
    for name,path in paths.items(): check_file(path,files[name])
    return raw, {'complete':True,'files_verified':119,'small_tensors_verified':393,'adapter_tensors_verified':224,
        'raw_manifest_sha256':sha(target/'manifest.json'),'hardlinks_created':119,
        'source_files_unchanged':True,'dense_projection_decode_repeated':False,
        'decoded_identities_bound_to_completed_quality':True}


def pack_raw_members(source, target):
    """Explicit raw-member E8HUF001 writer, consumed by the unchanged Reader."""
    if target.exists() or target.with_suffix(target.suffix+'.partial').exists(): raise FileExistsError(target)
    paths = sorted(source.iterdir())
    if any(not p.is_file() or p.is_symlink() for p in paths): raise ValueError('Flat regular files required')
    report = {'format':'E8HUF001','files':[],'codebooks':{},'entropy':{},
        'storage_mode':'all-raw-members-no-entropy-coding', 'source_manifest_sha256':sha(source/'manifest.json')}
    tmp = target.with_suffix(target.suffix+'.partial')
    with tmp.open('xb') as out:
        out.write(bytes(24))
        for path in paths:
            digest=hashlib.sha256(); count=0; start=out.tell()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(BUFFER),b''):
                    out.write(chunk);digest.update(chunk);count+=len(chunk)
            if count != path.stat().st_size or digest.hexdigest() != sha(path):
                raise ValueError('Source changed during raw packing: '+path.name)
            report['files'].append({'name':path.name,'original_bytes':count,'sha256':digest.hexdigest(),
                'family':None,'prefix_offset':start,'prefix_bytes':count})
        at=out.tell(); metadata=json.dumps(report,separators=(',',':'),allow_nan=False).encode()
        out.write(metadata);out.seek(0);out.write(struct.pack('<8sQQ',b'E8HUF001',at,len(metadata)))
        out.flush();os.fsync(out.fileno())
    os.replace(tmp,target)
    return report


def source_archive(target):
    """Exact corresponding sources plus a declared, bounded evidence subset."""
    provenance_path=ROOT/'SNAPSHOT_PROVENANCE.json'
    if not provenance_path.is_file() or (ROOT/'.git').exists():
        raise ValueError('Build from an immutable exported source snapshot with provenance')
    provenance=json.loads(provenance_path.read_text())
    for name,length in (('exported_git_commit',40),('exported_git_tree',40),('exported_archive_sha256',64)):
        value=provenance.get(name)
        if not isinstance(value,str) or len(value)!=length or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid immutable snapshot provenance: '+name)
    files=set()
    for folder in ('mamba_e8w5','scripts','tests','docs','licenses','third_party'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(p for p in ROOT.iterdir() if p.is_file())
    files.update(ROOT/name for name in PINS)
    suffixes={'.py','.cpp','.c','.h','.hpp','.md','.toml','.txt','.json','.sh','.yml','.yaml'}
    entries=[]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in sorted(files):
            name=path.relative_to(ROOT).as_posix()
            if path.is_symlink(): raise ValueError('Source archive cannot contain symlinks: '+name)
            if any(part.startswith('.') or part=='__pycache__' for part in Path(name).parts): continue
            if path.suffix not in suffixes and path.name not in ('LICENSE','COPYING','NOTICE'): continue
            entry={'path':name,'bytes':path.stat().st_size,'sha256':sha(path)}
            archive.write(path,name)
            if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise ValueError('Immutable source changed while archiving: '+name)
            entries.append(entry)
        index={'git_metadata_present':False,'git_commit':None,'git_worktree_dirty':None,
            'snapshot_provenance':{**provenance,'sha256':sha(provenance_path)},'files':entries,
            'evidence_scope':f'Only the {len(PINS)} explicitly pinned reports are included; all other historical reports remain in Git and are omitted from source.zip. Resolve other documentation report links from the immutable release tag in the public repository.',
            'source_scope':'Exact available source, tests, docs, licenses and pinned QuIP files from the declared immutable export.'}
        archive.writestr('SOURCE_FILES.json',encoded(index))
    # Independently read every archived source member and verify declared bytes.
    with zipfile.ZipFile(target) as archive:
        if len(archive.namelist())!=len(entries)+1 or len(set(archive.namelist()))!=len(entries)+1:
            raise ValueError('Unexpected source archive member inventory')
        for entry in entries:
            data=archive.read(entry['path'])
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('Corresponding source archive readback differs')
    return index


def model_card(raw):
    q=raw['quality']; score=q['full_ppl']; base=q['paired_base_ppl']; source=q['historical_source_full_ppl']
    mk=q['confirmation']; current=mk['current_readapted']; active=mk['active_resurface']
    return f'''# Mamba2-8B axis E8 + soft Resurface — {TAG}

Public research prerelease of NVIDIA Mamba2-8B with 112 E8-plus-axis projections,
W4 input embeddings, W5 output head, 393 readapted FP16 small tensors and the same
enabled soft Resurface-inspired adapter for every task. No external weight base
or original checkpoint is needed for inference. The adapter is mandatory.

## Measured quality

Full WikiText-2 validation: PPL **{score['ppl']:.12f}**, compared with
**{base['ppl']:.12f}** for the readapted compressed base without the adapter;
{score['windows']} windows, {score['target_tokens']:,} predicted tokens, 2048-token
windows with the declared final tail. Validation informed development; these
are not untouched test results.

The aligned historical original FP16 source scored **{source['ppl']:.12f}**;
this release is **{q['source_relative_ppl_change_percent']:+.6f}%** higher (worse)
in PPL. The original source was not rerun in the soft-adapter evaluation process.
This is not an equal-quality claim.

Fresh synthetic multi-key confirmation: **{active['normal']['correct']}/{active['normal']['count']}**
normal exact answers versus **{current['normal']['correct']}/{current['normal']['count']}**
without the adapter. Target-removed controls: **{active['target_removed']['correct']}/{active['target_removed']['count']}**
versus **{current['target_removed']['correct']}/{current['target_removed']['count']}**.
These task-specific results do not establish general reasoning performance.
Original-source MK was not measured on this fresh confirmation split; earlier
DEV comparisons are separately documented in the repository.

The original strict DEV run stopped on cross-process historical MK mismatch.
A separately declared continuation retained the same final adapter and required
same-process baseline restoration, full PPL nonregression and fresh confirmation.
The release does not claim that the original strict prerequisite passed. Exact
reports and hashes are shipped with this distribution.

## Storage and execution

The base model data is **3,138,928,792 bytes**; the actual serialized adapter is
**2,539,647 bytes**, totaling **3,141,468,439 bytes** before tokenizer and manifests.
The adapter has 1,154,104 parameters (2,308,208 FP16 payload bytes).
All shipped assets and metadata are counted in `release_manifest.json`.

This E8HUF001 envelope uses raw members: it is exact to the quantized files and
does not add entropy compression. Quantization itself is lossy relative to the
original model. The native quality runtime expands base weights to FP16; the
download size is not its GPU memory usage. Native Mamba dependencies and a
compatible CUDA environment are required; no optimized compressed GPU kernel,
ASIC performance or globally smallest-model claim is made.

## Download, restore, run

Use `scripts/download_release.py` with tag `{TAG}`, then extract `source.zip`.
Run the archived `scripts/package_release.py restore` with the independently
published release-manifest SHA to restore exact files. Load the restored `raw`
directory only through `mamba_e8w5.release_runtime.load_model`, or use archived
`python -m mamba_e8w5.release_generate --help`. That entry point verifies the soft adapter
and all model identities; directly calling the old base loader would omit it.

See archived `docs/DOWNLOAD.md` and `docs/RESURFACE_RELEASE.md` for exact commands.
Corresponding source is in `source.zip`; license notices are in `licenses/`.
The upstream NVIDIA model/tokenizer and native Mamba runtime are Apache-2.0;
this repository's software, including the QuIP-derived codec, is GPL-3.0.
Read the shipped notices for the component-specific terms.
Only selected pinned evidence is archived. For other report links in the docs,
use the public repository at tag `{TAG}`; these reports are not inference inputs.
The original checkpoint is needed only to reproduce quantization/training,
not to download, restore or use this package.
'''


def build(args, report, save):
    workspace,raw_dir,destination,restored=(getattr(args,k).resolve() for k in
        ('workspace','raw_dir','release_dir','restored_dir'))
    staging=destination.with_name(destination.name+'.building')
    for path in (raw_dir,destination,staging,restored):
        if path.exists(): raise FileExistsError(path)
        if path==ROOT or ROOT in path.parents:
            raise ValueError('Build outputs must be outside the immutable source snapshot')
    if len({raw_dir,destination,staging,restored})!=4:
        raise ValueError('Distinct raw/release/restore paths required')
    for first in (raw_dir,destination,staging,restored):
        if any(first in second.parents for second in (raw_dir,destination,staging,restored) if first!=second):
            raise ValueError('Build directories cannot contain each other')
    if not (ROOT/'SNAPSHOT_PROVENANCE.json').is_file() or (ROOT/'.git').exists():
        raise ValueError('Run from the immutable source export, not the working checkout')
    # Raw composition uses hardlinks. Restore from final parts temporarily joins
    # one container while writing one independent raw copy: at most3 raw sizes.
    destination.parent.mkdir(parents=True,exist_ok=True)
    tokenizer_bytes=(workspace/'models/source'/TOKENIZER_FILENAME).stat().st_size
    raw_upper=3141468439+tokenizer_bytes+(8<<20)
    required=3*raw_upper+(512<<20)
    free=shutil.disk_usage(destination.parent).free
    if free<required: raise RuntimeError(f'Need {required} available bytes; observed {free}')
    report['disk_budget']={'available_before':free,'required_conservative':required,
        'assumption':'Hardlink raw; final parts plus temporary joined container plus one independent raw restore;512MiB for software/metadata.'}
    report['stage']='compose';save()
    raw,composition=compose(workspace,raw_dir)
    report['composition']=composition;save()
    staging.mkdir(); container=staging/'mamba2-8b-axis-resurface.raw.huff'
    report['stage']='pack_raw_members';save()
    pack_raw_members(raw_dir,container)
    report['stage']='full_container_readback';save()
    receipt=verify_container(container,trusted_directory=raw_dir)
    if receipt['files_verified']!=120 or receipt['independent_chunks_verified']!=0:
        raise ValueError('Unexpected raw-member container coverage')
    report['container_verification']=receipt;save()
    report['stage']='split_and_assets';save()
    parts=release.split_container(container,1_500_000_000)
    shutil.copyfile(raw_dir/TOKENIZER_FILENAME,staging/TOKENIZER_FILENAME)
    shutil.copyfile(raw_dir/'config.json',staging/'config.json')
    shutil.copyfile(raw_dir/'manifest.json',staging/'raw_manifest.json')
    release.copy_license_assets(ROOT,staging/'licenses')
    write(staging/'container_verification.json',receipt)
    write(staging/'composition_verification.json',composition)
    (staging/'MODEL_CARD.md').write_text(model_card(raw))
    quality_paths=[]
    (staging/'reports').mkdir()
    for name in PINS:
        source=ROOT/name;target=staging/'reports'/source.name
        shutil.copyfile(source,target)
        if sha(target)!=PINS[name]:raise ValueError('Copied evidence changed')
        if name.endswith(('_full_eval.json','_confirm_eval.json')):quality_paths.append(target.relative_to(staging).as_posix())
    source_index=source_archive(staging/'source.zip')
    assets=[release.record_asset(staging,p,'weight-container-part' if p.name in {r['path'] for r in parts} else 'distribution-support')
            for p in sorted(staging.rglob('*')) if p.is_file()]
    for part in parts:part['sha256']=next(e['sha256'] for e in assets if e['path']==part['path'])
    if any(not 0<e['bytes']<2**31 for e in assets):raise ValueError('Every GitHub asset must fit below2GiB')
    manifest={'format':release.FORMAT,'complete':True,'release_tag':TAG,'research_prerelease':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'container':{'format':'E8HUF001','storage_mode':'all-raw-members-no-entropy-coding',
            'bytes':receipt['actual_file_bytes'],'sha256':receipt['sha256'],'parts':parts,
            'raw_manifest_sha256':composition['raw_manifest_sha256'],
            'raw_package_bytes':receipt['original_package_bytes'],'raw_members':120},
        'tokenizer':{'path':TOKENIZER_FILENAME,'sha256':TOKENIZER_SHA256},
        'quality':{'status':'same-soft-adapter-full-PPL-and-fresh-synthetic-confirmation-reports-attached',
            'reports':quality_paths,'smallest_claim':'not certified by packaging'},
        'software':{'source_archive':'source.zip','git_commit':None,'git_worktree_dirty':None,
            'git_metadata_present':False,'snapshot_provenance':source_index['snapshot_provenance']},
        'assets':assets,'sizes':{'weight_container_bytes':receipt['actual_file_bytes'],
            'tokenizer_bytes':tokenizer_bytes,'configuration_bytes':(staging/'config.json').stat().st_size,
            'scope':'Every shipped asset+release manifest; includes both raw-contained and standalone tokenizer copies, source, licenses and quality evidence.'},
        'inference':{'entrypoint':'mamba_e8w5.release_runtime.load_model','adapter_required':True,'gate_mode':'soft'},
        'integrity_note':'Trust the independently published release_manifest.json SHA. Internal hashes alone do not provide authenticity.'}
    release.finalize_manifest(staging,manifest)
    report['stage']='final_release_verification';save()
    verified=release.verify_release(staging)
    # Also exercise the unchanged public downloader/uploader manifest contracts.
    from download_release import validate_manifest
    from upload_release import upload_plan
    validate_manifest((staging/'release_manifest.json').read_bytes(),verified['manifest_sha256'])
    upload_plan(staging/'release_manifest.json')
    audit_raw(raw_dir)
    for name,digest in PINS.items():
        if sha(ROOT/name)!=digest:raise ValueError('Evidence changed during build')
    os.replace(staging,destination)
    # Exercise exactly the public archived CLI after final manifest creation.
    # Its receipt binds the outer manifest SHA and every actual restored member.
    report['stage']='generic_public_restore';save()
    command=[sys.executable,'-u',str(ROOT/'scripts/package_release.py'),'restore',
        '--release-dir',str(destination),'--output',str(restored),
        '--expected-manifest-sha256',verified['manifest_sha256']]
    restore_log=args.report.with_name(args.report.stem+'.restore.log')
    report['restore_command']=command;report['restore_log']=str(restore_log);save()
    with restore_log.open('x') as stream:
        process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    report['restore_exit_code']=process.returncode;save()
    if process.returncode:raise RuntimeError('Public restore CLI failed; inspect '+str(restore_log))
    restored_raw=audit_raw(restored/'raw')
    restore_receipt=json.loads((restored/'restore_receipt.json').read_text())
    if (restored_raw!=raw or sha(restored/'raw/manifest.json')!=composition['raw_manifest_sha256'] or
        restore_receipt.get('complete') is not True or restore_receipt['manifest_sha256']!=verified['manifest_sha256'] or
        restore_receipt['restored_raw_files']!=120 or restore_receipt['container_sha256']!=receipt['sha256']):
        raise ValueError('Public restored package or bound receipt differs')
    report['restore']=restore_receipt
    report.update(complete=True,stage='complete',release_dir=str(destination),raw_dir=str(raw_dir),
        restored_dir=str(restored),release_manifest_sha256=verified['manifest_sha256'],
        total_release_bytes=verified['total_release_bytes'],container_bytes=receipt['actual_file_bytes'],
        container_sha256=receipt['sha256'],gpu_used=False,quality_remeasured=False,
        public_manifest_contracts_verified=True,available_disk_after=shutil.disk_usage(destination.parent).free)


def self_test():
    """Tiny full-byte raw container/restore/split checks using the frozen Reader."""
    with tempfile.TemporaryDirectory(prefix='resurface-release-cpu-') as temporary:
        root=Path(temporary);source=root/'source';source.mkdir()
        payloads={'manifest.json':b'{"complete":true}\n','axis.e8':bytes(range(256))*257,
                  'embedding.uniform':b'\x00\xff\x80\x7f'*4097,'adapter_fp16.pt':b'unchanged-adapter-fixture'}
        for name,data in payloads.items():(source/name).write_bytes(data)
        target=root/'fixture.huff';pack_raw_members(source,target)
        verified=verify_container(target,trusted_directory=source)
        assert verified['files_verified']==4 and verified['independent_chunks_verified']==0
        reader=Reader(target);assert reader.manifest['codebooks']=={}
        assert all(e['family'] is None for e in reader.files.values())
        reader.restore(root/'restored')
        assert {p.name:p.read_bytes() for p in (root/'restored').iterdir()}==payloads
        with target.open('r+b') as stream:stream.seek(24);byte=stream.read(1);stream.seek(24);stream.write(bytes([byte[0]^1]))
        try:verify_container(target,trusted_directory=source)
        except ValueError:pass
        else:raise AssertionError('Corrupted raw member accepted')
        target2=root/'second.huff';pack_raw_members(source,target2);wanted=target2.read_bytes()
        parts=release.split_container(target2,32768)
        assert not target2.exists() and b''.join((root/e['path']).read_bytes() for e in parts)==wanted
        assert sum(e['bytes'] for e in parts)==len(wanted)
    return {'complete':True,'checks':['full-byte raw-member restore','empty codebook and no entropy claim',
        'corruption rejected','split concatenation exact'],'frozen_reader_used':True,'gpu_used':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/home/horde/Mamba2-8B-E8W5'))
    parser.add_argument('--raw-dir',type=Path)
    parser.add_argument('--release-dir',type=Path)
    parser.add_argument('--restored-dir',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise RuntimeError('CUDA must be hidden and uninitialized for CPU packaging')
    torch.set_num_threads(8)
    args.report=args.report.resolve()
    if args.report.exists():raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    report={'format':'MAMBA2_AXIS_RESURFACE_DISTRIBUTION_BUILD_V1','complete':False,
        'script_sha256':sha(__file__),'pid':os.getpid(),'started_unix':time.time(),
        'publication_performed':False,'gpu_used':False,'command':sys.argv}
    with args.report.open('x') as stream:stream.write('{}\n')
    def save():write(args.report,report)
    try:
        if args.self_test:report.update(complete=True,self_test=self_test())
        else:
            if any(getattr(args,k) is None for k in ('raw_dir','release_dir','restored_dir')):
                raise ValueError('Pass all three exclusive output directories')
            build(args,report,save)
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix']=time.time();report['cuda_initialized']=torch.cuda.is_initialized();save()
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
