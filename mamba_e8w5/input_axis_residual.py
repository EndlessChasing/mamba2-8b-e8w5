"""Strict reader and immutable overlay checks for the input-axis experiment."""
import hashlib
import json
import math
from pathlib import Path
import struct
import torch
from . import codec
from .runtime import MODEL_CONFIG

FORMAT='MAMBA2_INPUT_AXIS_RESIDUAL_V1'
LABELS=tuple(f'layer{i}.in_proj' for i in range(56))
SHAPE=[18560,4096]
RECIPE={'residual_bits':4,'index_bits':20,'values_per_index':8,'tune_iters':2,
    'damping':.01,'scale_override':.9,'feedback':True,'seed_rule':'1000 + 2*layer'}


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def read_axis_e8(path,expected_shape=None):
    """Validate the two-plane 20-bit file before frozen-reader allocation."""
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Regular E8 file required')
    with path.open('rb') as stream:
        if stream.read(8)!=b'ME8HD001':raise ValueError('Wrong E8 magic')
        size=stream.read(4)
        if len(size)!=4:raise ValueError('Truncated E8 header')
        length=struct.unpack('<I',size)[0]
        if not 1<=length<=65536:raise ValueError('Invalid E8 header length')
        raw=stream.read(length)
        if len(raw)!=length:raise ValueError('Truncated E8 JSON')
        header=json.loads(raw)
    if (set(header)!={'format','shape','index_order','rotation','axis_residual_amplitude'}
            or header['format']!='mamba-e8-dct-hadamard-v1' or header['index_order']!='row-major'
            or header['rotation']!='dct2-kron-hadamard; maximal power-of-two factor'):
        raise ValueError('Unsupported E8 header')
    shape=header['shape'];amplitude=header['axis_residual_amplitude']
    if (not isinstance(shape,list) or len(shape)!=2 or any(type(x)!=int or x<=0 for x in shape)
            or shape[1]%8 or math.prod(shape)%16 or (expected_shape is not None and shape!=list(expected_shape))):
        raise ValueError('Invalid even-code matrix geometry')
    if type(amplitude) not in (float,int) or not math.isfinite(amplitude) or amplitude<=0:
        raise ValueError('Axis amplitude must be finite and strictly positive')
    m,n=shape;count=m*n//8
    layout={'shape':shape,'index_count':count,'index_bits':20,'header_bytes':12+length,
        'base_uint16_bytes':count*2,'axis_nibble_bytes':count//2,'balance_fp16_bytes':n*2,
        'input_sign_bytes':(n+7)//8,'output_sign_bytes':(m+7)//8,'scale_fp32_bytes':4,
        'axis_residual_amplitude':amplitude,'header':header,'header_sha256':hashlib.sha256(raw).hexdigest()}
    layout['bytes']=sum(layout[k] for k in ('header_bytes','base_uint16_bytes','axis_nibble_bytes',
        'balance_fp16_bytes','input_sign_bytes','output_sign_bytes','scale_fp32_bytes'))
    if path.stat().st_size!=layout['bytes']:raise ValueError('Axis file length mismatch')
    payload=codec.read_e8(path);indices=payload['indices']
    if (indices.dtype!=torch.int32 or tuple(indices.shape)!=(m,n//8)
            or int(indices.min())<0 or int(indices.max())>=1<<20):raise ValueError('Invalid 20-bit indices')
    if (payload['balance'].dtype!=torch.float16 or tuple(payload['balance'].shape)!=(n,)
            or not torch.isfinite(payload['balance']).all() or not (payload['balance']>0).all()
            or payload['scale'].numel()!=1 or not torch.isfinite(payload['scale']).all() or not payload['scale']>0):
        raise ValueError('Invalid finite positive balance/scale')
    if any(not torch.all((payload[k]==1)|(payload[k]==-1)) for k in ('input_sign','output_sign')):
        raise ValueError('Invalid sign vectors')
    return payload,layout


def verify_overlay(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    names={label+'.e8' for label in LABELS}
    if (manifest.get('format')!=FORMAT or manifest.get('complete') is not True
            or manifest.get('model_config')!=MODEL_CONFIG or manifest.get('recipe')!=RECIPE
            or set(manifest.get('matrices',{}))!=set(LABELS) or set(manifest.get('files',{}))!=names
            or {p.name for p in directory.iterdir()}!=names|{'manifest.json'}):
        raise ValueError('Incomplete or unexpected axis overlay inventory/recipe')
    for layer,label in enumerate(LABELS):
        entry=manifest['matrices'][label];path=directory/(label+'.e8');file=manifest['files'][path.name]
        if (entry['file']!=path.name or entry['shape']!=SHAPE or entry['seed']!=1000+2*layer
                or entry['source_key']!=f'backbone.layers.{layer}.mixer.in_proj.weight'
                or entry['index_bits']!=20 or entry['values_per_index']!=8
                or entry['tune_iters']!=2 or entry['damping']!=.01 or entry['scale_override']!=.9
                or entry['feedback'] is not True or entry['disk_roundtrip_fp16_equal'] is not True
                or entry['packed_indices_equal'] is not True):raise ValueError('Axis matrix recipe differs')
        if file!={k:entry[k] for k in ('bytes','sha256','decoded_fp16_sha256')}:
            raise ValueError('Matrix/file ledgers differ')
        if path.stat().st_size!=file['bytes'] or sha(path)!=file['sha256']:raise ValueError('Axis file identity differs')
        payload,layout=read_axis_e8(path,SHAPE)
        if layout!=entry['layout'] or payload['axis_residual_amplitude']!=entry['axis_residual_amplitude']:
            raise ValueError('Axis header/length metadata differs')
        if len(entry['decoded_fp16_sha256'])!=64:raise ValueError('Missing decoded tensor identity')
    if sum(row['bytes'] for row in manifest['files'].values())!=manifest['data_file_bytes']:
        raise ValueError('Overlay byte ledger differs')
    return manifest
