"""Verified direct current-axis runtime/data for small-tensor readaptation.

No Hessian or superseded projection payload is read. Source checkpoint casting,
actual current files and prepared TRAIN identities are independently checked.
This module never invokes a training forward or loads validation data.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch

from . import codec, runtime
from .input_axis_residual import read_axis_e8

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
PROTOCOL_SHA = 'a2509bec3d82748b18e452c9bbb5053a894c96ed5309eeb67d45f7dfa6029282'
CURRENT_SHA = '23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4'
INPUT_SHA = '0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85'
OUTPUT_SHA = '2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f'
PARENT_SHA = 'ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed'
SMALL_SHA = 'edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006'
SMALL_VALUES_SHA = '15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4'
DATA_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
CODE = {
    'mamba_e8w5/runtime.py': 'bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369',
    'mamba_e8w5/codec.py': 'ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f',
    'mamba_e8w5/input_axis_residual.py': 'baff8c56c875258830ac00076f880105b7933a173f57ba0bf38e27cbf1436c06',
    'mamba_e8w5/calibration.py': 'b5a7356a2ab769538940aed218653a092a7055aa26a9ec6f62d3783b9ac78ee5',
    'scripts/prepare_teacher_kl_data.py': '4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13',
}
LARGE_KEYS = tuple(f'backbone.layers.{i}.mixer.{part}.weight'
    for i in range(56) for part in ('in_proj','out_proj')) + ('backbone.embedding.weight','lm_head.weight')


def small_inventory():
    result = {}
    for i in range(56):
        for suffix, shape in (
            ('norm.weight',[4096]), ('mixer.norm.weight',[8192]),
            ('mixer.dt_bias',[128]), ('mixer.A_log',[128]), ('mixer.D',[128]),
            ('mixer.conv1d.weight',[10240,1,4]), ('mixer.conv1d.bias',[10240])):
            result[f'backbone.layers.{i}.{suffix}'] = shape
    result['backbone.norm_f.weight'] = [4096]
    return result


def tensor_hash(value):
    value = value.detach()
    if value.is_meta:
        raise ValueError('Cannot hash an unmaterialized tensor')
    digest = hashlib.sha256()
    rows = value.reshape(1) if value.ndim == 0 else value
    for first in range(0,len(rows),128):
        digest.update(rows[first:first+128].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_hashes(model):
    return {name:tensor_hash(value) for name,value in model.named_parameters()}


def checked_file(path, digest, size=None):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or (size is not None and path.stat().st_size != size)
            or runtime.sha256_file(path) != digest):
        raise ValueError(f'Bound regular-file identity differs:{path}')


def validate_small(values, expected=None):
    inventory = small_inventory()
    if not isinstance(values,dict) or set(values) != set(inventory):
        raise ValueError('Expected exactly393 eligible small tensors')
    hashes = {}
    for name,shape in inventory.items():
        value = values[name]
        if (not isinstance(value,torch.Tensor) or value.dtype != torch.float16
                or list(value.shape) != shape or value.is_meta or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid finite native FP16 small tensor:{name}')
        hashes[name] = tensor_hash(value)
        if expected is not None and hashes[name] != expected[name]:
            raise ValueError(f'Small tensor content differs:{name}')
    count = sum(value.numel() for value in values.values())
    if count != 3580928:
        raise ValueError('Small scalar-parameter count differs')
    return {'tensor_count':393, 'parameter_count':count, 'fp16_tensor_sha256':hashes}


def audit_model(model, expected):
    params = dict(model.named_parameters())
    if (set(params) != set(expected) or len(params) != 507
            or sum(value.numel() for value in params.values()) != 8236999680
            or any(value.dtype != torch.float16 or value.is_meta for value in params.values())):
        raise ValueError('Native507/8,236,999,680 FP16 parameter coverage differs')
    actual = parameter_hashes(model)
    if actual != expected:
        wrong = [name for name in actual if actual[name] != expected[name]]
        raise ValueError(f'Actual native parameter content differs:{wrong[:8]}')
    return {'verified_tensor_count':507, 'parameter_count':8236999680,
            'unchanged_content_sha256':actual}


def large_identity(model):
    return {name:(id(model.get_parameter(name)),model.get_parameter(name).data_ptr(),
                  model.get_parameter(name)._version) for name in LARGE_KEYS}


def verify_fixed_large(model, context, identity=None):
    actual_identity = large_identity(model)
    if identity is not None and actual_identity != identity:
        raise ValueError('A fixed large tensor object/storage/version changed')
    hashes = {name:tensor_hash(model.get_parameter(name)) for name in LARGE_KEYS}
    expected = {name:context['current_expected507'][name] for name in LARGE_KEYS}
    if hashes != expected:
        raise ValueError('Fixed114 axis/vocabulary tensor content changed')
    return {'verified_tensor_count':114, 'unchanged_content_sha256':hashes,
            'identities':actual_identity, 'identity_match':identity is None or identity==actual_identity}


@torch.no_grad()
def install_small(model, values, expected=None):
    receipt = validate_small(values,expected)
    for name,value in values.items():
        old = model.get_parameter(name)
        device = old.device if not old.is_meta else next(p.device for p in model.parameters() if not p.is_meta)
        runtime._set_parameter(model,name,value,device,torch.float16)
    return receipt


def verify_inputs():
    parent_dir = ROOT/'artifacts/e8w5_v1'
    input_dir = ROOT/'artifacts/input_axis_residual_v1'
    output_dir = ROOT/'artifacts/output_axis_residual_v1'
    small_dir = ROOT/'artifacts/small_compensation_v1'
    data_dir = ROOT/'training_data/teacher_kl_compensation_v1'
    pins = {
        ROOT/'docs/AXIS_SMALL_READAPTATION_PROTOCOL.md':PROTOCOL_SHA,
        ROOT/'reports/output_axis_residual_v1_eval.json':CURRENT_SHA,
        parent_dir/'manifest.json':PARENT_SHA, input_dir/'manifest.json':INPUT_SHA,
        output_dir/'manifest.json':OUTPUT_SHA, small_dir/'manifest.json':SMALL_SHA,
        small_dir/'other_fp16.pt':SMALL_VALUES_SHA, data_dir/'manifest.json':DATA_SHA,
        Path(__file__).resolve():runtime.sha256_file(__file__)}
    pins.update({ROOT/name:digest for name,digest in CODE.items()})
    for path,digest in pins.items():
        checked_file(path,digest)
    read = lambda path: json.loads(Path(path).read_text())
    parent,inputs,outputs,small = [read(directory/'manifest.json') for directory in (parent_dir,input_dir,output_dir,small_dir)]
    prior = read(ROOT/'reports/output_axis_residual_v1_eval.json')
    if (any(document.get('complete') is not True for document in (parent,inputs,outputs,small,prior))
            or any(document['model_config'] != runtime.MODEL_CONFIG for document in (parent,inputs,outputs,small))
            or prior['primary_arm'] != 'axis_in_out_w4_w5' or not prior['baseline_repeat_exact']
            or not prior['full_validation']['performed']
            or prior['integrity']['overlay_manifest_sha256'] != OUTPUT_SHA
            or prior['integrity']['input_manifest_sha256'] != INPUT_SHA):
        raise ValueError('Pinned current-model chain differs')
    current_expected = prior['integrity']['resolved507_fp16_sha256_by_arm']['axis_in_out_w4_w5']
    source_expected = prior['full_validation']['arms']['source_fp16']['actual507_before']['unchanged_content_sha256']
    inventory = small_inventory()
    if (len(inventory) != 393 or set(inventory)&set(LARGE_KEYS)
            or set(current_expected) != set(inventory)|set(LARGE_KEYS)
            or set(source_expected) != set(current_expected)):
        raise ValueError('Expected393 small/114 large/507 total identities')
    resolved = {}
    for layer in range(56):
        for part,directory,manifest,shape in (
            ('in_proj',input_dir,inputs,[18560,4096]),('out_proj',output_dir,outputs,[4096,8192])):
            label = f'layer{layer}.{part}'
            name = f'backbone.layers.{layer}.mixer.{part}.weight'
            entry,file = manifest['matrices'][label],manifest['files'][label+'.e8']
            if (entry['shape'] != shape or entry['source_key'] != name or entry['index_bits'] != 20
                    or entry['values_per_index'] != 8 or entry['file'] != label+'.e8'
                    or file != {key:entry[key] for key in ('bytes','sha256','decoded_fp16_sha256')}
                    or file['decoded_fp16_sha256'] != current_expected[name]):
                raise ValueError(f'Current axis matrix identity differs:{label}')
            resolved[label+'.e8'] = {**file,'path':str(directory/(label+'.e8')),
                                    'origin':directory.name,'source_key':name,'shape':shape}
    inherited = {name:row for name,row in outputs['inherited_files'].items() if not name.endswith('.e8')}
    if set(inherited) != {'embedding.uniform','lm_head.uniform','other_fp16.pt','e8_codebook.bin','config.json'}:
        raise ValueError('Five current inherited files differ')
    origins = {'parent':parent_dir,'small_compensation_v1':small_dir,'vocab_w4_v1':ROOT/'artifacts/vocab_w4_v1'}
    for name,row in inherited.items():
        resolved[name] = {**row,'path':str(origins[row['origin']]/name)}
    if (resolved['other_fp16.pt']['sha256'] != SMALL_VALUES_SHA
            or len(resolved) != 117 or sum(row['bytes'] for row in resolved.values()) != 3138928792):
        raise ValueError('Current117-file raw model accounting differs')
    for name,row in resolved.items():
        path = Path(row['path'])
        if path in pins and pins[path] != row['sha256']:
            raise ValueError('Contradictory actual model identity')
        if path not in pins:
            checked_file(path,row['sha256'],row['bytes'])
        elif path.stat().st_size != row['bytes']:
            raise ValueError('Current model file length differs')
        pins[path] = row['sha256']
    if read(resolved['config.json']['path']) != runtime.MODEL_CONFIG:
        raise ValueError('Native model configuration differs')
    for name,bits in (('embedding.uniform',4),('lm_head.uniform',5)):
        header,_ = codec.uniform_header(resolved[name]['path'])
        if header != {'shape':[256000,4096],'bits':bits,'group':128,'scale':'fp16'}:
            raise ValueError('Fixed native vocabulary file geometry differs')
    for name,digest in parent['binding']['quip_sha256'].items():
        path = ROOT/'third_party/quip-sharp'/name
        checked_file(path,digest)
        pins[path] = digest
    tokenizer_path = ROOT/'models/source'/runtime.TOKENIZER_FILENAME
    checked_file(tokenizer_path,runtime.TOKENIZER_SHA256)
    pins[tokenizer_path] = runtime.TOKENIZER_SHA256
    # Safe pinned runtime loader hashes the full source once and preserves mmap.
    # No full source model is instantiated on the GPU here.
    source = runtime.load_source_state(ROOT/'models/source')
    source_small = {name:source[name].to(dtype=torch.float16).contiguous().clone() for name in inventory}
    source_types = {name:str(source[name].dtype) for name in inventory}
    del source
    source_cast = validate_small(source_small,source_expected)
    source_cast['original_dtypes'] = source_types
    source_cast['cast'] = 'Verified original checkpoint values cast directly to native FP16'
    pins[runtime.checkpoint_path(ROOT/'models/source')] = runtime.SOURCE_CHECKPOINT_SHA256
    old_small = torch.load(small_dir/'other_fp16.pt',map_location='cpu',weights_only=True)
    old_receipt = validate_small(old_small,current_expected)
    prepared = importlib.import_module('prepare_teacher_kl_data')
    data,training,heldout = prepared.verify_data(data_dir,DATA_SHA)
    # Small historical provenance documents are required by the frozen data
    # verifier; their Hessian/weight payloads are not traversed.
    for name,digest in {**prepared.BOUND,**prepared.CODE_BOUND}.items():
        path = ROOT/name
        if path in pins and pins[path] != digest:
            raise ValueError('Data/model provenance contradiction')
        pins[path] = digest
    pins[ROOT/'docs/TEACHER_KL_COMPENSATION_PROTOCOL.md'] = prepared.PROTOCOL_SHA
    for filename,key in (('training_tokens.pt','training_tokens_file_sha256'),('heldout_tokens.pt','heldout_tokens_file_sha256')):
        pins[data_dir/filename] = data[key]
    plan = [{'index':index,**row,'target_tokens':row['targets']} for index,row in enumerate(data['heldout_windows'])]
    if (plan != prior['development_plan'] or data['dataset'] != prior['dataset']
            or tuple(training.shape) != (448,2048) or tuple(heldout.shape) != (64,2048)):
        raise ValueError('Fixed TRAIN data/current development plan differs')
    hybrid = {**current_expected,**source_cast['fp16_tensor_sha256']}
    binding = {'protocol_sha256':PROTOCOL_SHA,'current_evaluation_sha256':CURRENT_SHA,
        'input_manifest_sha256':INPUT_SHA,'output_manifest_sha256':OUTPUT_SHA,'parent_manifest_sha256':PARENT_SHA,
        'small_manifest_sha256':SMALL_SHA,'small_values_sha256':SMALL_VALUES_SHA,
        'source_checkpoint_sha256':runtime.SOURCE_CHECKPOINT_SHA256,'tokenizer_sha256':runtime.TOKENIZER_SHA256,
        'data_manifest_sha256':DATA_SHA,'training_tokens_file_sha256':data['training_tokens_file_sha256'],
        'heldout_tokens_file_sha256':data['heldout_tokens_file_sha256'],
        'helper_source_sha256':runtime.sha256_file(__file__),'frozen_code_sha256':CODE,
        'checked_input_sha256':{str(path):digest for path,digest in pins.items()}}
    return {'binding':binding,'parent':parent,'input':inputs,'output':outputs,'small':small,'prior':prior,
        'current_expected507':current_expected,'source_expected507':source_expected,'original_small_expected507':hybrid,
        'old_small':old_small,'source_small':source_small,'fixed_large_keys':LARGE_KEYS,'resolved_files':resolved,
        'source393_cast_receipt':source_cast,'old393_receipt':old_receipt,'data_manifest':data,
        'training_windows':training,'heldout_windows':heldout,'heldout_plan':plan}


@torch.no_grad()
def load_model(context, device='cuda'):
    model = runtime.make_model(device='meta',dtype=torch.float16)
    book,_ = codec.load_reference_primitives()
    if (book.grid_packed_abs.cpu().numpy().astype('<i4').tobytes()
            != Path(context['resolved_files']['e8_codebook.bin']['path']).read_bytes()):
        raise ValueError('Stored current codebook differs from pinned decoder')
    book = book.to(device).requires_grad_(False)
    installed = set()
    decoded_hashes = {}
    for layer in range(56):
        for part in ('in_proj','out_proj'):
            row = context['resolved_files'][f'layer{layer}.{part}.e8']
            payload,_ = read_axis_e8(row['path'],expected_shape=row['shape'])
            decoded = codec.decode_e8(payload,book,device=device)
            digest = tensor_hash(decoded)
            if not torch.isfinite(decoded).all() or digest != context['current_expected507'][row['source_key']]:
                raise ValueError('Independent current-axis native decode differs')
            runtime._set_parameter(model,row['source_key'],decoded,device,torch.float16)
            installed.add(row['source_key']);decoded_hashes[row['source_key']] = digest
            del payload,decoded
    for filename,name in (('embedding.uniform','backbone.embedding.weight'),('lm_head.uniform','lm_head.weight')):
        output = torch.empty((256000,4096),dtype=torch.float16,device=device)
        next_row = 0
        for first,rows in codec.iter_uniform(context['resolved_files'][filename]['path'],device=device,chunk_rows=256):
            if first != next_row:
                raise ValueError('Noncontiguous current vocabulary decode')
            output[first:first+len(rows)].copy_(rows);next_row += len(rows)
        digest = tensor_hash(output)
        if next_row != 256000 or not torch.isfinite(output).all() or digest != context['current_expected507'][name]:
            raise ValueError('Actual current vocabulary decode differs')
        runtime._set_parameter(model,name,output,device,torch.float16)
        installed.add(name);decoded_hashes[name] = digest
        del output,rows
    for name,value in context['old_small'].items():
        runtime._set_parameter(model,name,value,device,torch.float16);installed.add(name)
    if installed != set(context['current_expected507']) or any(p.is_meta for p in model.parameters()):
        raise ValueError('Direct current-model loader left a missing/extra tensor')
    if model.backbone.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise ValueError('Fixed untied vocabulary matrices became tied')
    model._package_receipt = {'format':'Current112-axis/W4-embedding/W5-head/old393 native FP16 reference',
        'binding':context['binding'],'resolved_files':context['resolved_files'],
        'logical_model_data_bytes':sum(row['bytes'] for row in context['resolved_files'].values()),
        'parameter_count':8236999680,'independent114_decoded_hashes':decoded_hashes}
    model._axis_load_audit = audit_model(model,context['current_expected507'])
    del book
    gc.collect()
    return model.eval().requires_grad_(False)


def final_recheck(context):
    for path,digest in context['binding']['checked_input_sha256'].items():
        checked_file(path,digest)
    return {'verified_file_count':len(context['binding']['checked_input_sha256']),
            'checked_input_sha256':context['binding']['checked_input_sha256']}


def self_test():
    inventory = small_inventory()
    assert len(inventory)==393 and len(LARGE_KEYS)==114 and not set(inventory)&set(LARGE_KEYS)
    assert sum(torch.tensor(shape).prod().item() for shape in inventory.values())==3580928
    values = {name:torch.zeros(shape,dtype=torch.float16) for name,shape in inventory.items()}
    proof = validate_small(values)
    assert proof['parameter_count']==3580928
    broken = dict(values);broken.pop(next(iter(broken)))
    try:validate_small(broken)
    except ValueError:pass
    else:raise AssertionError('Missing eligible tensor accepted')
    name = next(iter(values));broken = dict(values);broken[name] = values[name].float()
    try:validate_small(broken)
    except ValueError:pass
    else:raise AssertionError('Wrong serialized precision accepted')
    plus,minus = torch.tensor([0.],dtype=torch.float16),torch.tensor([-0.],dtype=torch.float16)
    assert tensor_hash(plus)!=tensor_hash(minus)
    assert not torch.cuda.is_initialized()
    return {'passed':True,'cuda_initialized':False,'checks':[
        'Exact393/3580928 inventory disjoint from114 fixed keys',
        'Missing-key/precision rejection and signed-zero content hashing']}
