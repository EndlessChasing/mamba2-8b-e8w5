"""Standalone v0.2 axis/Resurface release loader.

Only the restored package and this shipped software snapshot are inputs. No
source checkpoint, Hessian, dataset or training receipt is opened. Weights are
expanded to FP16; this is not compressed-resident inference. The mandatory soft
adapter is installed for every prompt, including prose and recall.
"""
from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path

import torch

from . import codec, resurface_native as native, runtime
from .input_axis_residual import read_axis_e8

FORMAT = 'MAMBA2_AXIS_RESURFACE_RESOLVED_V1'
BASE_PARAMETERS = 8236999680
ADAPTER_PARAMETERS = 1154104
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CODE = frozenset({
    'mamba_e8w5/__init__.py', 'mamba_e8w5/release_runtime.py',
    'mamba_e8w5/release_generate.py', 'mamba_e8w5/runtime.py',
    'mamba_e8w5/codec.py', 'mamba_e8w5/input_axis_residual.py',
    'mamba_e8w5/resurface_native.py', 'mamba_e8w5/evaluation.py',
    'mamba_e8w5/calibration.py', 'third_party/quip-sharp/SOURCE.json',
    'third_party/quip-sharp/lib/codebook/latticee8_padded12.py',
    'third_party/quip-sharp/lib/algo/quip.py', 'third_party/quip-sharp/LICENSE',
})


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _json(path, maximum=16 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f'Invalid bounded regular JSON file: {path}')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def safe_file(root, name):
    """Portable relative path, with no symlink traversal inside the snapshot."""
    if (not isinstance(name, str) or '\\' in name or '\x00' in name
            or ':' in name or not name or name.startswith('/')):
        raise ValueError(f'Unsafe relative filename: {name!r}')
    parts = name.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError(f'Unsafe relative filename: {name!r}')
    path = Path(root)
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Symlink forbidden: {path}')
    if not path.is_file():
        raise ValueError(f'Required regular file absent: {path}')
    return path


def small_shapes():
    result = {}
    for i in range(56):
        for suffix, shape in (
            ('norm.weight', [4096]), ('mixer.norm.weight', [8192]),
            ('mixer.dt_bias', [128]), ('mixer.A_log', [128]), ('mixer.D', [128]),
            ('mixer.conv1d.weight', [10240, 1, 4]), ('mixer.conv1d.bias', [10240])):
            result[f'backbone.layers.{i}.{suffix}'] = shape
    result['backbone.norm_f.weight'] = [4096]
    return result


def projection_specs():
    return {f'layer{i}.{part}': {
        'file': f'layer{i}.{part}.e8',
        'source_key': f'backbone.layers.{i}.mixer.{part}.weight',
        'shape': [18560, 4096] if part == 'in_proj' else [4096, 8192],
        'index_bits': 20,
    } for i in range(56) for part in ('in_proj', 'out_proj')}


def vocabulary_specs():
    return {
        'embedding': {'file': 'embedding.uniform', 'source_key': 'backbone.embedding.weight',
                      'shape': [256000, 4096], 'bits': 4},
        'lm_head': {'file': 'lm_head.uniform', 'source_key': 'lm_head.weight',
                    'shape': [256000, 4096], 'bits': 5},
    }


def base_shapes():
    return {**small_shapes(), **{row['source_key']: row['shape']
        for row in list(projection_specs().values()) + list(vocabulary_specs().values())}}


def adapter_shapes():
    return {f'layer{i}.{name}': shape for i in range(56) for name, shape in (
        ('V_read', [128, 128]), ('g_read', [128]), ('router_w', [4096]), ('router_b', []))}


def payload_names():
    return {row['file'] for row in projection_specs().values()} | {
        'embedding.uniform', 'lm_head.uniform', 'other_fp16.pt', 'e8_codebook.bin',
        'config.json', 'adapter_fp16.pt', runtime.TOKENIZER_FILENAME}


def validate_manifest(manifest):
    """Geometry and complete inventory checks before opening tensor payloads."""
    if (not isinstance(manifest, dict) or manifest.get('format') != FORMAT
            or manifest.get('complete') is not True
            or manifest.get('model_config') != runtime.MODEL_CONFIG
            or type(manifest.get('total_parameter_count')) is not int
            or manifest['total_parameter_count'] != BASE_PARAMETERS
            or manifest.get('source_checkpoint_sha256') != runtime.SOURCE_CHECKPOINT_SHA256
            or manifest.get('tokenizer_sha256') != runtime.TOKENIZER_SHA256):
        raise ValueError('Release format, completeness, architecture or original identity differs')
    files = manifest.get('files', {})
    if not isinstance(files, dict) or set(files) != payload_names():
        raise ValueError('Expected exactly119 payload files, including the mandatory soft adapter')
    for name, row in files.items():
        if (not isinstance(row, dict) or type(row.get('bytes')) is not int or row['bytes'] <= 0
                or not _digest(row.get('sha256'))):
            raise ValueError(f'Invalid payload ledger: {name}')
    if (files['other_fp16.pt']['bytes'] > 64 << 20
            or files['adapter_fp16.pt']['bytes'] > 16 << 20
            or files['config.json']['bytes'] > 65536
            or files['e8_codebook.bin']['bytes'] != 1024
            or files[runtime.TOKENIZER_FILENAME]['sha256'] != runtime.TOKENIZER_SHA256):
        raise ValueError('Small payload bounds, codebook size or tokenizer identity differs')
    ledger = manifest.get('base507_fp16_sha256', {})
    if set(ledger) != set(base_shapes()) or not all(_digest(v) for v in ledger.values()):
        raise ValueError('Exact507 native parameter hash ledger required')
    for group, specs in (('matrices', projection_specs()), ('vocabularies', vocabulary_specs())):
        entries = manifest.get(group, {})
        if set(entries) != set(specs):
            raise ValueError(f'Incomplete {group} coverage')
        for label, spec in specs.items():
            row = entries[label]
            if any(row.get(k) != v for k, v in spec.items()):
                raise ValueError(f'Wrong fixed geometry or format: {label}')
            if row.get('decoded_fp16_sha256') != ledger[spec['source_key']]:
                raise ValueError(f'Decoded hash disagrees with native507 ledger: {label}')
    adapter = manifest.get('adapter', {})
    if (adapter.get('file') != 'adapter_fp16.pt' or adapter.get('gate_mode') != 'soft'
            or adapter.get('variant') != native.VARIANT
            or adapter.get('geometry') != [{'width': 4096, 'heads': 128, 'head_dim': 64}] * 56
            or type(adapter.get('parameters')) is not int or adapter['parameters'] != ADAPTER_PARAMETERS
            or type(adapter.get('payload_bytes')) is not int or adapter['payload_bytes'] != 2 * ADAPTER_PARAMETERS
            or not isinstance(adapter.get('binding'), dict)
            or adapter.get('sha256') != files['adapter_fp16.pt']['sha256']
            or adapter.get('bytes') != files['adapter_fp16.pt']['bytes']
            or set(adapter.get('tensor_sha256', {})) != set(adapter_shapes())
            or not all(_digest(v) for v in adapter['tensor_sha256'].values())):
        raise ValueError('Mandatory complete soft adapter identity differs')
    codes = manifest.get('binding', {}).get('code_sha256', {})
    if not isinstance(codes, dict) or not REQUIRED_CODE.issubset(codes) or not all(_digest(v) for v in codes.values()):
        raise ValueError('Required shipped decoder/runtime software bindings absent')
    for name in codes:
        if (not isinstance(name, str) or '\\' in name or ':' in name
                or name.startswith('/') or any(p in ('', '.', '..') for p in name.split('/'))):
            raise ValueError('Unsafe software binding path')
    return manifest


def tensor_hash(value):
    """Bounded CPU transfers, preserving raw FP16 signed-zero bits."""
    value = value.detach()
    if value.is_meta:
        raise ValueError('Cannot hash an unmaterialized tensor')
    rows = value.reshape(1) if value.ndim == 0 else value
    digest = hashlib.sha256()
    for first in range(0, len(rows), 128):
        digest.update(rows[first:first + 128].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_tensors(values, shapes, expected):
    if not isinstance(values, dict) or set(values) != set(shapes):
        raise ValueError('FP16 tensor payload inventory differs')
    actual = {}
    for name, shape in shapes.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float16
                or list(value.shape) != shape or value.is_meta or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid finite FP16 tensor: {name}')
        actual[name] = tensor_hash(value)
        if actual[name] != expected[name]:
            raise ValueError(f'Actual FP16 tensor hash differs: {name}')
    return actual


def verify_package(directory, expected_manifest_sha256=None):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Restored package must be a regular directory')
    manifest_path = safe_file(directory, 'manifest.json')
    digest = runtime.sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError('Trusted raw manifest SHA-256 mismatch')
    manifest = validate_manifest(_json(manifest_path))
    if {p.name for p in directory.iterdir()} != payload_names() | {'manifest.json'}:
        raise ValueError('Unlisted or missing file in restored package')
    for name, row in manifest['files'].items():
        path = safe_file(directory, name)
        if path.stat().st_size != row['bytes'] or runtime.sha256_file(path) != row['sha256']:
            raise ValueError(f'Actual file identity differs: {name}')
    for name, expected in manifest['binding']['code_sha256'].items():
        if runtime.sha256_file(safe_file(ROOT, name)) != expected:
            raise ValueError(f'Shipped source identity differs: {name}')
    if _json(directory / 'config.json', 65536) != runtime.MODEL_CONFIG:
        raise ValueError('Actual native configuration differs')
    for label, spec in vocabulary_specs().items():
        header, _ = codec.uniform_header(directory / spec['file'])
        if header['shape'] != spec['shape'] or header['bits'] != spec['bits']:
            raise ValueError(f'Actual stored vocabulary geometry/bitrate differs: {label}')
    small = torch.load(directory / 'other_fp16.pt', map_location='cpu', weights_only=True)
    small_hashes = _validate_tensors(small, small_shapes(), manifest['base507_fp16_sha256'])
    payload = native.read_fp16(directory / 'adapter_fp16.pt', expected_binding=manifest['adapter']['binding'])
    if (payload['gate_mode'] != 'soft' or payload['geometry'] != manifest['adapter']['geometry']
            or payload['variant'] != manifest['adapter']['variant']):
        raise ValueError('Actual adapter is not the required soft production adapter')
    adapter_hashes = _validate_tensors(payload['tensors'], adapter_shapes(), manifest['adapter']['tensor_sha256'])
    return manifest, {'format': FORMAT, 'manifest_sha256': digest,
        'verified_payload_files': 119, 'verified_payload_bytes': sum(row['bytes'] for row in manifest['files'].values()),
        'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
        'tokenizer_sha256': manifest['tokenizer_sha256'], 'source_checkpoint_opened': False,
        'training_artifacts_opened': False, 'small393_fp16_sha256': small_hashes,
        'adapter224_fp16_sha256': adapter_hashes, 'soft_adapter_required': True,
        'software_sha256': manifest['binding']['code_sha256']}


def _require_precision(device):
    if torch.device(device).type == 'cuda' and (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != 'highest' or torch.is_autocast_enabled('cuda')):
        raise ValueError('Set CUDA/cuDNN TF32 off, float32_matmul_precision highest and autocast off before loading')


@torch.no_grad()
def load_model(directory, device='cuda', expected_manifest_sha256=None):
    """Materialize native FP16 base and mandatory production soft adapter."""
    _require_precision(device)
    directory = Path(directory)
    manifest, receipt = verify_package(directory, expected_manifest_sha256)
    model = runtime.make_model(device='meta', dtype=torch.float16)
    book, _ = codec.load_reference_primitives()
    if book.grid_packed_abs.cpu().numpy().astype('<i4').tobytes() != (directory / 'e8_codebook.bin').read_bytes():
        raise ValueError('Actual E8 codebook differs from pinned decoder')
    book = book.to(device).requires_grad_(False)
    expected = manifest['base507_fp16_sha256']
    decoded_hashes = {}
    for label, row in manifest['matrices'].items():
        payload, _ = read_axis_e8(directory / row['file'], expected_shape=row['shape'])
        decoded = codec.decode_e8(payload, book, device=device)
        decoded_hashes.update(_validate_tensors({row['source_key']: decoded},
            {row['source_key']: row['shape']}, expected))
        runtime._set_parameter(model, row['source_key'], decoded, device, torch.float16)
        del payload, decoded
    for label, row in manifest['vocabularies'].items():
        output = torch.empty(row['shape'], dtype=torch.float16, device=device)
        next_row = 0
        for first, rows in codec.iter_uniform(directory / row['file'], device=device, chunk_rows=256):
            if first != next_row or first + len(rows) > row['shape'][0]:
                raise ValueError(f'Invalid vocabulary row stream: {label}')
            output[first:first + len(rows)].copy_(rows)
            next_row += len(rows)
        if next_row != row['shape'][0]:
            raise ValueError(f'Incomplete vocabulary row stream: {label}')
        decoded_hashes.update(_validate_tensors({row['source_key']: output},
            {row['source_key']: row['shape']}, expected))
        runtime._set_parameter(model, row['source_key'], output, device, torch.float16)
        del output, rows
    small = torch.load(directory / 'other_fp16.pt', map_location='cpu', weights_only=True)
    _validate_tensors(small, small_shapes(), expected)
    for name, value in small.items():
        runtime._set_parameter(model, name, value, device, torch.float16)
    params = dict(model.named_parameters())
    actual = _validate_tensors(params, base_shapes(), expected)
    if sum(p.numel() for p in params.values()) != BASE_PARAMETERS:
        raise ValueError('Materialized parameter count differs')
    if model.backbone.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise ValueError('Required independent vocabularies became tied')
    model.eval().requires_grad_(False)
    bank = native.install_fp16(model, directory / 'adapter_fp16.pt',
        expected_binding=manifest['adapter']['binding'], expected_base_hashes=actual, production=True)
    try:
        adapter_hashes = _validate_tensors(bank.masters, adapter_shapes(), manifest['adapter']['tensor_sha256'])
        if bank.gate_mode != 'soft' or any(p.requires_grad for p in bank.parameters()):
            raise ValueError('Release adapter is not a frozen soft inference adapter')
        bank.assert_base_frozen()
    except Exception:
        bank.close()
        raise
    # Plain bank object remains external to nn.Module registration and base507.
    model._release_resurface_adapter = bank
    model._package_receipt = {**receipt, 'actual507_fp16_sha256': actual,
        'actual224_adapter_fp16_sha256': adapter_hashes, 'base_parameter_count': BASE_PARAMETERS,
        'adapter_parameter_count': ADAPTER_PARAMETERS, 'soft_adapter_enabled': True,
        'gate_mode': 'soft', 'variant': native.VARIANT,
        'decoded_base_weight_bytes': 2 * BASE_PARAMETERS,
        'decoded_adapter_weight_bytes': 2 * ADAPTER_PARAMETERS,
        'weight_execution': 'fully decoded FP16 reference; not compressed model residency'}
    del small, params, book
    gc.collect()
    return model


def audit_loaded_model(model, check_values=True):
    bank = getattr(model, '_release_resurface_adapter', None)
    if bank is None or bank.gate_mode != 'soft':
        raise ValueError('Mandatory release soft adapter is not installed')
    receipt = model._package_receipt
    base = bank.assert_base_frozen(check_values=check_values)
    adapters = _validate_tensors(bank.masters, adapter_shapes(), receipt['actual224_adapter_fp16_sha256'])
    return {'base': base, 'adapter224_fp16_sha256': adapters, 'soft_adapter_enabled': True}


def load_tokenizer(directory):
    return runtime.SentencePieceTokenizer(Path(directory) / runtime.TOKENIZER_FILENAME)


def close_model(model):
    bank = getattr(model, '_release_resurface_adapter', None)
    if bank is not None:
        bank.close()
        del model._release_resurface_adapter
