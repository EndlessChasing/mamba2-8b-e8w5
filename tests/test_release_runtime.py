"""CPU format boundaries; no 8B model allocation or native GPU claims."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest

import torch

from mamba_e8w5 import release_runtime as r


def metadata():
    """Independent production geometry with placeholder hashes, no weight files."""
    digest = 'a' * 64
    small = {}
    matrices = {}
    for index in range(56):
        prefix = f'backbone.layers.{index}.'
        for suffix in ('norm.weight', 'mixer.norm.weight', 'mixer.dt_bias', 'mixer.A_log',
                       'mixer.D', 'mixer.conv1d.weight', 'mixer.conv1d.bias'):
            small[prefix + suffix] = digest
        for part, shape in [('in_proj', [18560, 4096]), ('out_proj', [4096, 8192])]:
            label = f'layer{index}.{part}'
            name = prefix + f'mixer.{part}.weight'
            matrices[label] = dict(file=label + '.e8', source_key=name, shape=shape,
                                   index_bits=20, decoded_fp16_sha256=digest)
            small[name] = digest
    small.update({'backbone.norm_f.weight': digest, 'backbone.embedding.weight': digest,
                  'lm_head.weight': digest})
    vocabularies = {
        'embedding': dict(file='embedding.uniform', source_key='backbone.embedding.weight',
                          shape=[256000, 4096], bits=4, decoded_fp16_sha256=digest),
        'lm_head': dict(file='lm_head.uniform', source_key='lm_head.weight',
                       shape=[256000, 4096], bits=5, decoded_fp16_sha256=digest)}
    names = {row['file'] for row in matrices.values()} | {'embedding.uniform', 'lm_head.uniform',
        'other_fp16.pt', 'config.json', 'e8_codebook.bin', 'adapter_fp16.pt', r.runtime.TOKENIZER_FILENAME}
    files = {name: {'bytes': 1, 'sha256': digest} for name in names}
    files['e8_codebook.bin']['bytes'] = 1024
    files[r.runtime.TOKENIZER_FILENAME]['sha256'] = r.runtime.TOKENIZER_SHA256
    adapter = dict(file='adapter_fp16.pt', bytes=1, sha256=digest, gate_mode='soft',
        variant=r.native.VARIANT, geometry=[dict(width=4096, heads=128, head_dim=64)] * 56,
        binding={}, parameters=1154104, payload_bytes=2308208,
        tensor_sha256={f'layer{i}.{field}': digest for i in range(56)
                      for field in ('V_read', 'g_read', 'router_w', 'router_b')})
    return dict(format=r.FORMAT, complete=True, model_config=copy.deepcopy(r.runtime.MODEL_CONFIG),
        total_parameter_count=8236999680, source_checkpoint_sha256=r.runtime.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=r.runtime.TOKENIZER_SHA256, files=files, matrices=matrices,
        vocabularies=vocabularies, base507_fp16_sha256=small, adapter=adapter,
        binding={'code_sha256': {name: digest for name in r.REQUIRED_CODE}})


class ReleaseRuntimeTests(unittest.TestCase):
    def test_independent_geometry_and_counts(self):
        m = metadata()
        r.validate_manifest(m)
        self.assertEqual(len(m['files']), 119)
        self.assertEqual(len(m['base507_fp16_sha256']), 507)
        self.assertEqual(sum(torch.tensor(v).prod().item() for v in r.base_shapes().values()), 8236999680)
        self.assertEqual(sum(__import__('math').prod(v) for v in r.adapter_shapes().values()), 1154104)

    def test_missing_adapter_or_wrong_policy_fails_closed(self):
        for mutate in (lambda m: m['files'].pop('adapter_fp16.pt'),
                       lambda m: m['adapter'].update(gate_mode='hard'),
                       lambda m: m['adapter']['tensor_sha256'].pop('layer55.router_b'),
                       lambda m: m.update(complete=1)):
            m = metadata(); mutate(m)
            with self.assertRaises(ValueError): r.validate_manifest(m)

    def test_matrix_vocab_hash_geometry_and_source_boundaries(self):
        mutations = (
            lambda m: m['matrices']['layer0.in_proj'].update(shape=[18432, 4096]),
            lambda m: m['matrices']['layer55.out_proj'].update(index_bits=16),
            lambda m: m['vocabularies']['embedding'].update(bits=5),
            lambda m: m['vocabularies']['lm_head'].update(source_key='backbone.embedding.weight'),
            lambda m: m['base507_fp16_sha256'].update({'lm_head.weight': 'b' * 64}),
            lambda m: m.update(source_checkpoint_sha256='0' * 64),
            lambda m: m['binding']['code_sha256'].pop('mamba_e8w5/resurface_native.py'),
            lambda m: m['files']['other_fp16.pt'].update(bytes=1 << 30),
        )
        for mutate in mutations:
            m = metadata(); mutate(m)
            with self.assertRaises(ValueError): r.validate_manifest(m)

    def test_safe_files_duplicate_json_and_trusted_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ok').write_text('safe')
            self.assertEqual(r.safe_file(root, 'ok'), root / 'ok')
            (root / 'link').symlink_to(root / 'ok')
            for name in ('../ok', './ok', '/ok', 'a\\b', 'link', 'a//b', 'C:ok'):
                with self.assertRaises(ValueError): r.safe_file(root, name)
            (root / 'manifest.json').write_text('{"complete":true,"complete":false}')
            with self.assertRaises(ValueError): r._json(root / 'manifest.json')
            with self.assertRaisesRegex(ValueError, 'Trusted raw manifest'):
                r.verify_package(root, 'f' * 64)

    def test_fp16_raw_bits_shape_dtype_and_finiteness(self):
        value = torch.tensor([0., -0., 1., -2.], dtype=torch.float16)
        # IEEE754 little-endian wire oracle independent of tensor_hash.
        expected = hashlib.sha256(bytes.fromhex('00000080003c00c0')).hexdigest()
        self.assertEqual(r.tensor_hash(value), expected)
        r._validate_tensors({'x': value}, {'x': [4]}, {'x': expected})
        for wrong in (value.float(), value.reshape(2, 2), torch.tensor([0., 0., 1., -2.], dtype=torch.float16),
                      torch.tensor([0., -0., 1., float('nan')], dtype=torch.float16)):
            with self.assertRaises(ValueError): r._validate_tensors({'x': wrong}, {'x': [4]}, {'x': expected})

    def test_actual_adapter_file_boundary(self):
        # Real production224 tensor payload (~2.3MB), no native model needed.
        tensors = {name: torch.zeros(shape, dtype=torch.float16) for name, shape in r.adapter_shapes().items()}
        payload = dict(format=r.native.FORMAT, variant=r.native.VARIANT, gate_mode='soft',
            geometry=[dict(width=4096, heads=128, head_dim=64)] * 56,
            binding={'opaque_prior_path': '/not/opened/model_optim_rng.pt'}, tensors=tensors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter_fp16.pt'
            torch.save(payload, path)
            loaded = r.native.read_fp16(path, expected_binding=payload['binding'])
            hashes = {k: r.tensor_hash(v) for k, v in tensors.items()}
            self.assertEqual(r._validate_tensors(loaded['tensors'], r.adapter_shapes(), hashes), hashes)
            loaded['tensors']['layer55.router_b'] = torch.ones((), dtype=torch.float16)
            with self.assertRaises(ValueError):
                r._validate_tensors(loaded['tensors'], r.adapter_shapes(), hashes)


if __name__ == '__main__':
    unittest.main()
