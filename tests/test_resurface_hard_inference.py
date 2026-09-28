"""CPU-only checks for the prepared fixed-hard0 inference fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from mamba_e8w5 import resurface_native as native
from mamba_e8w5 import resurface_hard_inference as hard


def parameters():
    return {'V_read': torch.tensor([[2., 0.], [0., 3.]], dtype=torch.float16),
            'g_read': torch.tensor([.5, 2.], dtype=torch.float16),
            'router_w': torch.tensor([1., 0.], dtype=torch.float16),
            'router_b': torch.tensor(0., dtype=torch.float16), 'heads': 2, 'head_dim': 1}


class Norm(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(2, dtype=torch.float16)); self.seen = None
    def forward(self, x, z=None):
        self.seen = x
        return x


class Mixer(nn.Module):
    d_model, d_ssm, nheads, headdim, ngroups = 2, 2, 2, 1, 1
    rmsnorm, use_mem_eff_path = True, False
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(2, 2, bias=False, dtype=torch.float16)
        with torch.no_grad(): self.in_proj.weight.copy_(torch.eye(2, dtype=torch.float16))
        self.norm = Norm(); self.before_norm = None
    def forward(self, u, inference_params=None):
        step = inference_params is not None
        y = self.in_proj(u[:, 0] if step else u)
        self.before_norm = y
        result = self.norm(y, None)
        return result.unsqueeze(1) if step else result


class Block(nn.Module):
    fused_add_norm, residual_in_fp32, mlp = False, False, None
    def __init__(self): super().__init__(); self.mixer = Mixer()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.fused_add_norm = self.backbone.residual_in_fp32 = False
        self.backbone.layers = nn.ModuleList([Block()])
    def forward(self, value, inference_params=None):
        return self.backbone.layers[0].mixer(value, inference_params=inference_params)


def fixture(directory):
    model = Model().eval().requires_grad_(False)
    hashes = {name: native.tensor_hash(p) for name, p in model.named_parameters()}
    path = Path(directory)/'soft_adapter_fp16.pt'
    with native.ResurfaceNative(model, 'soft', production=False, expected_base_hashes=hashes) as bank:
        with torch.no_grad():
            for name in native.FIELDS: getattr(bank.adapters['layer0'], name).copy_(parameters()[name])
        receipt = bank.export_fp16(path, binding={'fixture': 'unchanged-soft-export'})
    return model, hashes, path, receipt


class FixedHardTests(unittest.TestCase):
    def test_all_closed_original_object_signed_zero_and_stride(self):
        storage = torch.tensor([[-0., 17., 0., 19.], [2., 21., -0., 23.]], dtype=torch.float16)
        y = storage[:, ::2]
        u = torch.tensor([[-1., 0.], [0., 0.]], dtype=torch.float16)
        before = native.tensor_hash(y)
        result, stats = hard.hard0_readout(y, u, **parameters())
        self.assertIs(result, y)
        self.assertEqual(result.data_ptr(), y.data_ptr()); self.assertEqual(result.stride(), y.stride())
        self.assertEqual(native.tensor_hash(result), before)
        self.assertTrue(torch.signbit(result[0, 0])); self.assertTrue(torch.signbit(result[1, 1]))
        self.assertEqual((stats['opened'], stats['closed'], stats['zero_logits']), (0, 2, 1))

    def test_mixed_select_preserves_negative_zero_and_does_not_mix_closed_rows(self):
        y = torch.tensor([[[-0., 0.], [1., 2.], [65504., -0.]]], dtype=torch.float16)
        u = torch.tensor([[[-1., 0.], [1., 0.], [0., 0.]]], dtype=torch.float16)
        result, stats = hard.hard0_readout(y, u, **parameters())
        self.assertEqual(native.tensor_hash(result[:, [0, 2]]), native.tensor_hash(y[:, [0, 2]]))
        self.assertTrue(torch.equal(result[:, 1], torch.tensor([[2., 14.]], dtype=torch.float16)))
        self.assertTrue(torch.isfinite(result).all())  # Mixing the closed65504 row would overflow.
        self.assertEqual((stats['opened'], stats['closed'], stats['zero_logits']), (1, 2, 1))

    def test_positive_gate_is_one_without_a_scale(self):
        y = torch.tensor([[1., 2.], [-2., 3.]], dtype=torch.float16)
        u = torch.tensor([[.125, 0.], [2., 0.]], dtype=torch.float16)
        result, stats = hard.hard0_readout(y, u, **parameters())
        expected = torch.tensor([[2., 14.], [-4., 21.]], dtype=torch.float16)
        self.assertEqual(native.tensor_hash(result), native.tensor_hash(expected))
        self.assertTrue(stats['all_open']); self.assertEqual(stats['opened'], 2)

    def test_nonfinite_logits_fail_closed(self):
        y = torch.zeros(1, 2, dtype=torch.float16)
        for value in (float('inf'), -float('inf'), float('nan')):
            with self.assertRaises(FloatingPointError):
                hard.hard0_readout(y, torch.tensor([[value, 0.]], dtype=torch.float16), **parameters())

    def test_actual_fp16_file_reuse_native_hooks_statistics_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='hard0-cpu-') as directory:
            model, hashes, path, receipt = fixture(directory)
            identity = {name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()}
            value = torch.tensor([[[-1., 0.], [0., -0.], [1., 2.]]], dtype=torch.float16)
            with torch.no_grad(): baseline = model(value).clone()
            with hard.install_fp16(model, path, expected_sha256=receipt['sha256'],
                    expected_binding={'fixture': 'unchanged-soft-export'}, expected_base_hashes=hashes,
                    production=False) as bank:
                output = model(value)
                self.assertTrue(torch.equal(output[0, 2], torch.tensor([2., 14.], dtype=torch.float16)))
                closed = torch.tensor([[[-1., -0.]]], dtype=torch.float16)
                model(closed, inference_params={})  # Same hook route as native forward->step.
                mixer = model.backbone.layers[0].mixer
                self.assertIs(mixer.before_norm, mixer.norm.seen)
                stats = bank.validate_gate_coverage(calls_per_layer=2, tokens_per_layer=4)
                self.assertEqual((stats['total']['opened'], stats['total']['closed']), (1, 3))
                self.assertEqual((stats['total']['minimum_logit'], stats['total']['maximum_logit']), (-1., 1.))
                self.assertFalse(bank._frames); self.assertFalse(bank._requests)
                json.dumps(stats, allow_nan=False)  # Only scalars/dicts, no retained tensor log.
                proof = bank.assert_adapter_unchanged()
                self.assertEqual(proof['fp16_tensor_sha256'], receipt['tensor_sha256'])
                bank.assert_base_frozen(check_values=True)
                with self.assertRaises(RuntimeError): bank.load_state_dict(bank.state_dict())
                with self.assertRaises(RuntimeError): bank.export_fp16(Path(directory)/'forbidden.pt')
                old = bank.reset_gate_stats()
                self.assertEqual(old, stats); self.assertIsNone(bank.gate_stats()['total']['all_observed_gates_closed'])
                with self.assertRaises(RuntimeError): bank.validate_gate_coverage(calls_per_layer=1, tokens_per_layer=1)
            self.assertEqual(native.file_hash(path), receipt['sha256'])
            self.assertEqual(identity, {name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()})
            self.assertEqual(native.tensor_hash(model(value)), native.tensor_hash(baseline))

    def test_failed_forward_cleans_hook_state_and_bad_file_identity_rejected(self):
        with tempfile.TemporaryDirectory(prefix='hard0-cpu-') as directory:
            model, hashes, path, receipt = fixture(directory)
            with self.assertRaises(ValueError): hard.install_fp16(model, path, expected_sha256='0'*64, production=False)
            with hard.install_fp16(model, path, expected_base_hashes=hashes, production=False) as bank:
                with self.assertRaises(FloatingPointError): model(torch.tensor([[[float('inf'), 0.]]], dtype=torch.float16))
                self.assertFalse(bank._frames); self.assertFalse(bank._requests)
                self.assertEqual(bank.gate_stats()['total']['failed_calls'], 1)
                bank.assert_base_frozen(check_values=True)
            self.assertFalse(model.backbone.layers[0].mixer._forward_pre_hooks)
            self.assertFalse(model.backbone.layers[0].mixer.norm._forward_pre_hooks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('Run with CUDA_VISIBLE_DEVICES empty; no CUDA initialization allowed')
    if args.report is not None and args.report.exists(): raise FileExistsError(args.report)
    torch.set_num_threads(8)
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixedHardTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'complete': result.wasSuccessful(), 'scope': 'CPU toy implementation checks only; no model-quality or GPU result',
        'command': 'CUDA_VISIBLE_DEVICES= /home/horde/.venvs/lodram/bin/python tests/test_resurface_hard_inference.py'+
                   (f' --report {args.report}' if args.report is not None else ''),
        'exit_code': 0 if result.wasSuccessful() else 1, 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'cuda_initialized': torch.cuda.is_initialized(),
        'elapsed_seconds': time.monotonic()-started, 'policy': hard.POLICY,
        'source_sha256': native.file_hash(Path(hard.__file__)), 'test_sha256': native.file_hash(__file__),
        'frozen_native_sha256': native.file_hash(Path(native.__file__)),
        'failure_details': [text for _, text in result.failures+result.errors]}
    if args.report is not None:
        with args.report.open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt, indent=2))
    raise SystemExit(receipt['exit_code'])
