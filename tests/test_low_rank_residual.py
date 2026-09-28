"""Independent tiny CPU factor-layout, arithmetic and gradient proofs."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mamba_e8w5.low_rank_residual import (
    factor_layout, merge_weight, read_factors, validate_factors, write_factors,
)


def bits(tensor):
    return tensor.detach().contiguous().numpy().tobytes()


class LowRankResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def factors(self):
        B = torch.tensor([[1., -0.], [-2., .5]], dtype=torch.float16)
        A = torch.tensor([[.25, -.5, 2.], [-1., 4., -8.]], dtype=torch.float16)
        return B, A

    def independent_file(self):
        # Literal specification; no implementation header/layout helper used.
        return (struct.pack('<8sHHIIIQ', b'ME8LR001', 1, 1, 2, 3, 2, 20)
            + struct.pack('<10e', 1., -0., -2., .5, .25, -.5, 2., -1., 4., -8.))

    def test_handmade_wire_format_and_deterministic_roundtrip(self):
        B, A = self.factors()
        raw = self.independent_file()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'handmade.lrf'
            path.write_bytes(raw)
            rb, ra = read_factors(path, expected_shape=(2, 3), expected_rank=2)
            self.assertEqual(bits(rb), bits(B))
            self.assertEqual(bits(ra), bits(A))
            for name, b, a in [('first', B, A),
                    ('strided', B.T.contiguous().T, A.T.contiguous().T)]:
                target = Path(directory) / name
                receipt = write_factors(target, b, a)
                self.assertEqual(target.read_bytes(), raw)
                self.assertEqual(receipt['bytes'], 52)
                self.assertEqual(receipt['header_bytes'], 32)
                self.assertTrue(receipt['disk_roundtrip_bitwise_equal'])
                self.assertFalse(target.with_name(target.name + '.partial').exists())
                with self.assertRaises(FileExistsError):
                    write_factors(target, b, a)
                self.assertEqual(target.read_bytes(), raw)
            rb[0, 0] = 8
            self.assertEqual(bits(ra), bits(A))
            self.assertEqual(path.read_bytes(), raw)

    def test_header_and_length_rejections_before_array_allocation(self):
        raw = self.independent_file()
        header = [b'ME8LR001', 1, 1, 2, 3, 2, 20]
        cases = [raw[:31], raw[:-1], raw + b'\0']
        for index, replacement in [(0, b'BADMAGIC'), (1, 2), (2, 2),
                (3, 0), (4, 0), (5, 0), (5, 3), (6, 18),
                (3, 65537), (3, 50000), (4, 50000), (5, 257)]:
            altered = list(header); altered[index] = replacement
            cases.append(struct.pack('<8sHHIIIQ', *altered) + raw[32:])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.lrf'
            for candidate in cases:
                with self.subTest(header=candidate[:32]):
                    path.write_bytes(candidate)
                    with patch('mamba_e8w5.low_rank_residual.np.frombuffer',
                            side_effect=AssertionError('Must reject before payload allocation')):
                        with self.assertRaises(ValueError):
                            read_factors(path)
            path.write_bytes(raw)
            for kwargs in [dict(expected_shape=(3, 2)), dict(expected_shape=(True, 3)),
                    dict(expected_rank=1), dict(expected_rank=2.), dict(expected_rank=True)]:
                with self.assertRaises(ValueError):
                    read_factors(path, **kwargs)

    def test_nonfinite_file_payloads_rejected_in_both_factors(self):
        original = self.independent_file()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nonfinite.lrf'
            for offset in (32, 40):
                for value in (0x7c00, 0xfc00, 0x7e00):
                    raw = bytearray(original)
                    raw[offset:offset + 2] = struct.pack('<H', value)
                    path.write_bytes(raw)
                    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                        read_factors(path)

    def test_strict_tensor_and_resource_bounds(self):
        B, A = self.factors()
        with self.assertRaises(TypeError):
            validate_factors([[1]], A)
        for b, a in [(B.float(), A), (B, A.float()), (B.flatten(), A),
                (B, A[:1]), (B.to('meta'), A), (B[:0], A)]:
            with self.assertRaises(ValueError):
                validate_factors(b, a)
        for value in (float('inf'), float('-inf'), float('nan')):
            b = B.clone(); b[0, 0] = value
            with self.assertRaises(ValueError):
                validate_factors(b, A)
        for shape in [(True, 3, 1), (2, 3, 1.), (65537, 2, 1),
                (20000, 20000, 1), (1024, 1024, 257), (2, 3, 3),
                (65536, 1024, 256)]:  # Factor bytes exceed32MiB, no allocation.
            with self.assertRaises(ValueError):
                factor_layout(*shape)
        self.assertEqual(factor_layout(18560, 4096, 16)['payload_bytes'], 724992)
        self.assertEqual(factor_layout(4096, 8192, 8)['payload_bytes'], 196608)

    def test_preexisting_partial_and_destination_are_preserved(self):
        B, A = self.factors()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'factors.lrf'
            partial = path.with_name(path.name + '.partial')
            partial.write_bytes(b'old failed receipt')
            with self.assertRaises(FileExistsError):
                write_factors(path, B, A)
            self.assertEqual(partial.read_bytes(), b'old failed receipt')
            self.assertFalse(path.exists())
            path.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                write_factors(path, B, A)
            self.assertEqual(path.read_bytes(), b'existing')

    def test_zero_merge_preserves_values_without_touching_base(self):
        base = torch.tensor([[1., -2., 0.], [2**-24, 65504., -4.]], dtype=torch.float16)
        original = bits(base); version = base._version
        B = torch.zeros(2, 2, dtype=torch.float16)
        _, A = self.factors()
        merged = merge_weight(base, B, A)
        self.assertEqual(bits(merged), original)
        self.assertEqual(bits(base), original)
        self.assertEqual(base._version, version)
        self.assertNotEqual(merged.data_ptr(), base.data_ptr())

    def test_signed_zero_follows_native_addition_not_a_zero_shortcut(self):
        base = torch.tensor([[-0.]], dtype=torch.float16)
        B = torch.zeros(1, 1, dtype=torch.float16)
        A = torch.ones(1, 1, dtype=torch.float16)
        result = merge_weight(base, B, A)
        self.assertEqual(bits(base), struct.pack('<H', 0x8000))
        self.assertEqual(bits(result), struct.pack('<H', 0x0000))

    def test_nonzero_merge_matches_independent_dyadic_arithmetic(self):
        base = torch.tensor([[1., 2., 3.], [-1., 0., .5]], dtype=torch.float16)
        B = torch.tensor([[.5, -.25], [1., 2.]], dtype=torch.float16)
        A = torch.tensor([[2., -1., .5], [.5, 1., -2.]], dtype=torch.float16)
        # Hand-computed dot products are exact dyadic rationals, including the
        # order B@A. This oracle does not call the implementation's matmul.
        expected = torch.tensor([[1.875, 1.25, 3.75], [2., 1., -3.]], dtype=torch.float16)
        before = tuple(bits(t) for t in (base, B, A))
        self.assertEqual(bits(merge_weight(base, B, A)), bits(expected))
        self.assertEqual(tuple(bits(t) for t in (base, B, A)), before)

    def test_zero_initialization_retains_correct_factor_gradient(self):
        base = torch.ones(2, 3, dtype=torch.float16, requires_grad=True)
        b_master = torch.zeros(2, 2, dtype=torch.float32, requires_grad=True)
        a_master = torch.tensor([[1., .5, -1.], [.25, -.5, 1.]], requires_grad=True)
        coefficients = torch.tensor([[1., 2., 4.], [2., 1., 3.]])
        loss = (merge_weight(base, b_master.half(), a_master.half()).float() * coefficients).sum()
        loss.backward()
        expected_b = torch.tensor([[-2., 3.25], [-.5, 3.]])
        self.assertTrue(torch.equal(b_master.grad, expected_b))
        self.assertTrue(torch.equal(a_master.grad, torch.zeros_like(a_master)))
        self.assertIsNone(base.grad)

    def test_checkpoint_replay_agrees_with_tiny_direct_forward_and_gradients(self):
        base = torch.tensor([[.25, -.5, 1.], [1., .75, -.25]], dtype=torch.float16, requires_grad=True)
        before = bits(base); version = base._version
        b_initial = torch.tensor([[.10001, -.07003], [.04007, .09001]])
        a_initial = torch.tensor([[.31001, -.29003, .41007], [.22009, .17003, -.13007]])
        inputs = torch.tensor([[1., -.5, .25], [-.25, .75, 1.]], dtype=torch.float16)
        results = []
        for use_checkpoint in (False, True):
            b = b_initial.clone().requires_grad_(True)
            a = a_initial.clone().requires_grad_(True)
            x = inputs.clone().requires_grad_(True)
            def block(x, b, a):
                # Both master casts belong inside the replayed function.
                return F.linear(x, merge_weight(base, b.half(), a.half()))
            output = checkpoint(block, x, b, a, use_reentrant=False) if use_checkpoint else block(x, b, a)
            (output.float().square().sum() + output.float().sum() / 8).backward()
            self.assertTrue(torch.isfinite(output).all())
            for gradient in (b.grad, a.grad, x.grad):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(torch.count_nonzero(gradient).item(), 0)
            self.assertEqual(b.grad.dtype, torch.float32)
            self.assertEqual(a.grad.dtype, torch.float32)
            results.append(tuple(bits(t) for t in (output, b.grad, a.grad, x.grad)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(bits(base), before)
        self.assertEqual(base._version, version)
        self.assertIsNone(base.grad)

    def test_merge_rejects_wrong_base_overflow_and_cpu_autocast(self):
        B, A = self.factors()
        base = torch.zeros(2, 3, dtype=torch.float16)
        for bad in (base.float(), base[:, :2], torch.full_like(base, float('nan'))):
            with self.assertRaises(ValueError):
                merge_weight(bad, B, A)
        with self.assertRaisesRegex(ValueError, 'overflows'):
            merge_weight(torch.zeros(1, 1, dtype=torch.float16),
                torch.full((1, 1), 65504., dtype=torch.float16),
                torch.full((1, 1), 65504., dtype=torch.float16))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'autocast'):
                merge_weight(base, B, A)


if __name__ == '__main__':
    unittest.main()
