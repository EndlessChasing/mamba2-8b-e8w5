"""CPU-only exact sign/layout, inverse decode and prototype-file proofs."""
import copy
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mamba_e8w5.codec import load_reference_primitives, decode_e8
from mamba_e8w5.learned_e8_codebook import (corrected_grid, corrected_grid_from_rounded_fp32,
    prototype_mapping, decode_e8_with_prototypes, write_prototypes, read_prototypes,
    HEADER, EXPECTED_HEADER, FILE_BYTES, PAYLOAD_BYTES)


def bitwise_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes()


def scalar_packed_oracle(packed, delta):
    """Independent scalar packed decode, following pinned upstream get_full_grid.

    This deliberately does not call the implementation's mapping, grid helper,
    sign table or SHUFFLE constant. The perturbation is inserted into the packed
    absolute magnitude before upstream sign flips and the quarter offset.
    """
    output = np.empty((65536, 8), dtype=np.float32)
    packed = [int(value) for value in packed]
    perturbation = delta.numpy().astype(np.float32)
    reorder = [0, 4, 1, 5, 2, 6, 3, 7]
    for code in range(65536):
        signs = code & 255
        prototype = code >> 8
        parity = 0
        for bit in range(8):
            parity ^= (signs >> bit) & 1
        signs ^= parity
        for coordinate in range(8):
            location = reorder[coordinate]
            value = (((packed[prototype] >> (4*location)) & 15)-8)*.5
            correction = float(perturbation[prototype, coordinate])
            value = value + (correction if value > 0 else -correction)
            if (signs >> location) & 1:
                value = -value
            output[code, coordinate] = value + (-.25 if parity else .25)
    return torch.from_numpy(output)


class LearnedE8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cb, _ = load_reference_primitives()
        cls.zero = torch.zeros((256, 8), dtype=torch.float16)
        # Distinct dyadic values make prototype/coordinate swaps observable.
        cls.delta = ((torch.arange(2048).reshape(256, 8) % 1023 - 511)/4096).half()
        cls.oracle = scalar_packed_oracle(cls.cb.grid_packed_abs.tolist(), cls.delta)

    def payload(self):
        generator = torch.Generator().manual_seed(349)
        # Non-power-of-two dimensions exercise both DCT and Hadamard factors.
        return {'indices': torch.randint(0, 65536, (20, 3), generator=generator, dtype=torch.int32),
            'balance': (.5+torch.rand(24, generator=generator)).half(),
            'input_sign': (2*torch.randint(0, 2, (24,), generator=generator)-1).to(torch.int8),
            'output_sign': (2*torch.randint(0, 2, (20,), generator=generator)-1).to(torch.int8),
            'scale': torch.tensor(.37, dtype=torch.float32), 'axis_residual_amplitude': 0.0}

    def test_every_zero_codeword_is_exact_pinned_upstream(self):
        result = corrected_grid(self.cb, self.zero, device='cpu')
        self.assertTrue(bitwise_equal(result, self.cb.grid))
        self.assertEqual(result.numel(), 65536*8)
        self.assertTrue(bitwise_equal(scalar_packed_oracle(self.cb.grid_packed_abs.tolist(), self.zero), self.cb.grid))
        negative_zero = torch.full((256, 8), -0., dtype=torch.float16)
        self.assertTrue(bitwise_equal(corrected_grid(self.cb, negative_zero), self.cb.grid))

    def test_all_nonzero_codes_match_independent_packed_oracle(self):
        self.assertTrue(bitwise_equal(corrected_grid(self.cb, self.delta), self.oracle))
        prototypes, directions = prototype_mapping(self.cb)
        self.assertTrue(torch.equal(prototypes, torch.arange(65536)//256))
        self.assertTrue(torch.equal(directions.abs(), torch.ones_like(directions)))
        # Deliberately chosen odd prototype family includes the packed base's
        # signed last coordinate, so the proof detects omission of that sign.
        signed_packed = [int(value) for value in self.cb.grid_packed_abs]
        self.assertTrue(any(((value >> 28)&15) < 8 for value in signed_packed))

    def test_decode_reuses_frozen_inverse_and_fp16_rounding(self):
        payload = self.payload()
        zero = decode_e8_with_prototypes(payload, self.cb, self.zero, device='cpu')
        self.assertTrue(bitwise_equal(zero, decode_e8(payload, self.cb, device='cpu')))
        result = decode_e8_with_prototypes(payload, self.cb, self.delta, device='cpu')
        reference = decode_e8(payload, SimpleNamespace(grid=self.oracle), device='cpu')
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(bitwise_equal(result, reference))
        self.assertFalse(bitwise_equal(result, zero))

    def test_float32_adjoint_keeps_exact_forward_at_zero_and_nonzero(self):
        for stored in (self.zero, self.delta):
            differentiable = stored.float().requires_grad_(True)
            result = corrected_grid_from_rounded_fp32(self.cb, differentiable)
            self.assertTrue(bitwise_equal(result.detach(), corrected_grid(self.cb, stored)))
            # Single-code adjoint checks prototype/sign/coordinate scatter
            # without averaging away the symmetric sign orbit.
            code = 0xAD91
            coefficients = torch.arange(1., 9.)
            (result[code]*coefficients).sum().backward()
            _, directions = prototype_mapping(self.cb)
            expected = torch.zeros((256, 8), dtype=torch.float32)
            expected[code >> 8] = directions[code]*coefficients
            self.assertEqual(differentiable.grad.dtype, torch.float32)
            self.assertTrue(torch.equal(differentiable.grad, expected))
        bad = self.zero.float(); bad[0, 0] = .00010001
        with self.assertRaises(ValueError):
            corrected_grid_from_rounded_fp32(self.cb, bad)

    def test_invalid_table_and_mutated_base_rejected(self):
        for delta in (torch.zeros(256, 8), self.zero[:-1], self.zero[:, :-1], self.zero.flatten()):
            with self.assertRaises(ValueError):
                corrected_grid(self.cb, delta)
        for value in (float('nan'), float('inf'), -float('inf')):
            delta = self.zero.clone(); delta[7, 2] = value
            with self.assertRaises(ValueError):
                corrected_grid(self.cb, delta)
        with self.assertRaises(TypeError):
            corrected_grid(self.cb, [[0.]])
        book = SimpleNamespace(grid=self.cb.grid.clone(), grid_packed_abs=self.cb.grid_packed_abs.clone())
        book.grid[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Expanded base grid'):
            corrected_grid(book, self.zero)
        book.grid = self.cb.grid
        book.grid_packed_abs[0] ^= 1
        with self.assertRaisesRegex(ValueError, 'Packed base codebook'):
            corrected_grid(book, self.zero)

    def test_invalid_payload_rejected(self):
        payload = self.payload()
        cases = [('indices', torch.full((20, 3), 65536, dtype=torch.int32)),
            ('indices', torch.full((20, 3), -1, dtype=torch.int32)),
            ('indices', torch.ones((20, 3), dtype=torch.float32)),
            ('indices', torch.empty((0, 3), dtype=torch.int32)),
            ('balance', torch.zeros(24, dtype=torch.float16)),
            ('input_sign', torch.zeros(24, dtype=torch.int8)),
            ('output_sign', torch.ones(19, dtype=torch.int8)),
            ('scale', torch.tensor(float('nan'))), ('scale', torch.tensor(-1.)),
            ('axis_residual_amplitude', .1)]
        for key, value in cases:
            with self.subTest(key=key):
                bad = dict(payload); bad[key] = value
                with self.assertRaises(ValueError):
                    decode_e8_with_prototypes(bad, self.cb, self.zero, device='cpu')

    def test_fixed_size_serialization_preserves_every_fp16_bit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'correction.e8p'
            delta = self.delta.clone(); delta[0, 0] = -0.0
            result = write_prototypes(path, delta)
            self.assertEqual(FILE_BYTES, 4128)
            self.assertEqual(PAYLOAD_BYTES, 4096)
            self.assertEqual(result['header_bytes'], 32)
            self.assertEqual(path.stat().st_size, 4128)
            self.assertTrue(result['disk_roundtrip_bitwise_equal'])
            self.assertTrue(bitwise_equal(read_prototypes(path), delta))
            self.assertFalse(path.with_name(path.name+'.partial').exists())
            with self.assertRaises(FileExistsError):
                write_prototypes(path, delta)
            raw = path.read_bytes()
            for value in (raw[:-1], raw+b'\0'):
                path.write_bytes(value)
                with self.assertRaises(ValueError):
                    read_prototypes(path)
            for field in range(7):
                header = list(EXPECTED_HEADER)
                header[field] = b'INVALID!' if field == 0 else header[field]+1
                path.write_bytes(HEADER.pack(*header)+raw[32:])
                with self.assertRaises(ValueError):
                    read_prototypes(path)
            damaged = bytearray(raw)
            damaged[32:34] = struct.pack('<H', 0x7c00)  # FP16 infinity.
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                read_prototypes(path)


if __name__ == '__main__':
    unittest.main()
