"""Independent CPU check of shared prototype gradients through actualFP16 casts."""
import unittest

import torch

from mamba_e8w5.codec import load_reference_primitives
from mamba_e8w5.learned_e8_codebook import corrected_grid


class SharedPrototypeGradientTest(unittest.TestCase):
    def test_repeated_codes_and_shared_prototypes_reduce_signed_gradients(self):
        torch.set_num_threads(2)
        cb, _ = load_reference_primitives()
        codes = [0xAD91, 0xAD91, 0xAD12, 0xAD13, 0xADFF, 0x1201, 0xAD91, 0x1200]
        coefficients = ((torch.arange(len(codes)*8).reshape(len(codes), 8)%9-4)/8).float()
        expected = torch.zeros(256, 8, dtype=torch.float32)
        reorder = [0, 4, 1, 5, 2, 6, 3, 7]
        for occurrence, code in enumerate(codes):
            prototype, bits = code >> 8, code & 255
            parity = sum((bits >> bit)&1 for bit in range(8))%2
            bits ^= parity
            packed = int(cb.grid_packed_abs[prototype])
            for coordinate, position in enumerate(reorder):
                packed_sign = -1 if (((packed >> (4*position))&15)-8) < 0 else 1
                bit_sign = -1 if (bits >> position)&1 else 1
                expected[prototype, coordinate] += coefficients[occurrence, coordinate]*packed_sign*bit_sign
        # Actual backward traverses the FP16 table cast, then FP32 master cast.
        expected = expected.half().float()
        for initial in (torch.zeros(256, 8),
                        ((torch.arange(2048).reshape(256, 8)%1023-511)/4096).float()):
            master = initial.clone().requires_grad_(True)
            grid = corrected_grid(cb, master.half(), device='cpu')
            (grid[torch.tensor(codes)]*coefficients).sum().backward()
            self.assertEqual(master.grad.dtype, torch.float32)
            self.assertTrue(torch.equal(master.grad, expected))
            self.assertEqual(int((master.grad.abs().sum(1) > 0).sum()), 2)
            self.assertTrue(torch.count_nonzero(expected))


if __name__ == '__main__':
    unittest.main()
