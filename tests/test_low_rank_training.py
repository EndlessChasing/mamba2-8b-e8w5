"""CPU toy proofs for native rank-4 merging, gradients, replay and export."""
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from mamba_e8w5.low_rank_residual import merge_weight, read_factors, write_factors
from mamba_e8w5.low_rank_training import (
    LowRankMasters, _check_precision, chunked_loss_backward, export_parity,
    gradient_receipt, merge_fp16_weight, selected_inventory, training_schedule,
)


class RMS(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, dtype=torch.float16))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).half() * self.weight


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_mem_eff_path = False
        self.in_proj = nn.Linear(4, 8, bias=False, dtype=torch.float16)
        self.out_proj = nn.Linear(8, 4, bias=False, dtype=torch.float16)
        self.norm = RMS(8)

    def forward(self, x, inference_params=None):
        return self.out_proj(self.norm(torch.tanh(self.in_proj(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_add_norm = self.residual_in_fp32 = False
        self.mlp = None
        self.norm, self.mixer = RMS(4), Mixer()

    def forward(self, h, r=None, inference_params=None):
        r = h + r if r is not None else h
        return self.mixer(self.norm(r), inference_params), r


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_add_norm = self.residual_in_fp32 = False
        self.embedding = nn.Embedding(19, 4, dtype=torch.float16)
        self.layers = nn.ModuleList([Block(), Block()])
        self.norm_f = RMS(4)

    def forward(self, ids):
        h, r = self.embedding(ids), None
        for layer in self.layers:
            h, r = layer(h, r)
        return self.norm_f(h + r)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.lm_head = nn.Linear(4, 19, bias=False, dtype=torch.float16)


class Scale:
    def scale(self, value):
        return 16 * value


class LowRankTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(155)
        self.model = Model().eval()
        self.bank = LowRankMasters(self.model)
        self.ids = torch.arange(7)[None]

    def clear(self):
        for master in self.bank.parameters():
            master.grad = None

    def perturb(self):
        with torch.no_grad():
            for name, master in self.bank.masters.items():
                if name.endswith('.B'):
                    master.copy_(torch.linspace(-.0151, .0249, master.numel()).reshape_as(master))

    @staticmethod
    def objective(hidden):
        return (hidden.float() * torch.linspace(-.125, .25, hidden.numel()).reshape_as(hidden)).sum()

    def test_deterministic_init_rng_order_and_full_geometry(self):
        generator = torch.Generator(device='cpu').manual_seed(20260928)
        for label, entry in self.bank.inventory.items():
            n = entry['shape'][1]
            expected = torch.randn(4, n, generator=generator, dtype=torch.float32) * (1 / math.sqrt(n))
            self.assertTrue(torch.equal(expected, self.bank.masters[label + '.A']))
            self.assertEqual(int(torch.count_nonzero(self.bank.masters[label + '.B'])), 0)
            source = self.model.get_parameter(entry['source_key'])
            self.assertEqual(self.bank.base_weights[label].data_ptr(), source.data_ptr())
            self.assertIs(self.bank.frozen_lookup(entry['source_key']), source)
            self.assertFalse(self.bank.base_weights[label].requires_grad)
        before = torch.get_rng_state().clone()
        LowRankMasters(self.model)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        inventory = selected_inventory()
        self.assertEqual(len(inventory), 112)
        self.assertEqual(sum(v['value_count'] for v in inventory.values()), 7827456)
        self.assertEqual(sum(v['bytes'] for v in inventory.values()), 15658496)
        schedule = training_schedule()
        self.assertEqual(len(schedule), 1024)
        expected_generator = torch.Generator(device='cpu').manual_seed(20260928)
        expected = [n for _ in range(4) for n in torch.randperm(256, generator=expected_generator).tolist()]
        self.assertEqual(schedule, expected)

    def test_zero_B_gradients_and_both_families_after_one_update(self):
        before = self.bank.frozen_hashes()
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model.backbone(self.ids), self.bank.forward(self.ids)))
        self.objective(self.bank.forward(self.ids)).backward()
        receipt = gradient_receipt(self.bank)
        self.assertTrue(receipt['finite'])
        for name, row in receipt['per_tensor'].items():
            grad = self.bank.masters[name].grad
            self.assertEqual(row['minimum'], float(grad.min()))
            self.assertEqual(row['maximum'], float(grad.max()))
            self.assertEqual(row['maximum_absolute'], float(grad.abs().max()))
        for name, value in self.bank.masters.items():
            if name.endswith('.A'):
                self.assertEqual(int(torch.count_nonzero(value.grad)), 0, name)
            else:
                self.assertGreater(int(torch.count_nonzero(value.grad)), 0, name)
        torch.optim.SGD(self.bank.parameters(), lr=.01).step()
        self.clear()
        self.objective(self.bank.forward(self.ids)).backward()
        for name, value in self.bank.masters.items():
            self.assertGreater(int(torch.count_nonzero(value.grad)), 0, name)
        self.assertEqual(before, self.bank.frozen_hashes())

    def test_independent_native_output_and_factor_gradients_match_checkpoint_replay(self):
        self.perturb()
        independent = {name: value.detach().clone().requires_grad_() for name, value in self.bank.masters.items()}
        weights = dict(self.model.backbone.named_parameters())
        for label, entry in self.bank.inventory.items():
            base = self.bank.frozen_lookup(entry['source_key'])
            B, A = independent[label + '.B'].half(), independent[label + '.A'].half()
            # Separate whole-backbone functional execution, with no bank merge/forward call.
            weights[entry['source_key'].removeprefix('backbone.')] = (base.detach().float() + B.float() @ A.float()).half()
        reference = functional_call(self.model.backbone, weights, (self.ids,), tie_weights=False, strict=True)
        self.objective(reference).backward()
        for enabled in (False, True):
            self.clear()
            hidden = self.bank.forward(self.ids, use_checkpoint=enabled)
            self.objective(hidden).backward()
            self.assertTrue(torch.equal(reference, hidden))
            for name, master in self.bank.masters.items():
                self.assertTrue(torch.equal(independent[name].grad, master.grad), name)
        self.bank.assert_frozen()

    def test_single_block_checkpoint_only_local_factors_and_input_gradient(self):
        self.perturb()
        with torch.no_grad():
            h, r = self.model.backbone.layers[0](self.model.backbone.embedding(self.ids), None)
        runs = []
        for enabled in (False, True):
            self.clear()
            local = h.detach().requires_grad_()
            output, residual = self.bank.forward_block(1, local, r, use_checkpoint=enabled)
            self.objective(output + residual).backward()
            gradients = {n: p.grad.clone() for n, p in self.bank.masters.items() if p.grad is not None}
            self.assertEqual(set(gradients), {f'layer1.{part}.{factor}' for part in ('in_proj', 'out_proj') for factor in ('B', 'A')})
            runs.append((output.detach(), residual.detach(), local.grad.clone(), gradients))
        for left, right in zip(runs[0][:3], runs[1][:3]):
            self.assertTrue(torch.equal(left, right))
        for name in runs[0][3]:
            self.assertTrue(torch.equal(runs[0][3][name], runs[1][3][name]))

    def test_fp16_rounding_and_manual_gradient_of_native_merge(self):
        # 1.0003 rounds to FP16 1; four unrounded products instead round above 4.
        B = torch.full((4, 4), 1.0003, requires_grad=True)
        A = torch.full((4, 4), 1.0003, requires_grad=True)
        base = torch.zeros(4, 4, dtype=torch.float16, requires_grad=True)
        coefficients = torch.arange(16).reshape(4, 4).float() / 16
        actual = merge_fp16_weight(base, B.half(), A.half())
        expected = merge_weight(base, B.half(), A.half())
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, (base.detach().float() + B @ A).half()))
        (actual.float() * coefficients).sum().backward()
        expected_B = (coefficients.half().float() @ A.detach().half().float().T).half().float()
        expected_A = (B.detach().half().float().T @ coefficients.half().float()).half().float()
        self.assertTrue(torch.equal(B.grad, expected_B))
        self.assertTrue(torch.equal(A.grad, expected_A))
        self.assertIsNone(base.grad)

    def test_disk_factor_export_native_parity_and_reference_restoration(self):
        self.perturb()
        identities = {n: id(p) for n, p in self.model.named_parameters()}
        hashes = self.bank.frozen_hashes()
        state = self.bank.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            factors = self.bank.export_factors()
            restored = {}
            for label, (B, A) in factors.items():
                self.assertTrue(torch.equal(B, state[label + '.B'].half()))
                self.assertTrue(torch.equal(A, state[label + '.A'].half()))
                path = Path(directory) / (label + '.lr')
                write_factors(path, B, A)
                restored[label] = read_factors(path, expected_shape=self.bank.inventory[label]['shape'], expected_rank=4)
            receipt = export_parity(self.bank, restored, self.ids)
        self.assertTrue(receipt['bitwise_equal'])
        self.assertEqual(identities, {n: id(p) for n, p in self.model.named_parameters()})
        self.assertEqual(hashes, self.bank.frozen_hashes())
        # torch.equal treats +/-0 as equal; a bitwise export claim must reject it.
        with patch.object(self.bank, 'forward', return_value=torch.zeros(1, 7, 4, dtype=torch.float16)), \
                patch.object(self.model.backbone, 'forward', return_value=torch.full((1, 7, 4), -0.0, dtype=torch.float16)):
            with self.assertRaisesRegex(RuntimeError, 'forward mismatch'):
                export_parity(self.bank, restored, self.ids)
        self.assertEqual(identities, {n: id(p) for n, p in self.model.named_parameters()})
        self.assertEqual(hashes, self.bank.frozen_hashes())
        wrong = copy.deepcopy(restored)
        wrong['layer0.in_proj'][0].fill_(.25)
        with self.assertRaisesRegex(RuntimeError, 'forward mismatch'):
            export_parity(self.bank, wrong, self.ids)
        self.assertEqual(identities, {n: id(p) for n, p in self.model.named_parameters()})
        self.assertEqual(hashes, self.bank.frozen_hashes())

    def test_staged_full_vocabulary_loss_matches_independent_loss(self):
        self.perturb()
        teacher = torch.randn(1, 7, 4, dtype=torch.float16)
        targets = torch.arange(7)[None]
        hidden = self.bank.forward(self.ids)
        receipt = chunked_loss_backward(hidden, teacher, targets, self.model.lm_head,
                                        self.model.lm_head, Scale(), chunk_tokens=3)
        staged = {n: p.grad.clone() for n, p in self.bank.masters.items()}
        self.clear()
        hidden = self.bank.forward(self.ids)
        logits = self.model.lm_head(hidden).float()
        with torch.no_grad():
            teacher_logp = F.log_softmax(self.model.lm_head(teacher).float(), -1)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='sum') / 7
        kl = F.kl_div(F.log_softmax(logits, -1), teacher_logp, log_target=True, reduction='sum') / 7
        (16 * .5 * (ce + kl)).backward()
        self.assertEqual(receipt['targets'], 7)
        self.assertAlmostEqual(receipt['loss'], float(.5 * (ce + kl).detach()), places=5)
        for name, master in self.bank.masters.items():
            self.assertTrue(torch.allclose(master.grad, staged[name], rtol=.003, atol=.001), name)
        self.bank.assert_frozen()

    def test_shape_dtype_master_validation_and_atomic_rejection(self):
        for rank in (True, 3, 8):
            with self.assertRaises(ValueError):
                LowRankMasters(self.model, rank=rank)
        with self.assertRaises(ValueError):
            LowRankMasters(self.model, seed=123)
        state = self.bank.state_dict()
        for replacement in (torch.ones(1), torch.ones_like(next(iter(state.values()))).half(),
                            torch.full_like(next(iter(state.values())), float('nan')),
                            torch.full_like(next(iter(state.values())), 1e8)):
            bad = dict(state)
            bad[next(iter(bad))] = replacement
            with self.assertRaises(ValueError):
                self.bank.load_state_dict(bad)
            for name in state:
                self.assertTrue(torch.equal(state[name], self.bank.masters[name]))
        other = LowRankMasters(copy.deepcopy(self.model))
        other.load_state_dict(state)
        for name in state:
            self.assertTrue(torch.equal(other.masters[name], state[name]))
        with self.assertRaises(ValueError):
            merge_fp16_weight(torch.zeros(4, 4).half(), torch.zeros(4, 3).half(), torch.zeros(3, 4).half())
        with self.assertRaises(ValueError):
            merge_fp16_weight(torch.zeros(4, 4).half(), torch.zeros(4, 4), torch.zeros(4, 4).half())
        with torch.no_grad():
            self.bank.masters['layer0.in_proj.B'].fill_(1e8)
        with self.assertRaises(ValueError):
            self.bank.export_factors()

    def test_mutation_guard_and_precision_context_no_global_change(self):
        self.perturb()
        for master in self.bank.parameters():
            master.grad = torch.zeros_like(master)
        self.bank.masters['layer0.in_proj.B'].grad[0, 0] = float('inf')
        overflow = gradient_receipt(self.bank)['per_tensor']['layer0.in_proj.B']
        self.assertFalse(overflow['finite'])
        for field in ('minimum', 'maximum', 'maximum_absolute'):
            self.assertIsNone(overflow[field])
        before = torch.get_float32_matmul_precision()
        reference = self.bank.merge('layer0.in_proj')
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            self.assertTrue(torch.equal(reference, self.bank.merge('layer0.in_proj')))
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(before, torch.get_float32_matmul_precision())
        # Test the CUDA guard without CUDA initialization or GPU operations.
        with patch('mamba_e8w5.low_rank_training.torch.backends.cuda.matmul', SimpleNamespace(allow_tf32=True)):
            with self.assertRaisesRegex(RuntimeError, 'explicitly'):
                _check_precision(torch.device('cuda'))
        with torch.no_grad():
            self.model.backbone.layers[0].norm.weight.add_(.01)
        with self.assertRaisesRegex(RuntimeError, 'Frozen native parameter'):
            self.bank.assert_frozen()


if __name__ == '__main__':
    unittest.main()
