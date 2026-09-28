"""Small CPU checks for the actual compensation orchestration and loss staging."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from mamba_e8w5.norm_training import (NormMasters, selected_inventory, training_schedule,
    chunked_loss_backward, gradient_receipt, export_parity)


class RMS(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, dtype=torch.float16))

    def forward(self, x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-5)).half()*self.weight


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_mem_eff_path = False
        self.in_proj = nn.Linear(4, 8, bias=False).half()
        self.norm = RMS(8)
        self.out_proj = nn.Linear(8, 4, bias=False).half()

    def forward(self, x, inference_params=None):
        return self.out_proj(self.norm(torch.tanh(self.in_proj(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_add_norm = self.residual_in_fp32 = False
        self.mlp = None
        self.norm, self.mixer = RMS(4), Mixer()

    def forward(self, hidden, residual=None, inference_params=None):
        residual = hidden+residual if residual is not None else hidden
        return self.mixer(self.norm(residual.to(self.norm.weight.dtype)), inference_params), residual


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_add_norm = self.residual_in_fp32 = False
        self.embedding = nn.Embedding(19, 4).half()
        self.layers = nn.ModuleList([Block(), Block()])
        self.norm_f = RMS(4)

    def forward(self, ids):
        h, r = self.embedding(ids), None
        for layer in self.layers:
            h, r = layer(h, r)
        return self.norm_f((h+r).to(self.norm_f.weight.dtype))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.lm_head = nn.Linear(4, 19, bias=False).half()


class Scale:
    def scale(self, value):
        return value*16


class NormTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = Toy()
        self.bank = NormMasters(self.model, selected_inventory(2, 4, 8))
        self.ids = torch.arange(7)[None]

    def test_native_checkpoint_recompute_and_changed_export(self):
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model.backbone(self.ids), self.bank.forward(self.ids, False)))
            # Deliberately change every master before checkpoint recomputation.
            for p in self.bank.parameters():
                p.add_(torch.linspace(-.04, .07, p.numel()))
        grads = []
        for checkpointed in (False, True):
            for p in self.bank.parameters(): p.grad = None
            h = self.bank.forward(self.ids, checkpointed)
            (h.float()*torch.arange(h.numel()).reshape(h.shape)/h.numel()).sum().backward()
            grads.append({n: p.grad.clone() for n,p in self.bank.masters.items()})
            receipt = gradient_receipt(self.bank)
            self.assertTrue(receipt['finite'])
            self.assertGreater(receipt['nonzero'], 0)
        for name in grads[0]:
            self.assertTrue(torch.equal(grads[0][name], grads[1][name]), name)
        original = {n:p.detach().cpu() for n,p in self.model.named_parameters() if n in self.bank.inventory}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'other_fp16.pt'
            torch.save(self.bank.export_other(original), path)
            self.assertTrue(export_parity(self.bank, path, self.ids)['bitwise_equal'])

    def test_chunk_tail_weighting_and_loss_scaled_once(self):
        head = self.model.lm_head
        teacher = torch.randn(1, 7, 4, dtype=torch.float16)
        initial = torch.randn_like(teacher)
        targets = torch.arange(7)[None]
        staged = initial.clone().requires_grad_()
        receipt = chunked_loss_backward(staged, teacher, targets, head, head, Scale(), chunk_tokens=3)
        full = initial.clone().requires_grad_()
        logits = head(full).float()
        with torch.no_grad(): teacher_logp = F.log_softmax(head(teacher).float(), -1)
        ce = F.cross_entropy(logits.flatten(0,1), targets.flatten(), reduction='sum')/7
        kl = F.kl_div(F.log_softmax(logits,-1), teacher_logp, log_target=True, reduction='sum')/7
        loss = .5*(ce+kl)
        (16*loss).backward()
        self.assertAlmostEqual(receipt['loss'], float(loss.detach()), places=5)
        self.assertTrue(torch.allclose(staged.grad, full.grad, rtol=.003, atol=.001))
        self.assertEqual(receipt['targets'], 7)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_schedule_and_inventory(self):
        schedule = training_schedule()
        self.assertEqual(len(schedule), 128)
        for epoch in range(4):
            self.assertEqual(sorted(x['window'] for x in schedule[epoch*32:(epoch+1)*32]), list(range(32)))
        self.assertEqual(len(selected_inventory()), 113)
        self.assertEqual(sum(x['numel'] for x in selected_inventory().values()), 692224)


if __name__ == '__main__':
    unittest.main()
