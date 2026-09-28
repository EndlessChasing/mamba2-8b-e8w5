"""Check actual all-small functional mapping, frozen weights and data exclusion."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from test_norm_training import Toy,RMS
from mamba_e8w5.norm_training import export_parity,gradient_receipt
from mamba_e8w5.small_training import (SmallMasters,selected_inventory,training_schedule,
    eligible_window_starts,family_ranges)


class StatefulToyMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_mem_eff_path=False
        self.in_proj=nn.Linear(4,8,bias=False).half()
        self.out_proj=nn.Linear(8,4,bias=False).half()
        self.norm=RMS(8)
        self.conv1d=nn.Conv1d(8,8,3,padding=2,groups=8).half()
        self.dt_bias=nn.Parameter(torch.linspace(-2,-1,8).half())
        self.A_log=nn.Parameter(torch.linspace(-1,0,8).half())
        self.D=nn.Parameter(torch.ones(8).half())

    def forward(self,x,inference_params=None):
        z=self.in_proj(x)
        conv=self.conv1d(z.transpose(1,2))[...,:z.shape[1]].transpose(1,2)
        a=-torch.exp(self.A_log.float())
        dt=F.softplus(conv.float()+self.dt_bias.float())
        y=(z.float()*torch.exp(a*dt)+conv.float()*self.D.float()).half()
        return self.out_proj(self.norm(y))


class SmallTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.model=Toy()
        for block in self.model.backbone.layers: block.mixer=StatefulToyMixer()
        inventory={name:{'shape':list(p.shape),'numel':p.numel()}
            for name,p in self.model.named_parameters()
            if name!='backbone.embedding.weight' and name!='lm_head.weight' and not ('.in_proj.' in name or '.out_proj.' in name)}
        self.bank=SmallMasters(self.model,inventory)
        self.ids=torch.arange(7)[None]

    def test_all_families_checkpoint_and_export(self):
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model.backbone(self.ids),self.bank.forward(self.ids,False)))
            for p in self.bank.parameters(): p.add_(torch.linspace(-.02,.03,p.numel()).reshape(p.shape))
        results=[]
        for enabled in (False,True):
            for p in self.bank.parameters(): p.grad=None
            hidden=self.bank.forward(self.ids,enabled)
            (hidden.float()*torch.arange(hidden.numel()).reshape(hidden.shape)/hidden.numel()).sum().backward()
            receipt=gradient_receipt(self.bank)
            self.assertTrue(receipt['finite'])
            groups=family_ranges(self.bank,True)
            self.assertEqual(set(groups),{'gated_norm','block_and_final_norm','conv_weight','conv_bias','dt_bias','A_log','D'})
            self.assertTrue(all(v['nonzero']>0 for v in groups.values()))
            results.append({n:p.grad.clone() for n,p in self.bank.masters.items()})
        for name in results[0]: self.assertTrue(torch.equal(results[0][name],results[1][name]),name)
        original={n:p.detach().cpu() for n,p in self.model.named_parameters() if n in self.bank.inventory}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'other_fp16.pt'
            torch.save(self.bank.export_other(original),path)
            self.assertTrue(export_parity(self.bank,path,self.ids)['bitwise_equal'])
        self.assertTrue(self.bank.stability_receipt()['native_A_finite'])

    def test_stability_rejects_underflow(self):
        with torch.no_grad(): self.bank.masters['backbone.layers.0.mixer.A_log'].fill_(-1000)
        with self.assertRaises(FloatingPointError): self.bank.stability_receipt()

    def test_deterministic_disjoint_selection_and_schedule(self):
        starts,receipt=eligible_window_starts(100,[0,40],window_tokens=10,count=4)
        self.assertEqual(starts,[10,30,60,90])
        self.assertEqual(receipt['original_calibration_overlap_tokens'],0)
        with self.assertRaises(ValueError): eligible_window_starts(30,[0,10],window_tokens=10,count=4)
        schedule=training_schedule()
        self.assertEqual(len(schedule),1024)
        for epoch in range(4): self.assertEqual(sorted(schedule[256*epoch:256*(epoch+1)]),list(range(256)))
        self.assertEqual(len(selected_inventory()),393)
        self.assertEqual(sum(v['numel'] for v in selected_inventory().values()),3580928)


if __name__=='__main__': unittest.main()
