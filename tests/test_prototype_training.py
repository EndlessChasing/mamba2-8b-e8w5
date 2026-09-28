"""CPU-only prototype-bank graph, rounding, freezing and serialization checks."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mamba_e8w5.codec import load_reference_primitives,decode_e8,write_e8
from mamba_e8w5.learned_e8_codebook import write_prototypes,read_prototypes
from mamba_e8w5.prototype_training import (PrototypeMasters,selected_inventory,
    load_projection_payloads,training_schedule,gradient_receipt,export_parity)


class RMS(nn.Module):
    def __init__(self,n):
        super().__init__();self.weight=nn.Parameter(torch.ones(n,dtype=torch.float16))

    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).half()*self.weight


class Mixer(nn.Module):
    def __init__(self):
        super().__init__();self.use_mem_eff_path=False
        self.in_proj=nn.Linear(8,16,bias=False,dtype=torch.float16)
        self.out_proj=nn.Linear(16,8,bias=False,dtype=torch.float16)
        self.norm=RMS(16)

    def forward(self,x,inference_params=None):
        return self.out_proj(self.norm(F.silu(self.in_proj(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.fused_add_norm=False;self.residual_in_fp32=False;self.mlp=None
        self.norm=RMS(8);self.mixer=Mixer()

    def forward(self,h,r=None,inference_params=None):
        r=h+r if r is not None else h
        return self.mixer(self.norm(r),inference_params=inference_params),r


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.fused_add_norm=False;self.residual_in_fp32=False
        self.embedding=nn.Embedding(31,8,dtype=torch.float16)
        self.layers=nn.ModuleList([Block(),Block()]);self.norm_f=RMS(8)

    def forward(self,ids):
        h=self.embedding(ids);r=None
        for layer in self.layers:h,r=layer(h,r)
        return self.norm_f(h+r)


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=Backbone();self.lm_head=nn.Linear(8,31,bias=False,dtype=torch.float16)


class PrototypeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cb,_=load_reference_primitives()

    def setUp(self):
        torch.manual_seed(155)
        self.model=Model().eval();self.payloads={}
        for label,entry in selected_inventory(2).items():
            weight=self.model.get_parameter(entry['source_key']);m,n=weight.shape
            payload={'indices':torch.randint(0,65536,(m,n//8),dtype=torch.int32),
                'balance':(.75+torch.rand(n)).half(),
                'input_sign':(2*torch.randint(0,2,(n,))-1).to(torch.int8),
                'output_sign':(2*torch.randint(0,2,(m,))-1).to(torch.int8),
                'scale':torch.tensor(.12,dtype=torch.float32),'axis_residual_amplitude':0.0}
            self.payloads[label]=payload
            with torch.no_grad():weight.copy_(decode_e8(payload,self.cb,device='cpu'))
        self.bank=PrototypeMasters(self.model,self.payloads,self.cb,payload_device='cpu')
        self.ids=torch.arange(7)[None]

    def perturb(self):
        with torch.no_grad():
            for i,master in enumerate(self.bank.parameters()):
                master.copy_(torch.linspace(-.0151,.0249,master.numel()).reshape_as(master)*(i+1))

    def test_zero_is_native_exact_and_every_table_gets_gradients(self):
        original={n:p.detach().clone() for n,p in self.model.named_parameters()}
        with torch.no_grad():self.assertTrue(torch.equal(self.model.backbone(self.ids),self.bank.forward(self.ids)))
        hidden=self.bank.forward(self.ids)
        (hidden.float()*torch.linspace(-1,2,hidden.numel()).reshape_as(hidden)).sum().backward()
        receipt=gradient_receipt(self.bank)
        self.assertEqual(receipt['tensors'],4);self.assertEqual(receipt['parameters'],8192)
        self.assertTrue(receipt['finite'])
        self.assertTrue(all(p['nonzero']>0 for p in receipt['per_tensor'].values()))
        for name,value in self.model.named_parameters():
            self.assertTrue(torch.equal(original[name],value));self.assertIsNone(value.grad)

    def test_full_toy_checkpoint_and_noncheckpoint_gradients_match(self):
        self.perturb();results=[];outputs=[]
        for enabled in (False,True):
            for master in self.bank.parameters():master.grad=None
            h=self.bank.forward(self.ids,use_checkpoint=enabled)
            (h.float()*torch.arange(h.numel()).reshape_as(h)/h.numel()).sum().backward()
            outputs.append(h.detach().clone())
            results.append({n:p.grad.clone() for n,p in self.bank.masters.items()})
        self.assertTrue(torch.equal(*outputs))
        for label in results[0]:self.assertTrue(torch.equal(results[0][label],results[1][label]),label)
        self.bank.assert_frozen()

    def test_single_block_checkpoint_gradients_match_and_only_local_tables_participate(self):
        self.perturb()
        with torch.no_grad():
            h=self.model.backbone.embedding(self.ids)
            h,r=self.model.backbone.layers[0](h,None)
        runs=[]
        for enabled in (False,True):
            for master in self.bank.parameters():master.grad=None
            local_input=h.detach().requires_grad_(True)
            output,residual=self.bank.forward_block(1,local_input,r,use_checkpoint=enabled)
            coefficients=torch.linspace(-.7,1.3,output.numel()).reshape_as(output)
            ((output.float()+residual.float())*coefficients).sum().backward()
            local={n:p.grad.clone() for n,p in self.bank.masters.items() if p.grad is not None}
            self.assertEqual(set(local),{'layer1.in_proj','layer1.out_proj'})
            runs.append((output.detach(),residual.detach(),local_input.grad,local))
        for a,b in zip(runs[0][:3],runs[1][:3]):self.assertTrue(torch.equal(a,b))
        for name in runs[0][3]:self.assertTrue(torch.equal(runs[0][3][name],runs[1][3][name]))

    def test_exported_tables_match_native_install_and_restore_all_references(self):
        self.perturb();before={n:id(p) for n,p in self.model.named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            tables=self.bank.export_tables()
            for name,table in tables.items():write_prototypes(Path(directory)/f'{name}.e8p',table)
            restored={name:read_prototypes(Path(directory)/f'{name}.e8p') for name in tables}
            receipt=export_parity(self.bank,restored,self.ids)
        self.assertTrue(receipt['bitwise_equal'])
        self.assertTrue(receipt['last8_logits_bitwise_equal'])
        self.assertEqual(before,{n:id(p) for n,p in self.model.named_parameters()})
        self.bank.assert_frozen()

    def test_load_export_and_payload_reader_contracts(self):
        self.perturb();state=self.bank.state_dict();tables=self.bank.export_tables()
        for name in state:self.assertTrue(torch.equal(tables[name],state[name].half()))
        other=PrototypeMasters(self.model,self.payloads,self.cb)
        other.load_state_dict(state)
        for name in state:self.assertTrue(torch.equal(other.masters[name],state[name]))
        bad=dict(state);bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):other.load_state_dict(bad)
        bad=dict(state);bad[next(iter(bad))]=torch.zeros(256,8,dtype=torch.float16)
        with self.assertRaises(ValueError):other.load_state_dict(bad)
        bad=dict(state);bad[next(iter(bad))]=torch.full((256,8),float('nan'))
        with self.assertRaises(ValueError):other.load_state_dict(bad)
        with tempfile.TemporaryDirectory() as directory:
            for label,payload in self.payloads.items():
                m,g=payload['indices'].shape
                write_e8(Path(directory)/f'{label}.e8',payload,{'shape':[m,g*8]})
            restored=load_projection_payloads(directory,layers=2)
        for name in restored:
            for key,value in self.payloads[name].items():
                if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(restored[name][key],value))
        inventory=selected_inventory()
        self.assertEqual(len(inventory),112);self.assertEqual(sum(v['numel'] for v in inventory.values()),229376)
        schedule=training_schedule();self.assertEqual(len(schedule),128)
        for i in range(4):self.assertEqual(sorted(schedule[i*32:(i+1)*32]),list(range(32)))

    def test_mutation_guard_and_nonfinite_export(self):
        with torch.no_grad():self.bank.masters['layer0.in_proj'].fill_(1e8)
        with self.assertRaises(ValueError):self.bank.export_tables()
        with torch.no_grad():self.model.backbone.layers[0].norm.weight.add_(.01)
        with self.assertRaisesRegex(RuntimeError,'Frozen native parameter'):self.bank.assert_frozen()


if __name__=='__main__':unittest.main()
