"""CPU algebra/native-rounding tests; no pretrained model or GPU required."""
import copy
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mamba_e8w5.projection_statistics import (
    collect_projection_statistics, projection_path, score_dense_from_statistics,
    score_native_fp16,
)


class Mixer(nn.Module):
    def __init__(self,width,expanded):
        super().__init__()
        self.in_proj=nn.Linear(width,expanded,bias=False)
        self.out_proj=nn.Linear(expanded,width,bias=False)
        self.use_mem_eff_path=False

    def forward(self,x):
        return self.out_proj(torch.tanh(self.in_proj(x)))


class Block(nn.Module):
    def __init__(self,width,expanded):
        super().__init__();self.mixer=Mixer(width,expanded)

    def forward(self,x):
        return x+self.mixer(x)


class Backbone(nn.Module):
    def __init__(self,width=5,expanded=7,layers=2):
        super().__init__()
        self.embedding=nn.Embedding(17,width)
        self.layers=nn.ModuleList(Block(width,expanded) for _ in range(layers))

    def forward(self,ids):
        x=self.embedding(ids)
        for block in self.layers:x=block(x)
        return x


class Model(nn.Module):
    def __init__(self,**kwargs):
        super().__init__();self.backbone=Backbone(**kwargs)


def capture(model,windows,names):
    values={name:[] for name in names};handles=[]
    def hook(name):
        def save(module,args,output):
            values[name].append((args[0].detach().reshape(-1,args[0].shape[-1]).clone(),
                                 output.detach().reshape(-1,output.shape[-1]).clone()))
        return save
    try:
        for name in names:handles.append(model.get_submodule(projection_path(name)).register_forward_hook(hook(name)))
        with torch.no_grad():
            for ids in windows:model.backbone(ids[None] if ids.ndim==1 else ids)
    finally:
        for handle in handles:handle.remove()
    return {name:(torch.cat([v[0] for v in rows]),torch.cat([v[1] for v in rows])) for name,rows in values.items()}


class ProjectionStatisticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        self.teacher=Model().half().eval()
        torch.manual_seed(207)
        self.candidate=Model().half().eval()
        self.names=('layer0.in_proj','layer0.out_proj','layer1.in_proj')
        self.windows=[torch.tensor([1,2,3],dtype=torch.long),torch.tensor([[4,5],[6,7]],dtype=torch.long)]

    def test_native_moments_orientation_count_and_independent_splits(self):
        teacher=capture(self.teacher,self.windows,self.names)
        candidate=capture(self.candidate,self.windows,self.names)
        result=collect_projection_statistics(self.teacher,self.candidate,self.windows,selected_names=self.names,split_label='fit')
        self.assertEqual(result['tokens'],7)
        self.assertEqual([b['tokens'] for b in result['batches']],[3,4])
        for name in self.names:
            x=candidate[name][0].float();y=teacher[name][1].float();s=result['matrices'][name]
            self.assertEqual(s['tokens'],7)
            torch.testing.assert_close(s['H'],x.T@x/7,rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(s['K'],y.T@x/7,rtol=2e-6,atol=2e-6)
            self.assertAlmostEqual(s['teacher_energy'],float(y.square().sum(dtype=torch.float64)/7),places=12)
        held=collect_projection_statistics(self.teacher,self.candidate,self.windows[:1],selected_names=self.names,split_label='holdout')
        self.assertEqual(held['tokens'],3)
        self.assertEqual(result['matrices']['layer0.in_proj']['tokens'],7)

    def test_native_teacher_rounding_is_not_weight_crossmoment_factorization(self):
        name='layer0.in_proj'
        t=capture(self.teacher,self.windows,(name,))[name]
        q=capture(self.candidate,self.windows,(name,))[name]
        result=collect_projection_statistics(self.teacher,self.candidate,self.windows,selected_names=(name,),split_label='fit')
        exact_native=t[1].float().T@q[0].float()/7
        factorized=self.teacher.get_submodule(projection_path(name)).weight.detach().float()@(t[0].float().T@q[0].float()/7)
        torch.testing.assert_close(result['matrices'][name]['K'],exact_native,rtol=2e-6,atol=2e-6)
        self.assertGreater(float((exact_native-factorized).abs().max()),1e-6)

    def test_quadratic_matches_direct_real_linear_error(self):
        name='layer0.in_proj'
        result=collect_projection_statistics(self.teacher,self.candidate,self.windows,selected_names=(name,),split_label='fit')
        x=capture(self.candidate,self.windows,(name,))[name][0].double()
        y=capture(self.teacher,self.windows,(name,))[name][1].double()
        weight=self.candidate.get_submodule(projection_path(name)).weight.detach()*1.25
        expected=float((F.linear(x,weight.double())-y).square().sum()/7)
        score=score_dense_from_statistics(weight,result['matrices'][name])
        self.assertAlmostEqual(score['mse_per_token'],expected,delta=2e-6*max(1,expected))
        self.assertAlmostEqual(score['mean_output_squared_error'],expected/weight.shape[0],delta=2e-6*max(1,expected))
        self.assertIn('does not reproduce native FP16',score['prediction_semantics'])

    def test_native_replacements_are_independent_and_json_safe(self):
        before={k:v.clone() for k,v in self.candidate.state_dict().items()}
        candidate=capture(self.candidate,self.windows,self.names)
        teacher=capture(self.teacher,self.windows,self.names)
        replacement={name:self.candidate.get_submodule(projection_path(name)).weight.detach().clone()*1.75 for name in self.names}
        score=score_native_fp16(self.teacher,self.candidate,self.windows,replacement,split_label='holdout')
        json.dumps(score,allow_nan=False)
        for name in self.names:
            y=teacher[name][1].float();predicted=F.linear(candidate[name][0],replacement[name]).float()
            error=float((predicted-y).square().sum(dtype=torch.float64)/7)
            self.assertAlmostEqual(score['matrices'][name]['mse_per_token'],error,places=12)
            self.assertAlmostEqual(score['matrices'][name]['mean_output_squared_error'],error/replacement[name].shape[0],places=12)
            for key,value in before.items():self.assertTrue(torch.equal(value,self.candidate.state_dict()[key]))
        self.assertFalse(score['replacements_installed'])
        self.assertFalse(score['combined_model_quality_evaluated'])
        after=capture(self.candidate,self.windows,self.names)
        for name in self.names:self.assertTrue(torch.equal(after[name][0],candidate[name][0]))

    def test_same_model_identity_and_zero_native_error(self):
        weights={name:self.teacher.get_submodule(projection_path(name)).weight for name in self.names}
        score=score_native_fp16(self.teacher,self.teacher,self.windows,weights,split_label='identity')
        for s in score['matrices'].values():
            self.assertEqual(s['mean_output_squared_error'],0)
            self.assertEqual(s['baseline_native_mse_per_token'],0)
        self.assertTrue(score['parameter_identity_and_version_unchanged'])

    def test_native_role_partitions_sum_to_total(self):
        torch.manual_seed(15)
        teacher=Model(width=3,expanded=18560,layers=1).half().eval()
        candidate=copy.deepcopy(teacher)
        name='layer0.in_proj'
        weight=candidate.get_submodule(projection_path(name)).weight.detach()*1.1
        score=score_native_fp16(teacher,candidate,self.windows[:1],{name:weight},split_label='holdout')['matrices'][name]
        self.assertEqual(list(score['partitions']),['z','x','B','C','dt'])
        self.assertAlmostEqual(sum(p['mse_per_token'] for p in score['partitions'].values()),score['mse_per_token'],places=9)

    def test_failures_cleanup_hooks_and_reject_invalid_contracts(self):
        def run(**kw):
            return collect_projection_statistics(self.teacher,self.candidate,self.windows,selected_names=self.names,split_label='fit',**kw)
        self.teacher.train()
        with self.assertRaises(ValueError):run()
        self.teacher.eval()
        with self.assertRaises(ValueError):collect_projection_statistics(self.teacher,self.candidate,[],selected_names=self.names,split_label='empty')
        with self.assertRaises(ValueError):collect_projection_statistics(self.teacher,self.candidate,self.windows,selected_names=(self.names[0],self.names[0]),split_label='fit')
        with torch.no_grad():self.teacher.get_submodule(projection_path(self.names[0])).weight[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):run()
        for model in (self.teacher,self.candidate):
            for name in self.names:self.assertEqual(len(model.get_submodule(projection_path(name))._forward_hooks),0)


if __name__=='__main__':unittest.main()
