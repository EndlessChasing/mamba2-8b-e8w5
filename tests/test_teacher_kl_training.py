"""CPU proofs of pure-KL gradients, scaling and the fixed fresh-data partition."""
import json
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from mamba_e8w5.teacher_kl_training import teacher_kl_backward
from scripts import prepare_teacher_kl_data as data


class Scale:
    def __init__(self,factor):
        self.factor,self.calls = factor,0

    def scale(self,value):
        self.calls += 1
        return self.factor*value


class TeacherKLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(928)
        self.student_head = nn.Linear(4,19,bias=False,dtype=torch.float16).requires_grad_(False)
        self.teacher_head = nn.Linear(4,19,bias=False,dtype=torch.float16).requires_grad_(False)
        self.initial = torch.randn(2,7,4,dtype=torch.float32)*.2
        self.teacher = torch.randn(2,7,4,dtype=torch.float16).requires_grad_()
        self.targets = torch.arange(14).reshape(2,7)

    def test_dense_kl_gradient_tail_and_single_scaling(self):
        master = self.initial.clone().requires_grad_()
        scaler = Scale(16)
        receipt = teacher_kl_backward(master.half(),self.teacher,self.targets,
            self.student_head,self.teacher_head,scaler,chunk_tokens=3)
        actual = master.grad.clone()
        reference = self.initial.clone().requires_grad_()
        with torch.no_grad():
            teacher_logp = F.log_softmax(self.teacher_head(self.teacher).float(),dim=-1)
        logits = self.student_head(reference.half()).float()
        kl = F.kl_div(F.log_softmax(logits,-1),teacher_logp,reduction='sum',log_target=True)/14
        (16*kl).backward()
        self.assertTrue(torch.allclose(actual,reference.grad,rtol=.003,atol=.001))
        self.assertAlmostEqual(receipt['loss'],float(kl.detach()),places=6)
        self.assertEqual(receipt['loss'],receipt['teacher_to_student_kl'])
        self.assertEqual(receipt['targets'],14)
        self.assertEqual(scaler.calls,3)
        self.assertTrue(receipt['ce_logging_only'])
        self.assertEqual(receipt['ce_gradient_coefficient'],0.)
        self.assertTrue(receipt['scaled_hidden_gradient_finite'])
        self.assertIsNone(self.teacher.grad)
        self.assertTrue(all(p.grad is None for head in (self.student_head,self.teacher_head) for p in head.parameters()))
        unscaled = self.initial.clone().requires_grad_()
        teacher_kl_backward(unscaled.half(),self.teacher,self.targets,
            self.student_head,self.teacher_head,None,chunk_tokens=3)
        self.assertTrue(torch.allclose(actual,16*unscaled.grad,rtol=.003,atol=.001))

    def test_label_permutation_changes_ce_only_not_kl_or_gradients(self):
        results = []
        for targets in (self.targets,self.targets.flip(1)):
            master = self.initial.clone().requires_grad_()
            receipt = teacher_kl_backward(master.half(),self.teacher,targets,
                self.student_head,self.teacher_head,None,chunk_tokens=3)
            results.append((receipt,master.grad.clone()))
        self.assertNotEqual(results[0][0]['ce'],results[1][0]['ce'])
        self.assertEqual(results[0][0]['loss'],results[1][0]['loss'])
        self.assertTrue(torch.equal(results[0][1],results[1][1]))

    def test_native_precision_context_and_invalid_forward(self):
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            master = self.initial.clone().requires_grad_()
            teacher_kl_backward(master.half(),self.teacher,self.targets,
                self.student_head,self.teacher_head,None,chunk_tokens=3)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        for chunk in (0,True,1.5):
            with self.assertRaises(ValueError):
                teacher_kl_backward(self.initial.half().requires_grad_(),self.teacher,self.targets,
                    self.student_head,self.teacher_head,None,chunk_tokens=chunk)
        bad = self.initial.half()
        bad[0,0,0] = float('nan')
        with self.assertRaises(FloatingPointError):
            teacher_kl_backward(bad.requires_grad_(),self.teacher,self.targets,
                self.student_head,self.teacher_head,None,chunk_tokens=3)

    def test_actual_pinned_partition_and_single_pass_schedule(self):
        documents = data.verify_pins()
        excluded = data.previous_intervals(documents)
        training,heldout,unused,overlap = data.select_starts(excluded)
        self.assertEqual((len(training),len(heldout),len(unused)),(448,64,7))
        self.assertFalse(set(training)&set(heldout))
        self.assertEqual(unused,data.UNUSED_STARTS)
        self.assertEqual(overlap['all_previous_overlap_tokens'],0)
        self.assertEqual(overlap['training_heldout_overlap_tokens'],0)
        self.assertEqual(overlap['eligible_blocks'],519)
        self.assertEqual((training[0],training[-1]),(18432,2514944))
        self.assertEqual((heldout[0],heldout[-1]),(14336,2516992))
        schedule = data.train_schedule()
        generator = torch.Generator(device='cpu').manual_seed(20260928)
        self.assertEqual(schedule,torch.randperm(448,generator=generator).tolist())
        self.assertEqual(sorted(schedule),list(range(448)))
        broken = dict(excluded)
        broken['generalization_fresh'] = broken['generalization_fresh'][:-1]
        with self.assertRaises(ValueError):
            data.select_starts(broken)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
