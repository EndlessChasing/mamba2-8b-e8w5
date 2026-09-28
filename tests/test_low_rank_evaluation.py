"""CPU boundaries for final rank4 export identity and native PPL accounting."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('low_rank_evaluation',ROOT/'scripts/evaluate_low_rank_compensation.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class LowRankEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.plan = evaluation.window_plan(evaluation.ppl_windows(torch.arange(evaluation.TARGETS+1),2048))

    def checkpoint(self):
        schedule = evaluation.expected_schedule()
        inventory = evaluation.expected_inventory()
        return {'format':'MAMBA2_LOW_RANK_TRAINING_CHECKPOINT_V1','binding':{'fixture':'bound'},
            'parent_manifest_sha256':evaluation.PARENT_SHA,'small_manifest_sha256':evaluation.SMALL_SHA,
            'smoke_report_sha256':'0'*64,'frozen_model_sha256':{},'resumable':False,'reason':'final',
            'schedule':schedule,'successful_updates':1024,'attempts':1024,'overflow_retries':0,
            'history':[{'attempt':i+1,'successful_updates_after':i+1,'overflow':False,
                        'targets':2047,'schedule_entry':row} for i,row in enumerate(schedule)],
            'masters':{label+'.'+part:torch.zeros(entry[part+'_shape'],dtype=torch.float32)
                       for label,entry in inventory.items() for part in ('B','A')}}

    @staticmethod
    def factors(state):
        return {label:tuple(state['masters'][label+'.'+part].half() for part in ('B','A'))
                for label in evaluation.LABELS}

    def verify(self,state,factors):
        return evaluation.verify_checkpoint_factors(state,factors,{'fixture':'bound'},{},'0'*64)

    def test_actual112_geometry_master_rounding_and_signed_zero(self):
        state = self.checkpoint()
        state['masters']['layer55.out_proj.A'][0,0] = 1.0001
        factors = self.factors(state)
        receipt = self.verify(state,factors)
        self.assertEqual(receipt['verified_factor_files'],112)
        self.assertEqual(receipt['verified_masters'],224)
        self.assertEqual(receipt['verified_parameters'],7827456)
        self.assertEqual(sum(row['bytes'] for row in evaluation.expected_inventory().values()),15658496)
        factors['layer0.in_proj'][0][0,0] = -0.0
        with self.assertRaisesRegex(ValueError,'differs from serialized'):
            self.verify(state,factors)
        factors['layer0.in_proj'][0][0,0] = 0.0
        del factors['layer55.out_proj']
        with self.assertRaisesRegex(ValueError,'Exactly112'):
            self.verify(state,factors)
        factors = self.factors(state)
        state['masters']['layer55.out_proj.A'] = state['masters']['layer55.out_proj.A'].half()
        with self.assertRaisesRegex(ValueError,'Invalid final master'):
            self.verify(state,factors)

    def test_exact1024_schedule_overflow_budget_and_counter_types(self):
        state = self.checkpoint()
        state.pop('masters')
        for key,value in (('successful_updates',1024.),('attempts',1024.),('overflow_retries',0.)):
            bad = copy.deepcopy(state)
            bad[key] = value
            with self.assertRaises(ValueError):
                evaluation.verify_accounting(bad)
        for key,value in (('targets',2047.),('successful_updates_after',True),('schedule_entry',1.)):
            bad = copy.deepcopy(state)
            bad['history'][0][key] = value
            with self.assertRaises(ValueError):
                evaluation.verify_accounting(bad)
        bad = copy.deepcopy(state)
        bad['schedule'][0] = float(bad['schedule'][0])
        with self.assertRaises(ValueError):
            evaluation.verify_accounting(bad)
        for retries in (8,9):
            bad = copy.deepcopy(state)
            prefix = [{'attempt':i+1,'successful_updates_after':0,'overflow':True,'targets':2047,
                       'schedule_entry':bad['schedule'][0]} for i in range(retries)]
            bad['history'] = prefix+bad['history']
            for i,row in enumerate(bad['history']):
                row['attempt'] = i+1
            bad.update(attempts=1024+retries,overflow_retries=retries)
            if retries == 8:
                receipt = evaluation.verify_accounting(bad)
                self.assertEqual(receipt['successful_target_exposures'],2096128)
                self.assertEqual(receipt['attempted_target_exposures'],1032*2047)
            else:
                with self.assertRaisesRegex(ValueError,'accounting'):
                    evaluation.verify_accounting(bad)

    def test_independent_native_merge_bytes_and_base_immutability(self):
        base = torch.linspace(-.1,.2,16).reshape(4,4).half()
        base[0,0] = -0.0
        B = torch.full((4,4),1.0003).half()
        A = torch.linspace(-.07,.11,16).reshape(4,4).half()
        before = evaluation.tensor_sha_fp16(base)
        expected = (base.float()+B.float()@A.float()).half()
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            actual = evaluation.merge_native_projection(base,B,A)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(evaluation.tensor_sha_fp16(actual),evaluation.tensor_sha_fp16(expected))
        self.assertEqual(evaluation.tensor_sha_fp16(base),before)
        self.assertFalse(actual.requires_grad)
        with self.assertRaises(ValueError):
            evaluation.merge_native_projection(base,torch.zeros(4,3).half(),torch.zeros(3,4).half())
        B[0,0] = float('inf')
        with self.assertRaises(ValueError):
            evaluation.merge_native_projection(base,B,A)

    def test_flat_file_ledger_rejects_size_hash_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'factors.lrf'
            path.write_bytes(b'fixture')
            receipt = {'bytes':7,'sha256':evaluation.sha(path)}
            self.assertEqual(evaluation.checked_file(root,path.name,receipt),path)
            for name in ('../factors.lrf','/factors.lrf'):
                with self.assertRaisesRegex(ValueError,'safe flat'):
                    evaluation.checked_file(root,name,receipt)
            (root/'linked.lrf').symlink_to(path)
            with self.assertRaisesRegex(ValueError,'identity'):
                evaluation.checked_file(root,'linked.lrf',receipt)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity'):
                evaluation.checked_file(root,path.name,receipt)

    def arm(self,ppl):
        rows = [{**row,'nll':math.log(ppl)*row['target_tokens'],'ppl':ppl} for row in self.plan]
        nll = sum(row['nll'] for row in rows)
        return {'windows':rows,'nll':nll,'ppl':math.exp(nll/evaluation.TARGETS),
                'target_tokens':evaluation.TARGETS,'execution':'prefill','logits_chunk_tokens':64}

    def test_three_arms_pairing_weighted_targets_and_independent_gates(self):
        results = {name:self.arm(ppl) for name,ppl in zip(evaluation.ARMS,[7,8,7.91999])}
        comparison,paired = evaluation.compare_results(results,self.plan)
        self.assertTrue(comparison['meaningful_improvement_reference']['met'])
        self.assertFalse(comparison['source_plus5_percent_reference']['met'])
        self.assertEqual(len(paired),130)
        self.assertEqual(sum(row['target_tokens'] for row in paired),264764)
        self.assertEqual(paired[-1]['target_tokens'],572)
        results['low_rank_e8w5'] = self.arm(7.92001)
        self.assertFalse(evaluation.compare_results(results,self.plan)[0]['meaningful_improvement_reference']['met'])
        results['low_rank_e8w5'] = self.arm(7.34999)
        self.assertTrue(evaluation.compare_results(results,self.plan)[0]['source_plus5_percent_reference']['met'])
        results['low_rank_e8w5']['windows'][0]['token_sha256_int64le'] = '0'*64
        with self.assertRaisesRegex(ValueError,'token identity'):
            evaluation.compare_results(results,self.plan)

    def test_prepared_train_data_pins_when_available(self):
        directory = ROOT/'training_data/low_rank_compensation_v1'
        if not directory.exists():
            self.skipTest('Prepared TRAIN token binary is retained on the remote host')
        data = evaluation.verify_training_data(directory)
        self.assertEqual(data['overlap']['all_previous_fitting_overlap_tokens'],0)
        self.assertEqual(data['schedule'],evaluation.expected_schedule())
        self.assertEqual(data['four_pass_target_exposures'],2096128)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
