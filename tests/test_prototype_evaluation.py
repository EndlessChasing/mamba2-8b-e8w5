"""CPU boundary checks for independent final prototype export evaluation."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('prototype_evaluation',ROOT/'scripts/evaluate_prototype_compensation.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class PrototypeEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.plan=evaluation.window_plan(evaluation.ppl_windows(torch.arange(evaluation.TARGETS+1),2048))

    def checkpoint(self):
        schedule=evaluation.expected_schedule()
        return {'format':'MAMBA2_PROTOTYPE_TRAINING_CHECKPOINT_V1','binding':{'fixture':'bound'},
            'parent_manifest_sha256':evaluation.PARENT_SHA,'small_manifest_sha256':evaluation.SMALL_SHA,
            'smoke_report_sha256':'0'*64,'frozen_model_sha256':{},'resumable':False,'reason':'final',
            'schedule':schedule,'successful_updates':128,'attempts':128,'overflow_retries':0,
            'history':[{'attempt':i+1,'successful_updates_after':i+1,'overflow':False,
                        'targets':2047,'schedule_entry':row}for i,row in enumerate(schedule)],
            'masters':{label:torch.zeros(256,8,dtype=torch.float32)for label in evaluation.LABELS}}

    def verify(self,state,tables):
        return evaluation.verify_checkpoint_tables(state,tables,{'fixture':'bound'},{},'0'*64)

    def test_final128_actual_export_rounding(self):
        state=self.checkpoint()
        state['masters']['layer55.out_proj'][0,0]=1.0001
        tables={label:value.half()for label,value in state['masters'].items()}
        receipt=self.verify(state,tables)
        self.assertEqual(receipt['verified_tables'],112)
        self.assertEqual(receipt['verified_parameters'],229376)
        tables['layer55.out_proj'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'differs from serialized'):
            self.verify(state,tables)

    def test_wrong_step_coverage_dtype_and_signed_zero_rejected(self):
        state=self.checkpoint();tables={label:value.half()for label,value in state['masters'].items()}
        state['successful_updates']=127
        with self.assertRaisesRegex(ValueError,'final128'):self.verify(state,tables)
        state['successful_updates']=128
        del tables['layer0.in_proj']
        with self.assertRaisesRegex(ValueError,'Exactly112'):self.verify(state,tables)
        tables['layer0.in_proj']=torch.zeros(256,8,dtype=torch.float16)
        state['masters']['layer0.in_proj']=torch.zeros(256,8,dtype=torch.float16)
        with self.assertRaisesRegex(ValueError,'Invalid final'):self.verify(state,tables)
        state['masters']['layer0.in_proj']=torch.zeros(256,8,dtype=torch.float32)
        tables['layer0.in_proj'][0,0]=-0.0
        with self.assertRaisesRegex(ValueError,'differs from serialized'):self.verify(state,tables)

    def test_schedule_retry_accounting_and_provenance(self):
        for key,value in (('successful_updates',128.),('attempts',128.),('overflow_retries',0.)):
            bad=self.checkpoint();bad[key]=value
            with self.assertRaises(ValueError):evaluation.verify_accounting(bad)
        for key,value in (('targets',2047.),('successful_updates_after',True)):
            bad=self.checkpoint();bad['history'][0][key]=value
            with self.assertRaises(ValueError):evaluation.verify_accounting(bad)
        bad=self.checkpoint();bad['schedule'][0]['epoch']=0.
        with self.assertRaises(ValueError):evaluation.verify_accounting(bad)
        state=self.checkpoint()
        overflow={'attempt':1,'successful_updates_after':0,'overflow':True,'targets':2047,
                  'schedule_entry':state['schedule'][0]}
        state['history'].insert(0,overflow)
        for i,row in enumerate(state['history']):row['attempt']=i+1
        state.update(attempts=129,overflow_retries=1)
        self.assertEqual(evaluation.verify_accounting(state)['attempted_target_exposures'],129*2047)
        state['history'][1]['schedule_entry']=state['schedule'][1]
        with self.assertRaisesRegex(ValueError,'schedule'):evaluation.verify_accounting(state)
        state=self.checkpoint();tables={label:value.half()for label,value in state['masters'].items()}
        state['smoke_report_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'provenance'):self.verify(state,tables)

    def test_file_ledger_paths_size_hash_and_symlink(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);path=root/'table.proto';path.write_bytes(b'fixture')
            receipt={'bytes':7,'sha256':evaluation.sha(path)}
            self.assertEqual(evaluation.checked_file(root,'table.proto',receipt),path)
            for name in ('../table.proto','/table.proto'):
                with self.assertRaisesRegex(ValueError,'safe flat'):evaluation.checked_file(root,name,receipt)
            (root/'linked.proto').symlink_to(path)
            with self.assertRaisesRegex(ValueError,'identity'):evaluation.checked_file(root,'linked.proto',receipt)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity'):evaluation.checked_file(root,'table.proto',receipt)

    def arm(self,ppl):
        rows=[{**row,'nll':math.log(ppl)*row['target_tokens'],'ppl':ppl}for row in self.plan]
        nll=sum(row['nll']for row in rows)
        return {'windows':rows,'nll':nll,'ppl':math.exp(nll/evaluation.TARGETS),'target_tokens':evaluation.TARGETS,
                'execution':'prefill','logits_chunk_tokens':64}

    def test_three_arm_full_validation_thresholds(self):
        cases={name:self.arm(ppl)for name,ppl in zip(evaluation.ARMS,[7,8,7.91999])}
        result,paired=evaluation.compare_results(cases,self.plan)
        self.assertTrue(result['meaningful_improvement_reference']['met'])
        self.assertFalse(result['source_plus5_percent_reference']['met'])
        self.assertEqual(len(paired),130)
        self.assertEqual(result['window_counts']['improved'],130)
        cases['prototype_e8w5']=self.arm(7.92001)
        self.assertFalse(evaluation.compare_results(cases,self.plan)[0]['meaningful_improvement_reference']['met'])
        cases['prototype_e8w5']=self.arm(7.34999)
        self.assertTrue(evaluation.compare_results(cases,self.plan)[0]['source_plus5_percent_reference']['met'])
        cases['prototype_e8w5']['windows'][0]['token_sha256_int64le']='0'*64
        with self.assertRaisesRegex(ValueError,'token identity'):evaluation.compare_results(cases,self.plan)


if __name__=='__main__':unittest.main()
