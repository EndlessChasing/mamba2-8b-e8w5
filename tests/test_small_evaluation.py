"""CPU-only evidence checks for the independent full-validation evaluator."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('small_evaluation', ROOT/'scripts/evaluate_small_compensation.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class SmallEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.windows = evaluation.ppl_windows(torch.arange(evaluation.TARGETS+1), 2048)
        cls.plan = evaluation.window_plan(cls.windows)

    def result(self, ppl):
        # Distinct window NLLs exercise target weighting, including the last572.
        rows = [{**row, 'nll': row['target_tokens']*math.log(ppl)} for row in self.plan]
        for row in rows:
            row['ppl'] = math.exp(row['nll']/row['target_tokens'])
        nll = sum(row['nll'] for row in rows)
        return {'windows': rows, 'nll': nll, 'ppl': math.exp(nll/evaluation.TARGETS),
            'target_tokens': evaluation.TARGETS, 'execution': 'prefill', 'logits_chunk_tokens': 64}

    def test_full_coverage_and_last_window(self):
        self.assertEqual(len(self.plan), 130)
        self.assertEqual(self.plan[-1]['target_tokens'], 572)
        self.assertEqual(sum(row['target_tokens'] for row in self.plan), 264764)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            evaluation.window_plan([(1, self.windows[0][1]), *self.windows[1:]])
        with self.assertRaisesRegex(ValueError, 'complete validation'):
            evaluation.window_plan(self.windows[:-1])

    def test_modified_tokens_or_execution_rejected(self):
        good = self.result(8)
        evaluation.check_arm(good, self.plan)
        for key, value in [('execution', 'tokenwise'), ('target_tokens', 4096), ('logits_chunk_tokens', 32)]:
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.assertRaises(ValueError):
                evaluation.check_arm(bad, self.plan)
        bad = copy.deepcopy(good)
        bad['windows'][4]['token_sha256_int64le'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'token identity'):
            evaluation.check_arm(bad, self.plan)

    def test_weighted_aggregate_and_finite_checks(self):
        result = self.result(8)
        result['windows'][-1]['nll'] *= 2
        row = result['windows'][-1]
        row['ppl'] = math.exp(row['nll']/row['target_tokens'])
        with self.assertRaisesRegex(ValueError, 'Aggregate NLL'):
            evaluation.check_arm(result, self.plan)
        result['nll'] = sum(row['nll'] for row in result['windows'])
        result['ppl'] = math.exp(result['nll']/evaluation.TARGETS)
        evaluation.check_arm(result, self.plan)
        result['ppl'] = sum(row['ppl'] for row in result['windows'])/130
        with self.assertRaisesRegex(ValueError, 'target-weighted'):
            evaluation.check_arm(result, self.plan)
        result = self.result(8)
        result['windows'][0]['nll'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            evaluation.check_arm(result, self.plan)

    def test_predeclared_comparison_thresholds(self):
        results = {name: self.result(ppl) for name, ppl in zip(evaluation.ARMS, [7, 9, 8, 7.91999])}
        comparison, rows = evaluation.comparisons(results, self.plan)
        self.assertTrue(comparison['meaningful_improvement_reference']['met'])
        self.assertFalse(comparison['source_plus5_percent_reference']['met'])
        self.assertEqual(len(rows), 130)
        results['small_e8w5'] = self.result(7.92001)
        self.assertFalse(evaluation.comparisons(results, self.plan)[0]['meaningful_improvement_reference']['met'])
        results['small_e8w5'] = self.result(7.34999)
        self.assertTrue(evaluation.comparisons(results, self.plan)[0]['source_plus5_percent_reference']['met'])
        results['small_e8w5'] = self.result(7.35001)
        self.assertFalse(evaluation.comparisons(results, self.plan)[0]['source_plus5_percent_reference']['met'])
        del results['parent_e8w5']
        with self.assertRaisesRegex(ValueError, 'four same-process'):
            evaluation.comparisons(results, self.plan)

    def test_loaded_and_frozen_byte_hash_checks(self):
        model = torch.nn.Linear(4, 2).half()
        values = {name: value.detach().clone() for name, value in model.named_parameters()}
        receipt = evaluation.audit_selected_values(model, values)
        hashes = receipt['loaded_and_export_fp16_sha256']
        self.assertEqual(evaluation.audit_frozen_values(model, hashes)['verified_tensor_count'], 2)
        with torch.no_grad():
            model.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            evaluation.audit_selected_values(model, values)
        with self.assertRaisesRegex(ValueError, 'changed'):
            evaluation.audit_frozen_values(model, hashes)
        # Signed zero must also be detectable through canonical bytes.
        model.bias.data.zero_()
        values = {'bias': torch.full_like(model.bias, -0.0)}
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            evaluation.audit_selected_values(model, values)

    def test_independent_final_checkpoint_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'final.pt'
            master = torch.tensor([1.0001, -0.0], dtype=torch.float32)
            values = {'fixture': master.half()}
            state = {'format': 'MAMBA2_SMALL_TRAINING_CHECKPOINT_V1', 'successful_updates': 1024,
                     'binding': {'fixture': 'bound'}, 'masters': {'fixture': master}}
            torch.save(state, path)
            result = evaluation.audit_checkpoint_export(path, values, state['binding'], evaluation.sha(path))
            self.assertEqual(result['verified_tensor_count'], 1)
            values['fixture'][0] += 1
            with self.assertRaisesRegex(ValueError, 'differs from export'):
                evaluation.audit_checkpoint_export(path, values, state['binding'], evaluation.sha(path))
            state['successful_updates'] = 1023
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'final1024'):
                evaluation.audit_checkpoint_export(path, values, state['binding'], evaluation.sha(path))


if __name__ == '__main__':
    unittest.main()
