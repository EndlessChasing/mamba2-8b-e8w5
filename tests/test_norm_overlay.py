"""CPU checks for the norm-only export boundary; no large model allocation."""
import tempfile
import copy
from pathlib import Path
import unittest

import torch

from mamba_e8w5.norm_overlay import (selected_norms, other_shapes, compare_other_tensors,
                                    checked_file, sha, verify_overlay, verify_final_checkpoint,
                                    PARENT_MANIFEST_SHA256)
from scripts.evaluate_norm_compensation import paired_dev_gate


class NormOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.parent = {name: torch.zeros(shape, dtype=torch.float16) for name, shape in other_shapes().items()}

    def test_inventory_is_existing_norms_only(self):
        selected = selected_norms()
        self.assertEqual(len(selected), 113)
        self.assertEqual(sum(entry['numel'] for entry in selected.values()), 692224)
        self.assertEqual(len(self.parent), 393)
        self.assertEqual(sum(value.numel() for value in self.parent.values()), 3580928)
        self.assertTrue(all(name.endswith('norm.weight') or name == 'backbone.norm_f.weight' for name in selected))

    def test_unchanged_selected_norms_are_allowed(self):
        result = compare_other_tensors(self.parent, self.parent)
        self.assertEqual(result['actual_changed_norm_tensor_count'], 0)
        self.assertEqual(result['protected_non_norm_tensor_count'], 280)

    def test_one_selected_norm_change_is_allowed(self):
        candidate = dict(self.parent)
        key = 'backbone.norm_f.weight'
        candidate[key] = candidate[key].clone()
        candidate[key][7] = 1
        result = compare_other_tensors(self.parent, candidate)
        self.assertEqual(result['actual_changed_norm_keys'], [key])

    def test_non_norm_signed_zero_change_is_rejected(self):
        candidate = dict(self.parent)
        key = 'backbone.layers.0.mixer.D'
        candidate[key] = candidate[key].clone()
        candidate[key][0] = -0.0
        self.assertTrue(torch.equal(candidate[key], self.parent[key]))
        with self.assertRaisesRegex(ValueError, 'Forbidden non-norm'):
            compare_other_tensors(self.parent, candidate)

    def test_dtype_shape_missing_extra_and_nonfinite_rejected(self):
        key = 'backbone.norm_f.weight'
        variants = []
        changed = dict(self.parent); changed[key] = changed[key].float(); variants.append(changed)
        changed = dict(self.parent); changed[key] = changed[key][:-1]; variants.append(changed)
        changed = dict(self.parent); del changed[key]; variants.append(changed)
        changed = dict(self.parent); changed['adapter.weight'] = torch.zeros(1).half(); variants.append(changed)
        changed = dict(self.parent); changed[key] = changed[key].clone(); changed[key][0] = float('nan'); variants.append(changed)
        for candidate in variants:
            with self.assertRaises(ValueError):
                compare_other_tensors(self.parent, candidate)

    def test_files_reject_escape_symlink_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'other_fp16.pt'; path.write_bytes(b'fixture')
            entry = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            self.assertEqual(checked_file(root, path.name, entry), path)
            with self.assertRaises(ValueError):
                checked_file(root, '../other_fp16.pt', entry)
            (root/'alias').symlink_to(path)
            with self.assertRaises(ValueError):
                checked_file(root, 'alias', entry)
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                checked_file(root, path.name, entry)

    def test_wrong_parent_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'original two-sweep'):
                verify_overlay(root, root)

    def test_export_must_match_final_fp32_masters(self):
        generator = torch.Generator().manual_seed(20260927)
        schedule = [index for _ in range(4) for index in torch.randperm(32, generator=generator).tolist()]
        checkpoint = {'format': 'MAMBA2_NORM_TRAINING_CHECKPOINT_V1', 'binding': {'fixture': True},
                      'parent_manifest_sha256': PARENT_MANIFEST_SHA256, 'smoke_report_sha256': 'fixture',
                      'schedule': schedule, 'successful_updates': 128, 'attempts': 129, 'overflow_retries': 1,
                      'masters': {name: self.parent[name].float() for name in selected_norms()}}
        verify_final_checkpoint(checkpoint, self.parent, {'fixture': True}, 'fixture')
        checkpoint['successful_updates'] = 127
        with self.assertRaisesRegex(ValueError, 'update accounting'):
            verify_final_checkpoint(checkpoint, self.parent, {'fixture': True}, 'fixture')
        checkpoint['successful_updates'] = 128
        checkpoint['masters']['backbone.norm_f.weight'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'rounded final FP32 master'):
            verify_final_checkpoint(checkpoint, self.parent, {'fixture': True}, 'fixture')

    def test_paired_gate_boundary_and_prompt_mismatch(self):
        rows = [{'id': str(i), 'condition': 'normal' if i < 12 else 'target_removed',
                 'correct': i < 10, 'prompt_token_sha256_int64le': str(i)} for i in range(24)]
        parent = {'ppl': {'ppl': 10., 'nll': 100., 'target_tokens': 100,
                          'windows': [{'start': 0, 'target_tokens': 100, 'token_sha256_int64le': 'tokens'}]},
                  'mk': {'rows': rows, 'summary': {'normal': {'correct': 10}, 'target_removed': {'correct': 0}}}}
        candidate = copy.deepcopy(parent); candidate['ppl']['ppl'] = 9.9
        self.assertTrue(paired_dev_gate(parent, candidate)['gate_passed'])
        candidate['ppl']['ppl'] = 9.90001
        self.assertFalse(paired_dev_gate(parent, candidate)['gate_passed'])
        candidate['ppl']['ppl'] = 9.8
        candidate['mk']['rows'][0]['correct'] = False
        candidate['mk']['summary']['normal']['correct'] = 9
        self.assertFalse(paired_dev_gate(parent, candidate)['gate_passed'])
        candidate['mk']['rows'][0]['prompt_token_sha256_int64le'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Paired MK input'):
            paired_dev_gate(parent, candidate)

    def test_removed_controls_may_not_increase(self):
        rows = [{'id': str(i), 'condition': 'normal' if i < 12 else 'target_removed',
                 'correct': i < 10, 'prompt_token_sha256_int64le': str(i)} for i in range(24)]
        parent = {'ppl': {'ppl': 10., 'nll': 100., 'target_tokens': 100, 'windows': []},
                  'mk': {'rows': rows, 'summary': {'normal': {'correct': 10}, 'target_removed': {'correct': 0}}}}
        candidate = copy.deepcopy(parent); candidate['ppl']['ppl'] = 9.0
        candidate['mk']['rows'][12]['correct'] = True
        candidate['mk']['summary']['target_removed']['correct'] = 1
        result = paired_dev_gate(parent, candidate)
        self.assertFalse(result['gate_passed'])
        self.assertFalse(result['target_removed_MK_gate_passed'])


if __name__ == '__main__':
    unittest.main()
