"""CPU-only rejection tests for final-step, provenance and FP16 export boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from mamba_e8w5 import small_overlay as overlay


class SmallOverlayBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.inventory = overlay.selected_inventory()
        cls.after = {name: torch.zeros(entry['shape'], dtype=torch.float16) for name, entry in cls.inventory.items()}
        cls.masters = {name: value.float() for name, value in cls.after.items()}
        cls.schedule = overlay.training_schedule()
        cls.history = [{'attempt': n+1, 'successful_updates_after': n+1,
                        'window': window, 'targets': 2047, 'overflow': False}
                       for n, window in enumerate(cls.schedule)]
        cls.binding = {'fixed_recipe': 'test fixture'}
        cls.training = {'smoke_report_sha256': 'a'*64}

    def state(self):
        return {'format': 'MAMBA2_SMALL_TRAINING_CHECKPOINT_V1', 'resumable': True,
                'binding': self.binding, 'parent_manifest_sha256': overlay.norm.PARENT_MANIFEST_SHA256,
                'initialization': dict(overlay.INITIALIZATION), 'smoke_report_sha256': 'a'*64,
                'successful_updates': 1024, 'attempts': 1024, 'overflow_retries': 0,
                'schedule': list(self.schedule), 'history': copy.deepcopy(self.history),
                'masters': dict(self.masters)}

    def verify(self, state):
        return overlay.verify_final_checkpoint(state, self.after, self.binding, self.training)

    def test_valid_complete393_checkpoint(self):
        result = self.verify(self.state())
        self.assertEqual(result['rounded_master_tensors_verified'], 393)
        self.assertEqual(result['rounded_master_parameters_verified'], 3580928)
        self.assertEqual(result['committed_history_rows_verified'], 1024)

    def test_reject_noninteger_or_early_progress(self):
        for field, value in [('successful_updates', 1023), ('successful_updates', 1024.0),
                             ('attempts', 1024.0), ('overflow_retries', False),
                             ('overflow_retries', 9)]:
            with self.subTest(field=field, value=value):
                state = self.state(); state[field] = value
                with self.assertRaises(ValueError): self.verify(state)

    def test_reject_attempt_history_mismatch(self):
        changes = [('attempt', 2), ('window', (self.schedule[0]+1)%256),
                   ('targets', 2048), ('overflow', 0), ('successful_updates_after', 0)]
        for field, value in changes:
            with self.subTest(field=field):
                state = self.state(); state['history'][0][field] = value
                with self.assertRaises(ValueError): self.verify(state)
        state = self.state(); state['history'].pop()
        with self.assertRaises(ValueError): self.verify(state)

    def test_reject_bool_disguised_as_schedule_zero(self):
        state = self.state(); state['schedule'][state['schedule'].index(0)] = False
        with self.assertRaises(ValueError): self.verify(state)

    def test_overflow_retry_uses_same_scheduled_window(self):
        state = self.state(); state.update(attempts=1025, overflow_retries=1)
        retry = {'attempt': 1, 'successful_updates_after': 0, 'window': self.schedule[0], 'targets': 2047, 'overflow': True}
        state['history'] = [retry]+[dict(row, attempt=row['attempt']+1) for row in self.history]
        self.assertEqual(self.verify(state)['overflow_retries'], 1)
        state['history'][1]['window'] = self.schedule[1]
        with self.assertRaises(ValueError): self.verify(state)

    def test_reject_wrong_checkpoint_provenance_or_failure_snapshot(self):
        for field, value in [('smoke_report_sha256', 'b'*64), ('binding', {'different': True}),
                             ('parent_manifest_sha256', 'b'*64), ('initialization', {}), ('resumable', False)]:
            with self.subTest(field=field):
                state = self.state(); state[field] = value
                with self.assertRaises(ValueError): self.verify(state)

    def test_reject_master_export_mismatch_missing_dtype_and_nonfinite(self):
        name = 'backbone.norm_f.weight'
        for mode in ['changed', 'missing', 'dtype', 'nonfinite']:
            with self.subTest(mode=mode):
                state = self.state()
                if mode == 'missing': del state['masters'][name]
                elif mode == 'dtype': state['masters'][name] = self.after[name]
                else:
                    value = self.masters[name].clone(); value[0] = 1 if mode == 'changed' else float('nan')
                    state['masters'][name] = value
                with self.assertRaises(ValueError): self.verify(state)

    def test_signed_zero_export_is_bitwise_checked(self):
        state = self.state(); name = 'backbone.norm_f.weight'
        value = self.masters[name].clone(); value[0] = -0.0; state['masters'][name] = value
        with self.assertRaises(ValueError): self.verify(state)

    def test_report_exposures_are_integer_and_exact(self):
        state = self.state(); state.update(final_step=1024, successful_target_exposures=2096128,
                                           attempted_target_exposures=2096128)
        overlay.verify_accounting(state, exposures=True)
        for field, value in [('final_step', 1024.0), ('successful_target_exposures', 2097152),
                             ('attempted_target_exposures', 2096128.0)]:
            changed = dict(state); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                overlay.verify_accounting(changed, exposures=True)

    def test_reject_changed_exclusion_or_validation_data_even_if_manifest_digest_matches(self):
        fixture = json.loads((overlay.ROOT/'reports/small_compensation_v1_data.json').read_text())
        for mode in ['exclusion', 'split']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                folder = Path(td); manifest = copy.deepcopy(fixture)
                if mode == 'exclusion': manifest['excluded_original_calibration_starts'][0] = 1
                else: manifest['dataset']['split'] = 'validation'
                mp = folder/'manifest.json'; mp.write_text(json.dumps(manifest))
                (folder/'training_tokens.pt').write_bytes(b'not reached')
                with patch.object(overlay, 'DATA_MANIFEST_SHA256', overlay.sha(mp)):
                    with self.assertRaises(ValueError): overlay.verify_training_data(folder)

    def test_reject_file_traversal_symlink_and_noninteger_size(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td); path = folder/'weights'; path.write_bytes(b'abc')
            entry = {'bytes': 3, 'sha256': overlay.sha(path)}
            self.assertEqual(overlay.checked_file(folder, 'weights', entry), path)
            with self.assertRaises(ValueError): overlay.checked_file(folder, '../weights', entry)
            (folder/'linked').symlink_to(path)
            with self.assertRaises(ValueError): overlay.checked_file(folder, 'linked', entry)
            with self.assertRaises(ValueError): overlay.checked_file(folder, 'weights', dict(entry, bytes=3.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
