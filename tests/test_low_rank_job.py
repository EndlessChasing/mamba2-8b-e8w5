"""Pure-stdlib launcher gates and read-only process/progress status checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import check_low_rank_job as checker
from scripts import run_low_rank_compensation_job as runner


class LowRankJobTests(unittest.TestCase):
    def fake_process(self, root, *, start=500, state='S', argv=None, boot='boot-a'):
        argv = argv or ['/venv/python', '-u', '/repo/scripts/train_low_rank_compensation.py']
        directory = root / '42'
        directory.mkdir(parents=True, exist_ok=True)
        # Explicit Linux stat field positions: state is3, starttime is22.
        fields = [state] + ['0'] * 18 + [str(start)] + ['0'] * 5
        (directory / 'stat').write_text('42 (name with spaces(and)parens) ' + ' '.join(fields))
        (directory / 'cmdline').write_bytes(b'\0'.join(s.encode() for s in argv) + b'\0')
        boot_path = root / 'sys/kernel/random/boot_id'
        boot_path.parent.mkdir(parents=True, exist_ok=True)
        boot_path.write_text(boot + '\n')

    def test_pending_pins_fail_before_opening_inputs(self):
        with patch.object(runner, 'sha', side_effect=AssertionError('Must not open inputs')):
            with self.assertRaisesRegex(ValueError, 'PENDING'):
                runner.verify_pins(Path('/missing'), {'trainer': 'a' * 64, 'eval': 'PENDING'})

    def test_valid_pins_are_verified_against_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'fake.py'; path.write_text('fixed bytes')
            pins = {'fake.py': runner.sha(path)}
            runner.verify_pins(root, pins)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'differs'):
                runner.verify_pins(root, pins)

    def test_proc_identity_requires_starttime_boot_and_exact_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fake_process(root)
            identity = checker.read_process(42, proc_root=root)
            self.assertEqual(identity['start_ticks'], 500)
            self.assertTrue(checker.verified_process(identity, proc_root=root)['matching_live_process'])
            for changed in ({'start': 501}, {'boot': 'boot-b'},
                    {'argv': ['/venv/python', '-u', '/repo/scripts/train_low_rank_compensation.py.extra']}):
                self.fake_process(root, **changed)
                self.assertFalse(checker.verified_process(identity, proc_root=root)['matching_live_process'])
            self.fake_process(root, state='Z')
            observed = checker.verified_process(identity, proc_root=root)
            self.assertTrue(observed['identity_matches'])
            self.assertFalse(observed['matching_live_process'])

    def test_missing_proc_is_unknown_not_proof_process_stopped(self):
        observed = checker.read_process(123, proc_root=Path('/definitely-not-an-existing-proc-mount'))
        self.assertIsNone(observed['present'])
        self.assertEqual(observed['reason'], 'proc_unavailable')

    def test_progress_resets_at_new_arm_and_uses_complete_receipt(self):
        prefix = '[low-rank evaluation] Scoring source_fp16\n[PPL prefill] 128/130, ppl=7.334, 20.0s\n'
        log = prefix + '[low-rank evaluation] Scoring best_small_e8w5\n'
        progress = checker.evaluation_progress({}, log)
        self.assertEqual(progress['current_or_last_started_arm'], 'best_small_e8w5')
        self.assertEqual(progress['current_arm_completed_windows'], 0)
        log += '[PPL prefill] 4/130, ppl=8.12, 1.0s\n'
        self.assertEqual(checker.evaluation_progress({}, log)['current_arm_completed_windows'], 4)
        evaluated = {'results': {'best_small_e8w5': {'ppl': 8.3, 'windows': [{}] * 130}}}
        self.assertEqual(checker.evaluation_progress(evaluated, log)['current_arm_completed_windows'], 130)

    def test_fixed_checkpoint_gate_and_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / 'work'; work.mkdir()
            checkpoint = work / 'final.pt'; checkpoint.write_bytes(b'checkpoint')
            training = {'complete': True, 'mode': 'train', 'successful_updates': 1024, 'final_step': 1024,
                'overflow_retries': 2, 'attempts': 1026, 'successful_target_exposures': 2096128,
                'final_checkpoint': {'file': str(checkpoint), 'sha256': runner.sha(checkpoint)}}
            self.assertEqual(runner.final_checkpoint(training, work), checkpoint.resolve())
            for field, value in [('successful_updates', 1023), ('successful_updates', 1024.),
                    ('attempts', 1025), ('overflow_retries', 9), ('complete', False),
                    ('successful_target_exposures', 2096128.)]:
                bad = copy.deepcopy(training); bad[field] = value
                with self.assertRaises(RuntimeError):
                    runner.final_checkpoint(bad, work)
            outside = root / 'outside.pt'; outside.write_bytes(b'checkpoint')
            training['final_checkpoint']['file'] = str(outside)
            with self.assertRaisesRegex(RuntimeError, 'inside'):
                runner.final_checkpoint(training, work)

    def test_final_evaluation_requires_all_full_arms_and_finite_ppl(self):
        value = {'target_tokens': 264764, 'windows': [{}] * 130, 'ppl': 8.0}
        evaluation = {'complete': True, 'mode': 'three_arm_full_validation',
            'protocol': {'target_tokens': 264764, 'windows': 130}, 'comparison': {},
            'results': {name: copy.deepcopy(value) for name in ('source_fp16','best_small_e8w5','low_rank_e8w5')}}
        runner.check_evaluation(evaluation)
        evaluation['results']['low_rank_e8w5']['ppl'] = float('nan')
        with self.assertRaises(RuntimeError):
            runner.check_evaluation(evaluation)
        evaluation['results']['low_rank_e8w5']['ppl'] = 8.
        evaluation['results']['wrong_candidate'] = evaluation['results'].pop('low_rank_e8w5')
        with self.assertRaises(RuntimeError):
            runner.check_evaluation(evaluation)

    def test_snapshot_does_not_create_a_missing_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'job.json'
            self.assertEqual(checker.snapshot(path)['status'], 'not_started')
            self.assertFalse(path.exists())

    def test_nonzero_child_exit_is_saved_and_blocks_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'train.log'
            process = Mock(pid=42, returncode=7)
            process.wait.return_value = 7
            report = {'stages': []}; saved = []
            with patch.object(runner, 'verify_pins'), patch.object(runner, 'read_process', return_value={'pid': 42}), \
                    patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
                with self.assertRaisesRegex(RuntimeError, 'exited7'):
                    runner.run_stage('training', ['fake', 'training'], log, report,
                        lambda: saved.append(copy.deepcopy(report)), root=Path(directory))
                popen.assert_called_once()
                process.terminate.assert_not_called()
            self.assertEqual(saved[-1]['stages'][0]['exit_code'], 7)
            self.assertIn('ended_at_unix', saved[-1]['stages'][0])
            self.assertEqual(len(report['stages']), 1)

    def test_existing_log_blocks_process_launch_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'train.log'; log.write_text('old evidence')
            report = {'stages': []}
            with patch.object(runner, 'verify_pins'), patch.object(runner.subprocess, 'Popen') as popen:
                with self.assertRaises(FileExistsError):
                    runner.run_stage('training', ['fake'], log, report, lambda: None, root=Path(directory))
                popen.assert_not_called()
            self.assertEqual(log.read_text(), 'old evidence')

    def test_interrupted_runner_terminates_its_child_and_records_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=42, returncode=-15)
            process.wait.side_effect = [runner.RequestedStop('test'), -15]
            process.poll.return_value = None
            report = {'stages': []}
            with patch.object(runner, 'verify_pins'), patch.object(runner, 'read_process', return_value={'pid': 42}), \
                    patch.object(runner.subprocess, 'Popen', return_value=process):
                with self.assertRaises(runner.RequestedStop):
                    runner.run_stage('training', ['fake'], Path(directory) / 'log', report, lambda: None, root=Path(directory))
                process.terminate.assert_called_once()
            self.assertEqual(report['stages'][0]['exit_code'], -15)
            self.assertTrue(report['stages'][0]['interrupted_or_launch_failed'])


if __name__ == '__main__':
    unittest.main()
