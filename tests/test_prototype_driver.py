"""Bounded CPU checks for fixed scheduling and actual prepared-data provenance."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import train_prototype_compensation as driver


def history_with_overflow(schedule):
    result = []
    # One overflow at the first window, then128 successful scheduled updates.
    result.append({'attempt':1,'successful_updates_after':0,'overflow':True,
                   'schedule_entry':schedule[0],'targets':2047})
    for success, entry in enumerate(schedule, 1):
        result.append({'attempt':success+1,'successful_updates_after':success,
                       'overflow':False,'schedule_entry':entry,'targets':2047})
    return result


class PrototypeDriverTests(unittest.TestCase):
    def test_fixed_schedule_and_exposure_accounting(self):
        schedule = driver.training_schedule()
        generator = torch.Generator(device='cpu').manual_seed(20260927)
        expected = [index for _ in range(4) for index in torch.randperm(32,generator=generator).tolist()]
        self.assertEqual([row['window'] for row in schedule], expected)
        result = driver.accounting(history_with_overflow(schedule), schedule)
        self.assertEqual(result, {'successful_updates':128,'attempts':129,'overflow_retries':1,
            'successful_target_exposures':262016,'attempted_target_exposures':264063})

    def test_reject_advanced_retry_and_noninteger_counter(self):
        schedule = driver.training_schedule()
        history = history_with_overflow(schedule)
        wrong = copy.deepcopy(history)
        wrong[1]['schedule_entry'] = schedule[1]
        with self.assertRaises(ValueError):
            driver.accounting(wrong, schedule)
        wrong = copy.deepcopy(history)
        wrong[0]['attempt'] = True
        with self.assertRaises(ValueError):
            driver.accounting(wrong, schedule)

    def test_actual_prepared_data_and_all_exclusions(self):
        args = SimpleNamespace(parent_dir=ROOT/'artifacts/e8w5_v1',
            small_dir=ROOT/'artifacts/small_compensation_v1',
            data_dir=ROOT/'training_data/prototype_compensation_v1',
            protocol=ROOT/'docs/PROTOTYPE_COMPENSATION_PROTOCOL.md',
            protocol_sha256=driver.PROTOCOL_SHA)
        _raw,_small,_baseline,data,windows,binding,_paths = driver.verify_inputs(args)
        self.assertEqual(tuple(windows.shape), (32,2048))
        self.assertEqual(binding['training_data_manifest_sha256'],driver.DATA_SHA)
        self.assertEqual(data['overlap']['eligible_blocks'],871)
        self.assertEqual(data['overlap']['all_previous_fitting_overlap_tokens'],0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
