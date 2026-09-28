"""Strict attempt accounting for the fixed one-pass pure-teacher KL experiment."""
import copy
import unittest

from scripts.train_teacher_kl_compensation import accounting, training_schedule


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.schedule = training_schedule()
        self.rows = [{'attempt': i + 1, 'successful_updates_after': i + 1,
                      'overflow': False, 'schedule_entry': window, 'targets': 2047}
                     for i, window in enumerate(self.schedule)]

    def test_complete_exposures(self):
        self.assertEqual(accounting(self.rows, self.schedule), {
            'successful_updates': 448, 'attempts': 448, 'overflow_retries': 0,
            'successful_target_exposures': 917056, 'attempted_target_exposures': 917056})

    def test_retry_same_window_and_bound(self):
        failed = dict(self.rows[0], overflow=True, successful_updates_after=0)
        rows = [dict(failed) for _ in range(8)] + copy.deepcopy(self.rows)
        for number, row in enumerate(rows, 1):
            row['attempt'] = number
        counts = accounting(rows, self.schedule)
        self.assertEqual(counts['successful_updates'], 448)
        self.assertEqual(counts['overflow_retries'], 8)
        self.assertEqual(counts['attempted_target_exposures'], 456 * 2047)
        rows[0]['schedule_entry'] = self.schedule[1]
        with self.assertRaises(ValueError):
            accounting(rows, self.schedule)
        rows = [dict(failed) for _ in range(9)]
        for number, row in enumerate(rows, 1):
            row['attempt'] = number
        with self.assertRaises(ValueError):
            accounting(rows, self.schedule)

    def test_wrong_numeric_types_are_rejected(self):
        for field, value in [('attempt', 1.0), ('successful_updates_after', True),
                             ('overflow', 0), ('targets', 2047.0),
                             ('schedule_entry', float(self.schedule[0]))]:
            with self.subTest(field=field):
                row = dict(self.rows[0], **{field: value})
                with self.assertRaises(ValueError):
                    accounting([row], self.schedule)
        schedule = self.schedule.copy()
        schedule[0] = float(schedule[0])
        with self.assertRaises(ValueError):
            accounting([], schedule)

    def test_changed_or_repeated_schedule_is_rejected(self):
        changed=self.schedule.copy()
        changed[0],changed[1]=changed[1],changed[0]
        with self.assertRaises(ValueError):
            accounting([],changed)
        changed=self.schedule.copy()
        changed[0]=changed[1]
        with self.assertRaises(ValueError):
            accounting([],changed)
        self.assertEqual(sorted(self.schedule),list(range(448)))

    def test_no_extra_update_or_target(self):
        with self.assertRaises(ValueError):
            accounting(self.rows + [dict(self.rows[-1], attempt=449)], self.schedule)
        row = dict(self.rows[0], targets=2048)
        with self.assertRaises(ValueError):
            accounting([row], self.schedule)


if __name__ == '__main__':
    unittest.main()
