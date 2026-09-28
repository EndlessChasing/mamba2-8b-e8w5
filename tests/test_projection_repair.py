"""Check fitting math against augmented least squares, independent of E8."""
import unittest

import torch

from mamba_e8w5.projection_repair import anchored_ridge_target, fresh_train_starts


class ProjectionRepairTests(unittest.TestCase):
    def test_matches_augmented_lstsq_with_nonzero_means(self):
        torch.manual_seed(819)
        x = torch.randn(53, 12) + 2
        y = torch.randn(53, 7) - 3
        anchor = torch.randn(7, 12).half()
        h, k = x.T@x/len(x), y.T@x/len(x)
        target, receipt = anchored_ridge_target(h, k, anchor, row_chunk=3)
        root_lambda = (.01*float(h.diag().mean()))**.5
        design = torch.cat((x.double()/len(x)**.5, root_lambda*torch.eye(12, dtype=torch.float64)))
        response = torch.cat((y.double()/len(x)**.5, root_lambda*anchor.double().T))
        expected = torch.linalg.lstsq(design, response).solution.T
        torch.testing.assert_close(target.double(), expected, atol=2e-5, rtol=2e-4)
        self.assertLess(receipt['relative_normal_equation_residual'], 1e-4)

    def test_anchor_is_preserved_in_unobserved_direction(self):
        x = torch.tensor([[1., 0.], [2., 0.], [3., 0.]])
        y = torch.tensor([[7.], [14.], [21.]])
        anchor = torch.tensor([[1., 9.]], dtype=torch.float16)
        target, _ = anchored_ridge_target(x.T@x/3, y.T@x/3, anchor)
        self.assertAlmostEqual(float(target[0, 1]), 9., places=5)
        self.assertGreater(float(target[0, 0]), 6.9)

    def test_self_target_remains_anchor(self):
        torch.manual_seed(11)
        x, anchor = torch.randn(30, 8), torch.randn(5, 8).half()
        h = x.T@x/30
        target, _ = anchored_ridge_target(h, anchor.float()@h, anchor)
        torch.testing.assert_close(target, anchor.float(), atol=2e-6, rtol=1e-5)

    def test_windows_do_not_overlap_excluded_or_each_other(self):
        excluded = [0, 15000, 40000, 100000]
        fit, heldout, audit = fresh_train_starts(2048*200, excluded)
        self.assertEqual((len(fit), len(heldout)), (32, 16))
        intervals = sorted(fit+heldout)
        self.assertTrue(all(b-a >= 2048 for a, b in zip(intervals, intervals[1:])))
        for start in intervals:
            self.assertTrue(all(start+2048 <= old or old+2048 <= start for old in excluded))
        self.assertEqual(audit['overlap_with_excluded_tokens'], 0)

    def test_invalid_ridge_is_rejected(self):
        with self.assertRaises(ValueError):
            anchored_ridge_target(torch.eye(2), torch.ones(1, 2), torch.ones(1, 2), damping=.02)


if __name__ == '__main__':
    unittest.main()
