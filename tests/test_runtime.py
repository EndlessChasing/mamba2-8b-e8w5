"""Architecture, token protocol, and small grouped-recurrence numerical checks."""
import json
import unittest

import torch
import torch.nn.functional as F

from mamba_e8w5.calibration import window_starts
from mamba_e8w5.evaluation import ppl_windows, synthetic_mk_cases
from mamba_e8w5.runtime import MODEL_CONFIG, make_model, normalize_source_key, normalize_source_state


class ProtocolTests(unittest.TestCase):
    def test_checkpoint_key_mapping_is_anchored(self):
        self.assertEqual(normalize_source_key("decoder.layers.10.mixer.in_proj.weight"),
                         "backbone.layers.10.mixer.in_proj.weight")
        self.assertEqual(normalize_source_key("output_layer.weight"), "lm_head.weight")
        self.assertEqual(normalize_source_key("embedding.word_embeddings.weight"), "backbone.embedding.weight")
        with self.assertRaises(ValueError):
            normalize_source_key("unrecognized.weight")
        with self.assertRaises(ValueError):
            normalize_source_state({"backbone.embedding.weight": torch.zeros(1),
                                    "embedding.word_embeddings.weight": torch.zeros(1)})

    def test_nonoverlapping_calibration_and_complete_ppl(self):
        starts = window_starts(200000, 2048, 32)
        self.assertEqual(len(starts), 32)
        self.assertTrue(all(b-a >= 2048 for a, b in zip(starts, starts[1:])))
        with self.assertRaises(ValueError):
            window_starts(64, 32, 3)
        ids = torch.arange(103)
        windows = ppl_windows(ids, 32)
        targets = torch.cat([window[1:] for _, window in windows])
        self.assertTrue(torch.equal(targets, ids[1:]))

    def test_mk_disjoint_and_removed_controls(self):
        dev = synthetic_mk_cases("validation", samples_per_cell=2)
        test = synthetic_mk_cases("test", samples_per_cell=2)
        self.assertFalse({row["key"] for row in dev} & {row["key"] for row in test})
        self.assertFalse({row["answer"] for row in dev} & {row["answer"] for row in test})
        for row in dev+test:
            if row["condition"] == "target_removed":
                self.assertNotIn(row["answer"], row["prompt"])
            else:
                self.assertIn(row["answer"], row["prompt"])

    def test_native_meta_shape_and_grouped_norm(self):
        model = make_model()
        self.assertEqual(len(model.backbone.layers), 56)
        self.assertEqual(tuple(model.backbone.layers[0].mixer.in_proj.weight.shape), (18560, 4096))
        self.assertEqual(tuple(model.backbone.layers[0].mixer.out_proj.weight.shape), (4096, 8192))
        self.assertEqual(model.backbone.layers[0].mixer.norm.group_size, 1024)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8236999680)
        self.assertIsNot(model.backbone.embedding.weight, model.lm_head.weight)


@unittest.skipUnless(torch.cuda.is_available(), "Small numerical test requires a CUDA runtime")
class GroupedNumericalTests(unittest.TestCase):
    @torch.inference_mode()
    def test_grouped_ssd_matches_independent_recurrence_256_tokens(self):
        """Check ngroups=8 against direct equations, not another wrapper of SSD."""
        from mamba_ssm.modules.mamba2 import Mamba2
        torch.manual_seed(9843)
        torch.backends.cuda.matmul.allow_tf32 = False
        mixer = Mamba2(128, d_state=16, headdim=16, ngroups=8,
                       chunk_size=32, use_mem_eff_path=False,
                       device="cuda", dtype=torch.float32).eval()
        u = torch.randn(1, 256, 128, device="cuda") * .3
        actual = mixer(u)
        z, xbc, dt = F.linear(u, mixer.in_proj.weight).split(
            [mixer.d_inner, mixer.d_inner+2*mixer.ngroups*mixer.d_state, mixer.nheads], -1)
        xbc = F.silu(F.conv1d(xbc.transpose(1, 2), mixer.conv1d.weight,
                            mixer.conv1d.bias, padding=mixer.d_conv-1,
                            groups=mixer.conv1d.groups)[..., :u.shape[1]].transpose(1, 2))
        x, b, c = xbc.split([mixer.d_inner, mixer.ngroups*mixer.d_state,
                             mixer.ngroups*mixer.d_state], -1)
        x = x.reshape(1, -1, mixer.nheads, mixer.headdim)
        b = b.reshape(1, -1, mixer.ngroups, mixer.d_state).repeat_interleave(
            mixer.nheads//mixer.ngroups, dim=2)
        c = c.reshape(1, -1, mixer.ngroups, mixer.d_state).repeat_interleave(
            mixer.nheads//mixer.ngroups, dim=2)
        dt = F.softplus(dt.float()+mixer.dt_bias.float())
        a = -mixer.A_log.float().exp()
        state = torch.zeros(1, mixer.nheads, mixer.headdim, mixer.d_state, device="cuda")
        outputs = []
        for position in range(u.shape[1]):
            decay = (dt[:, position]*a).exp()
            state = state*decay[..., None, None] + (
                dt[:, position, :, None, None] * x[:, position, :, :, None] * b[:, position, :, None, :])
            output = (state*c[:, position, :, None, :]).sum(-1) + mixer.D[None, :, None]*x[:, position]
            outputs.append(output.flatten(1))
        y = torch.stack(outputs, 1)*F.silu(z)
        grouped = y.reshape(1, u.shape[1], mixer.ngroups, mixer.d_inner//mixer.ngroups)
        y = (grouped*torch.rsqrt(grouped.square().mean(-1, keepdim=True)+mixer.norm.eps)).reshape_as(y)
        expected = F.linear(y*mixer.norm.weight, mixer.out_proj.weight)
        relative = float((actual-expected).norm()/expected.norm())
        print(json.dumps({"test": "independent_grouped_recurrence_256", "relative_error": relative}))
        self.assertLess(relative, 5e-5)


if __name__ == "__main__":
    unittest.main()
