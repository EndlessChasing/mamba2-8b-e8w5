"""CPU integrity guards for experimental overlays; no GPU/model weights needed."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluate_refinement', ROOT/'scripts/evaluate_refinement.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OverlayGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.parent, self.overlay = root/'parent', root/'overlay'
        self.parent.mkdir(); self.overlay.mkdir()
        self.calibration = root/'calibration.json'
        self.protocol = root/'protocol.md'
        self.protocol.write_text('Synthetic predeclared protocol fixture')
        self.screen = root/'screen.json'
        config = {'n_layer': 56}
        labels = [f'layer{i}.{part}' for i in range(56) for part in ('in_proj', 'out_proj')]
        cal = {'complete': True, 'model_config': config, 'source_checkpoint_sha256': module.SOURCE_SHA,
               'dataset': {'tokenizer_sha256': module.TOKENIZER_SHA}, 'calibration_split': 'train',
               'evaluation_data_used': False, 'matrices': {label: {'sha256': 'a'*64} for label in labels}}
        self.calibration.write_text(json.dumps(cal))
        parent = {'complete': True, 'source_checkpoint_sha256': module.SOURCE_SHA,
                  'tokenizer_sha256': module.TOKENIZER_SHA, 'model_config': config,
                  'binding': {'hessian_manifest_sha256': module.sha(self.calibration),
                              'code_sha256': {name: module.sha(ROOT/'mamba_e8w5'/name) for name in ('codec.py', 'runtime.py', 'quantize.py')},
                              'quip_sha256': {name: module.sha(ROOT/'third_party/quip-sharp'/name) for name in
                                              ('lib/codebook/latticee8_padded12.py', 'lib/algo/quip.py')}}, 'files': {}, 'matrices': {}}
        overlay = {'format': module.FORMAT, 'complete': True, 'model_config': config, 'recipe': module.RECIPE,
                   'binding': {'source_checkpoint_sha256': module.SOURCE_SHA, 'tokenizer_sha256': module.TOKENIZER_SHA,
                               'hessian_manifest_sha256': module.sha(self.calibration),
                               'codec_source_sha256': module.sha(ROOT/'mamba_e8w5/codec.py'),
                               'runtime_source_sha256': module.sha(ROOT/'mamba_e8w5/runtime.py'),
                               'quip_sha256': parent['binding']['quip_sha256']}, 'files': {}, 'matrices': {}}
        for label in labels:
            filename = label+'.e8'
            for directory in (self.parent, self.overlay):
                (directory/filename).write_text(label)
            receipt = {'bytes': len(label), 'sha256': module.sha(self.overlay/filename)}
            parent['files'][filename] = receipt
            parent['matrices'][label] = {'decoded_fp16_sha256': 'a'*64}
            overlay['files'][filename] = receipt.copy()
            layer, family = label.split('.')
            overlay['matrices'][label] = {'file': filename, 'source_key': f'backbone.layers.{layer[5:]}.mixer.{family}.weight',
                'seed': 1000+2*int(layer[5:])+(family == 'out_proj'),
                'tune_iters': 8, 'scale_override': .9, 'damping': .01, 'index_bits': 16,
                'values_per_index': 8, 'axis_residual_amplitude': 0., 'feedback': True,
                'shape': [18560, 4096] if family == 'in_proj' else [4096, 8192],
                'hessian_sha256': 'a'*64, 'decoded_fp16_sha256': 'b'*64, **receipt}
        inherited = {}
        for filename in ('embedding.uniform', 'lm_head.uniform', 'other_fp16.pt', 'e8_codebook.bin', 'config.json'):
            (self.parent/filename).write_text('inherited '+filename)
            inherited[filename] = {'bytes': (self.parent/filename).stat().st_size, 'sha256': module.sha(self.parent/filename)}
        parent['files'].update(inherited)
        overlay['inherited_files'] = inherited
        (self.parent/'manifest.json').write_text(json.dumps(parent))
        overlay['parent_manifest_sha256'] = module.sha(self.parent/'manifest.json')
        screen = {'format': 'MAMBA2_E8W5_REFINEMENT_SCREEN_V1', 'complete': True,
                  'parent_manifest_sha256': overlay['parent_manifest_sha256'], 'predeclared_labels': module.SCREEN_LABELS,
                  'sweeps': [2, 8], 'expansion_gate': module.SCREEN_GATE, 'calibration_split': 'train', 'evaluation_data_used': False,
                  'binding': {**overlay['binding'], 'runtime_source_sha256': module.sha(ROOT/'mamba_e8w5/runtime.py'),
                              'scale': .9, 'damping': .01, 'tune_iters': 8},
                  'matrices': {}, 'comparisons': {}, 'gate_result': {'pass': True}}
        for label in module.SCREEN_LABELS:
            candidate = dict(overlay['matrices'][label], disk_roundtrip_fp16_equal=True,
                             original_hessian_weighted_squared_error=98.)
            baseline = dict(candidate, tune_iters=2, decoded_fp16_sha256='a'*64,
                            baseline_decoded_and_raw_sha_match=True, original_hessian_weighted_squared_error=100.)
            screen['matrices'][label] = {'2': baseline, '8': candidate}
            screen['comparisons'][label] = {'relative_output_mse_reduction': .02}
        self.screen.write_text(json.dumps(screen))
        overlay['binding']['screen_report_sha256'] = module.sha(self.screen)
        overlay['binding']['refinement_protocol_sha256'] = module.sha(self.protocol)
        self.manifest = overlay
        self.save()

    def save(self):
        (self.overlay/'manifest.json').write_text(json.dumps(self.manifest))

    def verify(self):
        # Header arithmetic is tested independently with real small binary files.
        with mock.patch.object(module, 'read_header', side_effect=lambda p: {'shape': [18560, 4096] if '.in_proj.' in p.name else [4096, 8192]}):
            return module.verify_overlay(self.parent, self.overlay, self.calibration, self.screen, self.protocol)

    def test_complete_ledger_accounting(self):
        _, _, receipt = self.verify()
        self.assertEqual(receipt['replacement_matrices'], 112)
        self.assertEqual(receipt['logical_candidate_data_bytes'], receipt['parent_verified_file_bytes'])
        self.assertGreater(receipt['combined_parent_plus_overlay_disk_bytes'], receipt['logical_candidate_data_bytes'])

    def test_reject_partial_screen(self):
        self.manifest['complete'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.verify()

    def test_reject_missing_projection(self):
        self.manifest['matrices'].pop('layer55.out_proj')
        self.save()
        with self.assertRaisesRegex(ValueError, 'exactly112'):
            self.verify()

    def test_reject_wrong_parent(self):
        self.manifest['parent_manifest_sha256'] = '0'*64
        self.save()
        with self.assertRaisesRegex(ValueError, 'another parent'):
            self.verify()

    def test_reject_calibration_substitution(self):
        self.manifest['matrices']['layer0.in_proj']['hessian_sha256'] = 'c'*64
        self.save()
        with self.assertRaisesRegex(ValueError, 'calibration identity'):
            self.verify()

    def test_reject_changed_payload(self):
        (self.overlay/'layer0.in_proj.e8').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.verify()

    def test_reject_missing_inherited_file(self):
        self.manifest['inherited_files'].pop('lm_head.uniform')
        self.save()
        with self.assertRaisesRegex(ValueError, 'Inherited'):
            self.verify()

    def test_reject_seed_change(self):
        self.manifest['matrices']['layer0.in_proj']['seed'] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.verify()

    def test_reject_new_recipe(self):
        self.manifest['recipe'] = dict(module.RECIPE, tune_iters=16)
        self.save()
        with self.assertRaisesRegex(ValueError, 'recipe'):
            self.verify()

    def test_reject_failed_screen_even_when_marked_passed(self):
        screen = json.loads(self.screen.read_text())
        for label in module.SCREEN_LABELS:
            screen['matrices'][label]['8']['original_hessian_weighted_squared_error'] = 99.9
            screen['comparisons'][label]['relative_output_mse_reduction'] = .001
        self.screen.write_text(json.dumps(screen))
        self.manifest['binding']['screen_report_sha256'] = module.sha(self.screen)
        self.save()
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            self.verify()

    def test_reject_unlisted_disk_entry(self):
        (self.overlay/'undeclared.bin').write_bytes(b'not counted')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            self.verify()

    def test_reject_protocol_substitution(self):
        self.protocol.write_text('Changed after screening')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.verify()

    def test_header_rejects_residual_bits_and_length_mismatch(self):
        header = {'format': 'mamba-e8-dct-hadamard-v1', 'rotation': 'dct2-kron-hadamard; maximal power-of-two factor',
                  'shape': [8, 8], 'index_order': 'row-major', 'axis_residual_amplitude': 0.0}
        path = Path(self.temp.name)/'tiny.e8'
        def write(extra=b''):
            data = json.dumps(header).encode()
            path.write_bytes(b'ME8HD001'+struct.pack('<I', len(data))+data+b'\0'*(8*8//4+2*8+1+1+4)+extra)
        write()
        self.assertEqual(module.read_header(path)['shape'], [8, 8])
        write(b'\0')
        with self.assertRaisesRegex(ValueError, 'length'):
            module.read_header(path)
        header['axis_residual_amplitude'] = 0.25
        write()
        with self.assertRaisesRegex(ValueError, 'two-bit'):
            module.read_header(path)


if __name__ == '__main__':
    unittest.main()
