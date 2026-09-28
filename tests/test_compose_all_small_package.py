"""Focused CPU checks for resolved package identity and source preservation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('composer', ROOT/'scripts/compose_all_small_package.py')
composer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composer)


class CompositionChecks(unittest.TestCase):
    def test_tensor_identity_includes_signed_zero_and_dtype(self):
        values = {'a':torch.tensor([0., 1.], dtype=torch.float16)}
        inventory = {'a':{'shape':[2], 'numel':2}}
        hashes = {'a':composer.tensor_sha(values['a'])}
        self.assertEqual(composer.audit_small(values,inventory,hashes)['a']['numel'],2)
        for bad in ({'a':torch.tensor([-0.,1.],dtype=torch.float16)},
                    {'a':values['a'].float()}, {'a':values['a'].reshape(1,2)},
                    {'a':torch.tensor([float('nan'),1.],dtype=torch.float16)}, {}):
            with self.assertRaises(ValueError):composer.audit_small(bad,inventory,hashes)

    def test_hardlink_copy_and_no_overwrite_preserve_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory/'source'
            data = b'fixed inherited E8 bytes\x00\xff'
            source.write_bytes(data)
            entry = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            before = source.stat()
            for mode in ('hardlink','copy'):
                target = directory/mode
                self.assertEqual(composer.materialize(source,target,entry,mode),mode)
                self.assertEqual(target.read_bytes(),data)
                self.assertEqual(source.stat().st_mode,before.st_mode)
                self.assertEqual(source.stat().st_mtime_ns,before.st_mtime_ns)
                self.assertEqual(source.read_bytes(),data)
                self.assertEqual(source.stat().st_ino == target.stat().st_ino,mode == 'hardlink')
                with self.assertRaises(FileExistsError):composer.materialize(source,target,entry,mode)
            with self.assertRaises(ValueError):composer.check_file(directory/'copy',{**entry,'sha256':'0'*64})

    def test_flat_inventory_refuses_proto_extra_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'x').write_bytes(b'x')
            composer.check_inventory(directory,{'x'})
            for name in ('../x','/x',''):
                with self.assertRaises(ValueError):composer.plain_file(directory,name)
            (directory/'unsupported.proto').write_bytes(b'not supported')
            with self.assertRaises(ValueError):composer.check_inventory(directory,{'x'})
            (directory/'alias').symlink_to(directory/'x')
            with self.assertRaises(ValueError):composer.plain_file(directory,'alias')

    def test_fresh_manifest_omits_stale_parent_metrics_and_counts_itself(self):
        matrices, frozen, mapping, files = {}, {}, {}, {}
        for label in composer.LABELS:
            name = label+'.weight'
            matrices[label] = {'source_key':name,'shape':[8,8],'bytes':32,'sha256':'1'*64,
                'decoded_fp16_sha256':'2'*64,'index_bits':16,'values_per_index':8,
                'relative_weight_error':123,'elapsed_seconds':999}
            frozen[name] = '2'*64
            mapping[name] = label+'.e8'
            files[label+'.e8'] = {'bytes':32,'sha256':'1'*64}
        vocab = {}
        for label in ('embedding','lm_head'):
            name = label+'.weight'
            vocab[label] = {'source_key':name,'shape':[16,8],'bytes':80,'sha256':'3'*64,
                'codec_report':{'relative_weight_error':123}}
            frozen[name] = '4'*64
            mapping[name] = label+'.uniform'
            files[label+'.uniform'] = {'bytes':80,'sha256':'3'*64}
        files.update({'config.json':{'bytes':2,'sha256':'5'*64},
            'e8_codebook.bin':{'bytes':1024,'sha256':'6'*64},
            'other_fp16.pt':{'bytes':100,'sha256':composer.SMALL_VALUES_SHA}})
        records = {'norm':{'shape':[8],'numel':8,'dtype':'float16','decoded_fp16_sha256':'7'*64}}
        mapping['norm'] = 'other_fp16.pt'
        parent = {'matrices':matrices,'vocabularies':vocab,'parameter_mapping':mapping,
                  'binding':{'source':{'checkpoint_sha256':composer.SOURCE_CHECKPOINT_SHA256}},
                  'quality':{'ppl':123},'small_tensors':{'norm':{'decoded_fp16_sha256':'bad-old-small'}},
                  'elapsed_seconds_last_run':999}
        quality = {'results':{'small_e8w5':{'ppl':8.3,'nll':123}},'protocol':{},
                   'dataset':{},'window_plan_sha256':'8'*64}
        original = copy.deepcopy(parent)
        manifest = composer.make_manifest(parent,{},quality,files,records,frozen)
        serialized = composer.finalize_manifest(manifest)
        self.assertEqual(parent,original)
        self.assertEqual(manifest['files']['other_fp16.pt']['sha256'],composer.SMALL_VALUES_SHA)
        self.assertEqual(manifest['small_tensors'],records)
        self.assertEqual(manifest['decoded_fp16_sha256']['norm'],'7'*64)
        self.assertEqual(manifest['quality']['ppl'],8.3)
        self.assertEqual(manifest['binding']['code_sha256'], {
            name:composer.FROZEN_CODE['mamba_e8w5/'+name] for name in ('runtime.py','codec.py')})
        self.assertEqual(manifest['manifest_bytes'],len(serialized))
        self.assertEqual(manifest['raw_package_bytes'],sum(e['bytes'] for e in files.values())+len(serialized))
        for stale in ('bad-old-small','relative_weight_error','elapsed_seconds','codec_report'):
            self.assertNotIn(stale,serialized.decode())
        self.assertEqual(manifest['composition']['prototype_tables'],0)

    @unittest.skipUnless((ROOT/'artifacts/small_compensation_v1/other_fp16.pt').is_file(),
                         'Real small artifact exists on the GPU host only')
    def test_actual_393_export_matches_completed_validation(self):
        path = ROOT/'artifacts/small_compensation_v1/other_fp16.pt'
        report_path = ROOT/'reports/small_compensation_v1_eval.json'
        self.assertEqual(composer.sha(path),composer.SMALL_VALUES_SHA)
        self.assertEqual(composer.sha(report_path),composer.QUALITY_SHA)
        report = json.loads(report_path.read_text())
        values = torch.load(path,map_location='cpu',weights_only=True)
        records = composer.audit_small(values,composer.selected_inventory(),
            report['small_loaded_tensor_audit']['small']['loaded_and_export_fp16_sha256'])
        self.assertEqual(len(records),393)
        self.assertEqual(sum(row['numel'] for row in records.values()),3580928)
        self.assertEqual(sum(row['numel']*2 for row in records.values()),7161856)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
