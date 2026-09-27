"""Quantize the pinned 8B checkpoint; verify and hash every written component."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .codec import load_reference_primitives, vector_quantize, write_e8, read_e8, decode_e8, write_uniform
from .runtime import MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, load_source_state


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--hessian-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--quip-source', type=Path, default=Path('third_party/quip-sharp'))
    ap.add_argument('--scale', type=float, default=.9)
    ap.add_argument('--damping', type=float, default=.01)
    ap.add_argument('--tune-iters', type=int, default=2)
    ap.add_argument('--limit', type=int, default=112)
    ap.add_argument('--vocab-rows', type=int, default=256)
    args = ap.parse_args()
    assert 0 <= args.limit <= 112
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_receipt = json.loads((args.source_dir/'source_manifest.json').read_text())
    assert source_receipt['complete']
    calibration = json.loads((args.hessian_dir/'manifest.json').read_text())
    assert calibration['complete'], 'Calibration is incomplete'
    assert calibration['source_checkpoint_sha256'] == source_receipt['checkpoint_sha256']
    assert calibration['source_checkpoint_sha256'] == SOURCE_CHECKPOINT_SHA256
    assert calibration['model_config'] == MODEL_CONFIG
    assert calibration['dataset']['tokenizer_sha256'] == source_receipt['tokenizer_sha256']
    assert calibration['evaluation_data_used'] is False
    assert calibration['calibration_split'] == 'train'
    assert len(calibration['matrices']) == 112
    assert calibration['runtime_source_sha256'] == sha(Path(__file__).with_name('runtime.py'))
    binding = dict(source=source_receipt, hessian_manifest_sha256=sha(args.hessian_dir/'manifest.json'),
                   scale=args.scale, damping=args.damping, tune_iters=args.tune_iters,
                   embedding_bits=5, group_size=128,
                   code_sha256={name: sha(Path(__file__).parent/name)
                                for name in ('quantize.py', 'codec.py', 'runtime.py')},
                   quip_sha256={name: sha(args.quip_source/name) for name in
                                ('lib/codebook/latticee8_padded12.py', 'lib/algo/quip.py')})
    manifest_path = args.out_dir/'manifest.json'
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text())
        assert report['binding'] == binding, 'Resume inputs/source differ; use a new output directory.'
        if report['complete']:
            for name, entry in report['files'].items():
                path = args.out_dir/name
                assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
            print('Existing complete package passed all component hashes.', flush=True)
            return
    else:
        report = dict(format='MAMBA2_E8W5_RAW_V1', complete=False, binding=binding,
                      source_checkpoint_sha256=source_receipt['checkpoint_sha256'],
                      tokenizer_sha256=source_receipt['tokenizer_sha256'],
                      model_config=MODEL_CONFIG, calibration=calibration,
                      matrices={}, vocabularies={}, files={}, parameter_mapping={},
                      note='Lossy E8/W5 quantization; each serialized quantized representation is read back exactly. Untied vocabulary matrices remain separate.')
        write_json(manifest_path, report)
    sd = load_source_state(args.source_dir)
    assert sum(t.numel() for t in sd.values()) == 8236999680
    cb, ldlq = load_reference_primitives(args.quip_source)
    cb = cb.cuda()
    codebook_path = args.out_dir/'e8_codebook.bin'
    expected_book = cb.grid_packed_abs.detach().cpu().numpy().astype('<i4').tobytes()
    if codebook_path.exists():
        assert codebook_path.read_bytes() == expected_book
    else:
        codebook_path.write_bytes(expected_book)
    with torch.inference_mode():
        for index in range(args.limit):
            layer, part = divmod(index, 2)
            name = ('in_proj', 'out_proj')[part]
            label = f'layer{layer}.{name}'
            source_key = f'backbone.layers.{layer}.mixer.{name}.weight'
            hessian_path = args.hessian_dir/f'{label}.pt'
            hessian_sha = sha(hessian_path)
            assert hessian_sha == calibration['matrices'][label]['sha256'], label
            target = args.out_dir/f'{label}.e8'
            if label in report['matrices']:
                old = report['matrices'][label]
                assert old['hessian_sha256'] == hessian_sha and sha(target) == old['sha256'], label
                continue
            assert not target.exists(), f'Unrecorded output exists: {target}'
            t0 = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            weight = sd[source_key].to('cuda')
            expected_shape = (18560, 4096) if part == 0 else (4096, 8192)
            assert tuple(weight.shape) == expected_shape
            assert calibration['matrices'][label]['shape'] == [expected_shape[1], expected_shape[1]]
            h = torch.load(hessian_path, map_location='cuda', weights_only=True)
            assert tuple(h.shape) == (weight.shape[1], weight.shape[1])
            restored, info, payload = vector_quantize(weight, h, cb, ldlq, 1000+index,
                                                     args.damping, args.scale, args.tune_iters)
            tmp = target.with_suffix('.e8.part')
            write_e8(tmp, payload, info)
            disk_decoded = decode_e8(read_e8(tmp), cb)
            assert torch.equal(restored, disk_decoded), f'E8 readback mismatch: {label}'
            decoded_sha = tensor_sha(disk_decoded)
            os.replace(tmp, target)
            info.update(source_key=source_key, shape=list(expected_shape), hessian_sha256=hessian_sha,
                        bytes=target.stat().st_size, sha256=sha(target), decoded_fp16_sha256=decoded_sha,
                        disk_roundtrip_fp16_equal=True, elapsed_seconds=time.monotonic()-t0,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            report['matrices'][label] = info
            report['parameter_mapping'][source_key] = target.name
            write_json(manifest_path, report)
            print(json.dumps(dict(matrix=index+1, total=112, name=label, **info)), flush=True)
            del weight, h, restored, disk_decoded, payload
            torch.cuda.empty_cache()
        if len(report['matrices']) != 112:
            print('Partial projection run complete; vocabulary/final manifest deferred.', flush=True)
            return
        for name, key in [('embedding', 'backbone.embedding.weight'), ('lm_head', 'lm_head.weight')]:
            target = args.out_dir/f'{name}.uniform'
            if name in report['vocabularies']:
                assert sha(target) == report['vocabularies'][name]['sha256']
                continue
            assert not target.exists(), f'Unrecorded output exists: {target}'
            assert tuple(sd[key].shape) == (256000, 4096)
            t0 = time.monotonic()
            tmp = target.with_suffix('.uniform.part')
            info = write_uniform(tmp, sd[key], bits=5, device='cuda', rows_per_chunk=args.vocab_rows)
            os.replace(tmp, target)
            entry = dict(source_key=key, shape=list(sd[key].shape), bytes=target.stat().st_size,
                         sha256=sha(target), elapsed_seconds=time.monotonic()-t0, codec_report=info)
            report['vocabularies'][name] = entry
            report['parameter_mapping'][key] = target.name
            write_json(manifest_path, report)
            print(json.dumps(dict(vocabulary=name, **entry)), flush=True)
        other = {key: value.half().clone() for key, value in sd.items() if key not in report['parameter_mapping']}
        other_path = args.out_dir/'other_fp16.pt'
        if other_path.exists():
            previous = torch.load(other_path, weights_only=True, map_location='cpu')
            assert previous.keys() == other.keys() and all(torch.equal(previous[k], other[k]) for k in other)
        else:
            torch.save(other, other_path)
        for key in other:
            report['parameter_mapping'][key] = other_path.name
        assert set(report['parameter_mapping']) == set(sd)
        write_json(args.out_dir/'config.json', MODEL_CONFIG)
        for path in sorted(args.out_dir.iterdir()):
            if path.name == 'manifest.json' or path.suffix in ('.part', '.tmp'):
                continue
            report['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        report['total_parameter_count'] = sum(t.numel() for t in sd.values())
        report['data_file_bytes'] = sum(e['bytes'] for e in report['files'].values())
        report['complete'] = True
        report['elapsed_seconds_last_run'] = time.monotonic()-started
        write_json(manifest_path, report)
    print(json.dumps(dict(complete=True, matrices=len(report['matrices']),
                         data_file_bytes=report['data_file_bytes'], manifest_sha256=sha(manifest_path))), flush=True)


if __name__ == '__main__':
    main()
