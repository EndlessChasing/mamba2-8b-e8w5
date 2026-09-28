#!/usr/bin/env python3
"""Run the restored package with its archived software and restored tokenizer.

This is an end-user loading/generation check, not a new quality benchmark.
No source checkpoint or Hessian is an inference input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-dir', type=Path, required=True)
    ap.add_argument('--restored-dir', type=Path, required=True)
    ap.add_argument('--expected-manifest-sha256', required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    release, restored = args.release_dir.resolve(), args.restored_dir.resolve()
    manifest_path = release/'release_manifest.json'
    assert sha(manifest_path) == args.expected_manifest_sha256
    manifest = json.loads(manifest_path.read_text())
    restore = json.loads((restored/'restore_receipt.json').read_text())
    assert restore['complete'] and restore['manifest_sha256'] == args.expected_manifest_sha256
    archive = release/'source.zip'
    asset = next(a for a in manifest['assets'] if a['path'] == 'source.zip')
    assert sha(archive) == asset['sha256'] and archive.stat().st_size == asset['bytes']
    software = restored/'software-from-archive'
    with zipfile.ZipFile(archive) as z:
        entries = json.loads(z.read('SOURCE_FILES.json'))['files']
        expected = {e['path']: e for e in entries}
        assert set(z.namelist()) == set(expected) | {'SOURCE_FILES.json'}
        if not software.exists():
            software.mkdir()
            for name in z.namelist():
                path = Path(name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe software archive member')
                target = software/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
        for name, entry in expected.items():
            path = software/name
            assert not path.is_symlink() and path.stat().st_size == entry['bytes']
            assert sha(path) == entry['sha256'], name
    # Import from the actual distribution, rather than the current checkout.
    assert 'mamba_e8w5.runtime' not in sys.modules
    sys.path.insert(0, str(software))
    import torch
    import mamba_e8w5.runtime as runtime
    from mamba_e8w5.evaluation import generate_greedy
    assert Path(runtime.__file__).resolve().is_relative_to(software)
    attempts = []

    def reject_source(*_args, **_kwargs):
        attempts.append('source checkpoint loader')
        raise RuntimeError('Original checkpoint use is forbidden in restored inference')

    runtime.load_source_model = runtime.load_source_state = reject_source

    def open_guard(event, arguments):
        if event == 'open' and arguments and isinstance(arguments[0], (str, bytes)):
            value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            if Path(value).name == 'model_optim_rng.pt':
                reject_source()

    sys.addaudithook(open_guard)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = runtime.load_quantized_model(restored/'raw')
    tokenizer = runtime.SentencePieceTokenizer(restored)
    assert model._package_receipt['manifest_sha256'] == restore['raw_manifest_sha256']
    raw = json.loads((restored/'raw/manifest.json').read_text())
    for name, digest in raw['binding']['code_sha256'].items():
        assert sha(software/'mamba_e8w5'/name) == digest
    rows = []
    prompts = ('The capital of France is',
               'In a computer, memory is used to',
               'The scientist repeated the experiment because')
    for prompt in prompts:
        output, generated, count, memory = generate_greedy(model, tokenizer, prompt, 12, 'prefill')
        assert generated and all(0 <= token < tokenizer.vocab_size for token in generated)
        assert output.strip()
        rows.append({'prompt': prompt, 'output': output, 'generated_ids': generated,
                     'prompt_tokens': count, 'cache_bytes': memory})
    repeated = generate_greedy(model, tokenizer, prompts[0], 12, 'prefill')
    assert repeated[1] == rows[0]['generated_ids'], 'Fresh-cache generation repeat differs'
    assert not attempts
    report = {'complete': True, 'scope': 'Archived-source restored-package generation smoke; not PPL/recall evaluation',
        'release_manifest_sha256': args.expected_manifest_sha256,
        'restore_receipt_sha256': sha(restored/'restore_receipt.json'),
        'source_zip_sha256': sha(archive), 'archived_files_verified': len(expected),
        'runtime_path': str(Path(runtime.__file__).resolve()),
        'runtime_sha256': sha(runtime.__file__), 'raw_manifest_sha256': restore['raw_manifest_sha256'],
        'package_receipt': model._package_receipt, 'tokenizer_sha256': tokenizer.sha256,
        'source_checkpoint_argument_supplied': False, 'source_checkpoint_access_attempts': attempts,
        'source_independence_check': 'Source loader disabled and Python open audit rejects original checkpoint basename; this is not an OS filesystem sandbox.',
        'generation': rows, 'fresh_cache_repeat_identical': True,
        'environment': runtime.environment_receipt(), 'gpu_memory': runtime.gpu_memory_receipt(),
        'elapsed_seconds': time.monotonic()-started, 'script_sha256': sha(__file__)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, 'prompts': len(rows), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
