"""Download pinned official NVIDIA assets once; validate size and SHA256."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPOSITORY = 'nvidia/mamba2-8b-3t-4k'
REVISION = 'b915550c63ba9359f88f44d1f6a600d85af27302'
ASSETS = [
    ('release/mp_rank_00/model_optim_rng.pt', 'model_optim_rng.pt', 16474189490,
     '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'),
    ('mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model',
     'mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model', 4573028,
     '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'),
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('models/source'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verified = []
    for relative, name, size, expected in ASSETS:
        final = args.output / name
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{relative}'
        if not final.exists():
            part = final.with_suffix(final.suffix + '.part')
            print(f'Downloading {name} ({size:,} bytes)', flush=True)
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--retry', '4', '--retry-delay', '5', '--continue-at', '-',
                            '--output', str(part), url], check=True)
            assert part.stat().st_size == size, (name, part.stat().st_size, size)
            assert sha(part) == expected, f'SHA256 mismatch: {name}'
            part.rename(final)
        else:
            assert final.stat().st_size == size and sha(final) == expected, name
        verified.append(dict(path=name, source_path=relative, url=url, bytes=size, sha256=expected))
        print(f'Verified {name}', flush=True)
    manifest = dict(repository=REPOSITORY, revision=REVISION, files=verified,
                    checkpoint_sha256=ASSETS[0][3], tokenizer_sha256=ASSETS[1][3], complete=True)
    (args.output/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
