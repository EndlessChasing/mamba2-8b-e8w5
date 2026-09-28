# Reproducible release packaging and loading

`scripts/package_release.py` packages a **completed quantized raw directory**.
It never modifies that directory. Its output is an outer release containing:

- One `mamba2-8b-e8w5.huff` container, or all ordered parts of that container.
- The original pinned NVIDIA SentencePiece model and the runtime configuration.
- Original license/attribution snapshots and a generated experimental model card.
- Supplied quality/provenance JSON reports, copied verbatim.
- `source.zip`: exact corresponding public source, including any uncommitted
  changes used by this build, with its own per-file SHA ledger and upstream pins.
- The quantization manifest, the full container-readback receipt, and an outer
  `release_manifest.json` ledger of every asset.

No external base weight file is required to restore and load the release.
Reproducing **quantization** requires the separately downloaded original model
and calibration dataset. Huffman restoration does not recover the original
floating-point model: it recovers the quantized raw files exactly.

## Build after quantization

Run from the repository with its Python dependencies installed:

```sh
python scripts/package_release.py build \
  --raw-dir artifacts/e8w5_raw \
  --source-dir models/source \
  --output artifacts/release-e8w5 \
  --quality-report reports/baseline_quality.json \
  --quality-report reports/e8w5_quality.json
```

Use actual report paths produced by the evaluation run. Reports are never
invented by the packaging script. If `--quality-report` is omitted, the model
card and release manifest explicitly say that quality is unmeasured. Attaching
a report does not automatically declare a pass. The default provenance reports
are `reports/source_model.json` and `reports/source_download.json`, when present;
`--provenance-report` can select explicit JSON files instead.

`--tokenizer /path/to/the.model` can replace `--source-dir`; its contents must
match the pinned original tokenizer SHA. Completed raw files are checked against
their quantization manifest before packing. The generated container is then
fully read back with `verify_container(..., trusted_directory=raw_dir)`, binding
every restored raw file to its original size and SHA, with independent chunk
checks. E8 and W5 tables, offsets, scales, signs, metadata, and codebooks are all
inside the accounted container.

The builder writes into `<output>.building`, verifies that release, and renames
it to `<output>` only after success. It refuses an existing output/staging
directory. A failed staging directory is available for inspection; it is not a
completed release or a completed resume checkpoint.

### Split assets below 2 GiB

Add `--split-bytes 1500000000` for parts no larger than 1.5 billion bytes. The
finished release contains every part and removes the unsplit container. The
manifest records order, byte ranges, individual SHA values, and the whole
concatenated container SHA. Every part is required.

The default unsplit build writes the container directly into the release and
does not make a second container copy. Splitting needs temporary extra disk
space of approximately one container while parts are written and verified;
RAM stays bounded. Split restoration similarly needs one temporary assembled
container because the current reader seeks by absolute file offset. The script
does not claim an in-place split or multipart GPU decoder.

## Inspect all release bytes

```sh
python scripts/package_release.py verify \
  --release-dir artifacts/release-e8w5
```

The receipt reports real container bytes and **total release bytes**, including
tokenizer, configuration, reports, source, licenses, and the release manifest
itself. A manifest cannot contain its own SHA without a circular definition;
its SHA is printed separately in build/verify receipts. Retain that value in a
trusted release announcement or signed record. To check against it:

```sh
python scripts/package_release.py verify \
  --release-dir artifacts/release-e8w5 \
  --expected-manifest-sha256 TRUSTED_SHA256
```

Without an independently trusted manifest hash, verification checks internal
integrity and completeness, not publisher authenticity. Verification refuses
unlisted files, missing parts, symlink assets, and incorrect byte ledgers.

## Restore without the original NVIDIA weight checkpoint

Extract `source.zip` into a software checkout and install its documented
dependencies. A C++17 compiler builds the small portable Huffman codec in the
user cache; restoration does not need a GPU.

```sh
python scripts/package_release.py restore \
  --release-dir /path/to/downloaded-release \
  --output /path/to/restored-model \
  --expected-manifest-sha256 TRUSTED_SHA256
```

The output directory must not already exist. The command verifies every asset
and part, checks the full container hash, matches its member ledger to the
quantization manifest, and reconstructs each original raw member with SHA
verification. It then rechecks the raw manifest. Results are arranged as:

```text
restored-model/
  raw/                    # exact original quantized package files
  mt_nlg_plus_...model     # original pinned tokenizer, full filename retained
  config.json
  restore_receipt.json    # container/raw/manifest identities and validation scope
```

The receipt and tokenizer stay outside `raw/`, keeping its original file
inventory unchanged. Full source model downloads, FP16 weight exports, or a
shared external model base are unnecessary for restoration.

## Reference inference and quality evaluation

```python
import torch
from mamba_e8w5.runtime import load_quantized_model, SentencePieceTokenizer

model = load_quantized_model("/path/to/restored-model/raw", device="cuda")
tokenizer = SentencePieceTokenizer("/path/to/restored-model")
ids = torch.tensor([tokenizer.encode("The capital of France is")], device="cuda")
with torch.inference_mode():
    next_id = model(ids, num_last_tokens=1).logits[:, -1].argmax(-1)
print(tokenizer.decode(next_id))
```

This loader reconstructs full FP16 weights. Compressed archive bytes are not
FP16 GPU residency. Runtime evaluation uses the exact original tokenizer,
independent input/output vocabulary matrices, grouped Mamba-2 normalization,
and the protocol in `docs/EVALUATION.md`. The restoration receipt establishes
weight identity for that evaluation; it does not itself establish PPL, recall,
runtime speed, state memory, or a global smallest-model claim.
