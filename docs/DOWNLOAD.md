# Download and verify a release

The downloader needs **Python 3.10+ and curl**. It uses the public GitHub API and does not need Torch, CUDA, a GitHub login, or access to the original NVIDIA checkpoint.

The current research prerelease is `v0.2.0-resurface`. Download it from [GitHub Releases](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface). The complete release is **3,165,987,804 bytes (3.166 GB)**. A repository checkout is not itself a downloaded quantized model. The commands below pin the manifest SHA-256 verified during publication; the release notes and manifest record the exact byte totals.

```sh
python3 scripts/download_release.py \
  --tag "v0.2.0-resurface" \
  --manifest-sha256 "da5931dc8315bf576b773abdf4c77828a2994ccaf4fd14858c7798235b2ef19d" \
  --output ./downloaded-model
```

`--repo` defaults to `EndlessChasing/mamba2-8b-e8w5`. The script is standalone: copying just `scripts/download_release.py` is enough for downloading. It imports only the Python standard library and invokes curl with HTTPS verification enabled.

The independent manifest digest is optional:

```sh
python3 scripts/download_release.py \
  --tag "v0.2.0-resurface" \
  --output ./downloaded-model
```

Without `--manifest-sha256`, the script still checks every downloaded file against the manifest obtained from GitHub, but it has no separately trusted manifest identity. The final JSON printed to the terminal records that distinction.

## What is verified

1. The requested public tag exists, and every required GitHub asset is fully uploaded.
2. The manifest's format, complete flag, canonical relative paths, unique flat upload names, and byte ledger are valid.
3. Every asset has the expected size and SHA-256. The downloaded parts also concatenate to the complete weight container's expected SHA-256.
4. The output has exactly the files listed in the manifest plus `release_manifest.json`.

GitHub stores release assets under flat filenames. For example, the asset `quality-01-result.json` is restored to its manifest path `reports/quality-01-result.json`; `APACHE-2.0.txt` is restored under `licenses/`. Duplicate basenames, including case-only collisions, are rejected before any payload is downloaded.

The checks establish byte identity. They do not execute the model or infer a quality pass. Inspect the release's model card and attached PPL/recall reports.

## Resume an interrupted download

Run the same command with the same output directory. Verified final files are skipped. A `filename.part` file resumes with an HTTP Range request. If a completed partial file fails its SHA, it is discarded and downloaded once from the beginning. The `.part` files are removed after successful verification.

Use a dedicated output directory. The script rejects symlinks, traversal paths, unexpected files, and conflicting final files. It preserves an existing final file with the wrong identity; move that file outside the output before retrying. A directory containing a different release manifest must be kept separate.

Do not save logs, receipts, or extracted source inside the release directory. The downloader prints its receipt to standard output and creates no extra receipt file there. This preserves compatibility with the stricter packaging verifier.

## Restore model files

The downloaded release includes `source.zip`, tokenizer, licenses, configuration, and exact weight container parts. Extract the source into a separate directory:

```sh
mkdir model-software
unzip downloaded-model/source.zip -d model-software
```

Follow `model-software/README.md` for the Python/Torch dependencies. Restoration also needs a C++17 compiler because the shared container reader loads the portable codec. The `v0.2.0-resurface` container stores every raw member without an additional entropy-coding pass; this is explicitly recorded as `all-raw-members-no-entropy-coding`. It does not require downloading the original full-precision weights. Then:

```sh
python3 model-software/scripts/package_release.py verify \
  --release-dir ./downloaded-model \
  --expected-manifest-sha256 "da5931dc8315bf576b773abdf4c77828a2994ccaf4fd14858c7798235b2ef19d"

python3 model-software/scripts/package_release.py restore \
  --release-dir ./downloaded-model \
  --output ./restored-model \
  --expected-manifest-sha256 "da5931dc8315bf576b773abdf4c77828a2994ccaf4fd14858c7798235b2ef19d"
```

## Run the restored model

Use Linux, a compatible CUDA PyTorch installation and the public Mamba runtime.
The validated environment uses `mamba-ssm==2.3.2.post1`; wheel/build compatibility
also depends on PyTorch, CUDA and GPU architecture. From the extracted source:

```sh
python3 -m pip install -e ./model-software
python3 -m pip install 'mamba-ssm==2.3.2.post1' --no-build-isolation

python3 -m mamba_e8w5.release_generate \
  --model-dir ./restored-model/raw \
  --prompt "The capital of France is" \
  --max-new-tokens 12 \
  --repeat \
  --report ./generation-receipt.json
```

The release loader uses the restored native tokenizer, all encoded base weights
and the bundled **enabled soft adapter**. It does not need the NVIDIA source
checkpoint, calibration Hessians, previous model directories or training data.
The receipt belongs outside the restored raw directory. `--repeat` checks a new
cache on the same prompt; it is not a benchmark or cross-process reproducibility
guarantee. See the [model card](MODEL_CARD.md) for the known historical MK drift.

See [RESURFACE_RELEASE.md](RESURFACE_RELEASE.md) for this version's packaging
scope. Downloading consumes the complete release's listed bytes. Restoring additionally needs the raw quantized files and, for split containers, temporary space for one assembled container. The current quality reference expands weights to FP16; archive size is not runtime GPU residency.

## Publishing convention

Upload `release_manifest.json` and every **outer** release file to the same tag using its basename. Do not upload files inside `source.zip` separately. The publisher must reject duplicate basenames before upload. The downloader paginates the GitHub asset API and downloads only the assets listed in the manifest.

Offline downloader checks can run without model dependencies:

```sh
python3 -m unittest discover -s tests -p test_download_release.py -v
```
