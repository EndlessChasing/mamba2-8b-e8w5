# Download and verify a release

The downloader needs **Python 3.10+ and curl**. It uses the public GitHub API and does not need Torch, CUDA, a GitHub login, or access to the original NVIDIA checkpoint.

Choose an existing tag from [GitHub Releases](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases). A repository checkout is not itself a published quantized model. Replace the quoted placeholders below with the tag and manifest SHA-256 printed in that release's notes.

```sh
python3 scripts/download_release.py \
  --tag "RELEASE_TAG_FROM_GITHUB" \
  --manifest-sha256 "MANIFEST_SHA256_FROM_RELEASE_NOTES" \
  --output ./downloaded-model
```

`--repo` defaults to `EndlessChasing/mamba2-8b-e8w5`. The script is standalone: copying just `scripts/download_release.py` is enough for downloading. It imports only the Python standard library and invokes curl with HTTPS verification enabled.

The independent manifest digest is optional:

```sh
python3 scripts/download_release.py \
  --tag "RELEASE_TAG_FROM_GITHUB" \
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

Follow `model-software/README.md` for the Python/Torch dependencies. Restoration uses the portable C++ Huffman codec and therefore also needs a C++17 compiler. It does not require downloading the original full-precision weights. Then:

```sh
python3 model-software/scripts/package_release.py verify \
  --release-dir ./downloaded-model \
  --expected-manifest-sha256 "MANIFEST_SHA256_FROM_RELEASE_NOTES"

python3 model-software/scripts/package_release.py restore \
  --release-dir ./downloaded-model \
  --output ./restored-model \
  --expected-manifest-sha256 "MANIFEST_SHA256_FROM_RELEASE_NOTES"
```

See [PACKAGING.md](PACKAGING.md) for disk accounting and reference inference. Downloading consumes the complete release's listed bytes. Restoring additionally needs the raw quantized files and, for split containers, temporary space for one assembled container. The current quality reference expands weights to FP16; archive size is not runtime GPU residency.

## Publishing convention

Upload `release_manifest.json` and every **outer** release file to the same tag using its basename. Do not upload files inside `source.zip` separately. The publisher must reject duplicate basenames before upload. The downloader paginates the GitHub asset API and downloads only the assets listed in the manifest.

Offline downloader checks can run without model dependencies:

```sh
python3 -m unittest discover -s tests -p test_download_release.py -v
```
