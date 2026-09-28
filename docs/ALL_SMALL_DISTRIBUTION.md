# Offline distribution of the measured all-small candidate

This packaging task uses the already evaluated all-small candidate. It does not
change weights, train, rerun PPL, certify recall, or publish a model. The learned
prototype experiment remains a separate GPU job.

## Fixed inputs

- Raw manifest: `bc554db936b13cbbeaeef5267d96b8d6ba7c183bf740c7e64e5f26ac7c478af8`.
- Weight container: `46be4e12d18ed36e33adf002962f0b7bcafe8478b249b628eca853930d43efcd`,
  2,660,171,471 bytes including its raw manifest and decoding metadata.
- Full-validation report: `957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206`.
  Its all-small PPL is 8.359867548 over 264,764 paired validation targets;
  source FP16 is 7.334175947. The source+5% target remains unmet. These are prior
  measurements tied to exact file identity, not a new packaging-time score.

## Static corresponding source

The second build uses Git commit
`dd653cb0bd66f44bfeb8d9ffa9ac8e627838952f`, tree
`bbd3601b4321e3e69475510ab4494863d2464d1c`. Its 4,863,118-byte Git export has SHA
`bb40cf50ea97622225e5c6b0a18debc1c6ae8de7d17f3134dd424e3ae7f8559e`.
All 209 exported files were verified after transfer and extraction. The export
has no `.git` directory and lives outside the active GPU checkout. Files and
directories are read-only; Python bytecode writes are disabled for packaging.

`SNAPSHOT_PROVENANCE.json` is an additional receipt outside the original Git
archive. Its SHA is
`78e6738aba34591b65b9e4ba96031e10699ddd75bf8f0d31c1569a40bb2985b4`.
The package builder records this provenance identity and hashes the actual
shipped source bytes. The separate export receipt binds the external tarball;
the builder itself does not claim to revalidate that external tarball.

- [Git-export receipt](../reports/all_small_source_export_v2.json).
- [Transfer/extraction receipt](../reports/all_small_source_snapshot_v2.json).
- [Exact source-file ledger](../reports/all_small_source_snapshot_provenance_v2.json).

## Build and restoration

The CPU-only runner `scripts/build_all_small_distribution_job.py` requires a
static export and fresh output paths. It uses the actual package build and
restore CLIs, verifies the fixed container SHA and all 118 restored raw files,
and binds restoration to the independently returned outer-manifest SHA.
It includes the pinned tokenizer, configuration, licenses, corresponding
software and quality/provenance reports in the counted distribution.

The second build and actual restoration both completed successfully in 360.83
seconds (253.13 for build/readback, 107.70 for restoration). All 20 assets passed
their byte/SHA checks, and all 118 restored raw members matched the evaluated
candidate exactly. Restoration checked the outer manifest against the SHA
returned by the completed build.

| Scope | Actual bytes |
| --- | ---: |
| Huffman weight container, including raw manifest | 2,660,171,471 |
| Tokenizer, software, licenses, reports, configuration and outer metadata | 11,005,000 |
| Complete offline distribution | **2,671,176,471** |
| Restored raw files, including raw manifest | 2,886,729,362 |

The weight container is split into two required parts of 1,500,000,000 and
1,160,171,471 bytes. The complete distribution is stored on the GPU host at
`/home/horde/Mamba2-8B-E8W5/artifacts/release-all-small-v2`; its restored files
are under `artifacts/restored-all-small-v2/raw`. The `v2` suffix denotes the
second packaging attempt, not a different model or another PPL experiment.

Outer release-manifest SHA:
`5f5445803f4f0b12ae88c74fec0b00d331a536e12cc424ddfe2093e538d9b5e6`.
Completed job-receipt SHA:
`285e84019a744b258753a1da61911911c895709d72bb934d157ec9e7b97ed52e`.

- [Build and actual restoration receipt](../reports/all_small_distribution_v2_job.json).
- [Complete asset ledger](../reports/all_small_distribution_v2_manifest.json).
- [Full container readback and pinned identity](../reports/all_small_distribution_v2_container_verification.json).

No archived-software GPU loading or generation test has run for this all-small
distribution. After the independent prototype training and full validation
release the GPU, the existing verifier can check this remaining usability gate:

```sh
/home/horde/.venvs/lodram/bin/python -u scripts/verify_restored_inference.py \
  --release-dir artifacts/release-all-small-v2 \
  --restored-dir artifacts/restored-all-small-v2 \
  --expected-manifest-sha256 5f5445803f4f0b12ae88c74fec0b00d331a536e12cc424ddfe2093e538d9b5e6 \
  --report reports/all_small_restored_inference_v2.json
```

That pending check loads archived software and restored weights without the
original checkpoint and performs short generation. It does not remeasure PPL
or MK. Assertions must be enabled, CUDA visible, and the report path unused.

## Preserved first-attempt failure

The first static-source build passed full readback and pinned identity checks
for all 118 container members, then failed while copying `PROJECT-LICENSE`.
`shutil.copytree` had propagated read-only source modes into the destination.
It produced no complete distribution and did not run restoration.

The fix creates fresh writable destination directories and copies license
bytes without changing source content, permissions or timestamps. A focused
CPU regression covers missing and existing `PROJECT-LICENSE` paths, nested
licenses, and refusal of an existing destination. The fix passed independent
review and is included in the second static source export. The first staging
directory and failure receipts remain intact.

- [Failure and actual traceback](../reports/all_small_distribution_v1_failure.json).
- [Completed first-attempt container readback](../reports/all_small_distribution_v1_container_verification.json).
- [Focused permission regression](../reports/readonly_license_copy_cpu_regression.json).

Publication remains on hold. Compressed file size does not establish GPU
memory use: the existing reference loader expands the weights to FP16.
