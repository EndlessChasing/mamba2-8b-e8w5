# Frozen v0.2.0 WikiText-2 test

This directory documents an additional test of the unchanged public
`v0.2.0-resurface` release. It does not replace the original model assets,
validation reports, or original `e8w5_v1` test.

| Arm | Full test PPL |
| --- | ---: |
| Current E8/W5 base without Resurface | 7.534184760938831 |
| Same base with its published Resurface | 7.502958939186295 |

The [comparison](comparison.json) contains both complete 147-window arms and
507/224 tensor-ledger receipts. The [independent CPU audit](cpu_audit_v1.json)
passed. There are 300,963 targets, including the final 1,955-target window;
147/147 windows improve. The [reference raw manifest](reference_raw_manifest.json)
is byte-identical to the already published manifest, not a new model manifest.

## Reproduce

Restore the existing release using the unchanged [release guide](../../docs/RESURFACE_RELEASE.md).
Keep its raw package and archived source tree unchanged. Use the pinned versions
and flags in the [protocol](../../docs/RELEASE_V02_WT2_TEST_V1_PROTOCOL.md).
From a checkout containing the two external scripts below, run:

```bash
python scripts/run_release_v02_wt2_test_v1.py \
  --raw-dir artifacts/resurface_restored_v0_2_0/raw \
  --software-dir artifacts/resurface_restored_v0_2_0/software-from-archive \
  --protocol docs/RELEASE_V02_WT2_TEST_V1_PROTOCOL.md \
  --expected-protocol-sha256 99427c7998b9b18a44abfbd317b92d6da30706dd7bbbe159fd2b46d09449a13f \
  --out-dir artifacts/release_v02_wt2_test_reproduction
python scripts/audit_release_v02_wt2_test_v1.py \
  --comparison artifacts/release_v02_wt2_test_reproduction/comparison.json \
  --raw-manifest artifacts/resurface_restored_v0_2_0/raw/manifest.json \
  --software-dir artifacts/resurface_restored_v0_2_0/software-from-archive \
  --protocol docs/RELEASE_V02_WT2_TEST_V1_PROTOCOL.md \
  --runner scripts/run_release_v02_wt2_test_v1.py \
  --out artifacts/release_v02_wt2_test_reproduction/cpu_audit_v1.json
```

The runner exports `tokens.int64le` from the pinned public dataset for the
independent auditor; that temporary corpus export is not included in this Git
update. The auditor validates arithmetic, population, source hashes and reported
exact model tensors; it does not rerun GPU inference or rehash large weight files.
The raw package loader independently verifies all 119 actual payload files and
507 decoded base tensors before inference.

The current release uses native SSD parallel prefill, not compressed-state
per-token scoring. Its decoded base residency is 16,473,999,360 bytes and the
adapter tensor payload is 2,308,208 bytes. Actual per-arm CUDA memory peaks are
in `comparison.json`; persistent weight bytes and allocator peaks have different
accounting scopes.

The project previously used WT2 test for original pre-repair experiments. This
frozen retrospective measurement has no threshold, candidate selection, fitting,
or claim that the project test split was previously untouched.
