# Full eight-sweep overlay build

This follows the passed [six-matrix training screen](E8_REFINEMENT_SCREEN.md)
and the fixed [refinement protocol](REFINEMENT_PROTOCOL.md). It creates a
separate set of 112 E8 projections while inheriting the parent's two W5
vocabulary files, other FP16 tensors, codebook and configuration. An overlay
is an experiment artifact; it is not a standalone model or a publication.

## Completed build and CPU verification

All **112 projections** completed with exact decoded FP16 disk readback,
unchanged raw byte counts, and the six retained pilot files unchanged. The
independent CPU verifier passed the complete inventory, parent dependencies,
source/code/calibration bindings, and predeclared screen identity/gate.
Parent, pilot, frozen codec/runtime and calibration remain unchanged.

Original-H squared projection-output error improved in every matrix:

| Scope | Matrices improved / worsened | Median reduction | Minimum | Maximum |
|---|---:|---:|---:|---:|
| All projections | 112 / 0 | 1.8970% | 1.0351% | 2.3663% |
| Input projections | 56 / 0 | 1.6381% | 1.0351% | 1.9330% |
| Output projections | 56 / 0 | 2.0457% | 1.3899% | 2.3663% |

The six pilot pairs use independently accumulated original-H quadratic errors
for both arms. For the other 106 matrices, the parent recorded an FP32 relative
RMS proxy, whose square is compared with the new independently accumulated
relative MSE. Small accumulation-precision differences remain. These are
training-proxy statistics; there is no new acceptance threshold for the other
106 matrices and no PPL or MK result in this report.

| Storage scope | Bytes |
|---|---:|
| 112 replacement E8 raw files | 1,535,708,888 |
| Five inherited raw files | 1,350,773,574 |
| Complete logical raw payload, unchanged from parent | 2,886,482,462 |
| New overlay directory, including its manifest | 1,535,914,423 |
| Physical parent plus overlay directories | 4,422,600,757 |
| Physical parent, overlay, preserved pilot and work receipts | 4,587,330,346 |

Physical counts include every file in those named directories. Tokenizer,
software, licenses, external logs/quality reports, original checkpoint and
calibration are outside these counts. No refined Huffman archive was generated.

The full build session took **1,139.90 seconds (19.00 minutes)**, reusing six
pilots and computing 106 new matrices. Summed quantization time for all 112,
including the pilot measurements, is 1,006.35 seconds; summed entry time with
readback/metrics is 1,074.48 seconds. Peak recorded Torch CUDA allocation is
3,733,252,608 bytes. This is offline compression cost, not inference residency.

Evidence:

- [Full manifest](../reports/e8_refinement_cd8_manifest.json), SHA
  `5c7d912bb06e0af107218418f575d04c5b789f910568f0c847fa0322228fd035`.
- [CPU integrity receipt](../reports/e8_refinement_cd8_integrity.json), SHA
  `ad9fbe2afb07196b04e1e9f6784d531a764de14c15f4b04bd2f646e76aec56cc`.
- [Per-matrix summary and exact byte inventory](../reports/e8_refinement_cd8_summary.json), SHA
  `6b17ccafa330d7b89d5016dd640a7e7b354962487b6924a96bd66719d8027239`.

The next stage is the protocol's paired GPU quality evaluation, including
fresh decoded-hash and full parameter-coverage checks. Training-MSE improvement
alone does not authorize publication or establish the source-relative target.

## Run and resume

```bash
python scripts/build_e8_refinement.py --self-test
python scripts/build_e8_refinement.py \
  --source-dir models/source \
  --hessian-dir calibration/v1 \
  --parent-dir artifacts/e8w5_v1 \
  --pilot-dir artifacts/e8_refinement_screen_v1 \
  --out-dir artifacts/e8_refinement_cd8_v1 \
  --preflight-only
```

The preflight is CPU-only. It checks source/code/calibration bindings, verifies
the parent files and six pilot files, recomputes the declared screen gate, and
copies the successful pilot files unchanged. Remove `--preflight-only` to
perform the remaining quantizations. The same command resumes after an
interruption, checking the complete provenance binding and all existing hashes
before computing missing matrices. Do not restart an already running process.

The only recipe change from the parent is eight coordinate sweeps instead of
two. The driver calls the unchanged codec and records per-file SHA, decoded
FP16 SHA, original-H output MSE, same-byte checks, disk readback, time, memory,
and source/Hessian identity. It does not evaluate PPL or recall.

## Durable per-matrix records

Every finished matrix is committed in this order:

1. Write a uniquely named pending E8 file into the sibling
   `artifacts/e8_refinement_cd8_v1_work` directory.
2. Decode and verify it, then atomically write its receipt into that work
   directory. The receipt binds all quantization inputs and file identity.
3. Move the file into the final overlay directory, then atomically update the
   overlay's `manifest.json`.

On resume, a valid receipt can recover an interruption before or after the
rename without repeating quantization. Corrupt bytes or a provenance mismatch
are errors. Unfinished pending files are preserved outside the final inventory;
they are not silently promoted. CPU tests cover all three commit stages and
reject a corrupt restored file.

One intentionally fail-closed interruption corner is a leftover
`manifest.json.tmp` from an interrupted manifest write. The exact-inventory
check rejects this extra file. Inspect the current manifest and transaction
receipt, preserve the stale temporary manifest in the sibling work directory,
and then rerun the same command; do not overwrite the committed manifest with
an unchecked temporary file. No such interruption occurred in the screen.

The complete overlay directory contains exactly 112 E8 files and its manifest.
Journals, logs and pilot files remain separate. The parent and pilot are never
modified. Per-matrix progress is printed as JSON and checkpointed immediately.
Completion is recorded only after all 112 replacements and parameter/byte
accounting checks pass.

## Independent verification

After completion:

```bash
python scripts/evaluate_refinement.py \
  --source-dir models/source \
  --parent-dir artifacts/e8w5_v1 \
  --overlay-dir artifacts/e8_refinement_cd8_v1 \
  --calibration-manifest calibration/v1/manifest.json \
  --screen-report artifacts/e8_refinement_screen_v1/report.json \
  --protocol docs/REFINEMENT_PROTOCOL.md \
  --report reports/e8_refinement_cd8_integrity.json \
  --verify-only
```

This command performs CPU-only integrity and provenance checks, including the
screen gate, exact retained pilot identities, all inherited dependencies and
complete file inventories. Later GPU evaluation additionally checks all
replacement decoded FP16 hashes and full 507-tensor/8,236,999,680-parameter
coverage. It must compare parent and candidate in one process under the
predeclared development-quality gate.

Report logical candidate bytes separately from physical experiment disk use.
The current loader requires the entire immutable parent directory plus the
overlay, even though its old E8 files are logically superseded. The original
checkpoint and Hessians are quantization inputs, not inference dependencies.
Tokenizer, software, licenses, manifests and transaction records are additional
to raw parameter bytes. Eight sweeps preserve raw representation size, while
their effect on Huffman size has not yet been measured.
