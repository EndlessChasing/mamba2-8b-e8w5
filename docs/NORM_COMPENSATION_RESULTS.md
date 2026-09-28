# Existing-norm compensation: PPL improved, recall gate failed

**Status: fixed experiment complete; combined advancement gate failed.**
Publication remains on hold. The original E8/W5 candidate and the eight-sweep
experiment are unchanged. No full-validation or test run, entropy repackaging,
checkpoint selection or public push followed this negative gate.

## Paired development result

The parent is the original two-sweep E8/W5 candidate. The new candidate replaces
only its existing 113 normalization tensors, trained under the
[predeclared protocol](NORM_COMPENSATION_PROTOCOL.md). Both arms were evaluated
in one process from their actual decoded/exported FP16 parameters.

| Metric | Original quantized parent | Norm compensation |
| --- | ---: | ---: |
| PPL, four frozen validation windows / 4096 targets | 9.237163951 | 8.965396490 |
| Normal public MK | 10/12 | 9/12 |
| Target-removed matches | 0/12 | 0/12 |

PPL improved **2.942109%**, with a mean NLL decrease of **0.029862574302
nats/target**. All four windows improved: 3.9070%, 2.2218%, 3.7592% and 1.8635%.
The PPL advancement condition (at most 9.144792311) passed. Normal recall
nonregression failed; the control condition passed. The combined gate therefore
failed, and the evaluator correctly skipped full validation.

There was one changed correctness outcome among all 24 matched prompts:
`validation-n64-t2-s0`, a query for the first entry among 64 records, with 977
prompt tokens. The parent returned the correct `677753`; the candidate returned
`677533`. The other 23 correctness outcomes were unchanged. The historical
[original FP16 baseline](../reports/baseline_dev_prefill.json) also answered this
exact, hash-matched prompt correctly. No recall examples entered training.

Twelve normal recall cases are a small engineering screen. One lost case is not
an estimate of a population-wide recall decline. It still fails the rule fixed
before this experiment. The result establishes a prose-quality gain on these
four windows at unchanged raw capacity, without establishing preserved recall,
full-corpus quality or the original source-relative +5% target.

Evidence: [paired evaluation](../reports/norm_compensation_v1_eval.json).

## What was trained

Only **113 existing FP16 norm tensors / 692,224 parameters** were fitted with
FP32 master values and a native FP16 forward. The original two-sweep E8 indices,
rotation/balance metadata, both W5 vocabularies and the other 280 small FP16
tensors were frozen. No adapter, additional precision or inference parameter
was added.

The fixed objective was 0.5 next-token CE plus 0.5 full-vocabulary KL from the
original FP16 teacher. Training used only the original 32 calibration windows,
2047 next-token targets each, four passes: **128 successful updates, 128
attempts, zero overflows and 262,016 target exposures**. Only checkpoint 128 was
exported; no intermediate checkpoint was scored on validation or recall.
Training, including loading and audits, took 284.06 seconds. These timings are
not an inference-throughput comparison.

Mean online training objectives by pass were 1.116037, 1.106649, 1.100821, 1.096145.
They were measured at changing parameters and shuffled windows, so they are
descriptive training values, not fixed-checkpoint quality evaluations.

The GPU smoke demonstrated exact native/functional hidden and logit equality,
exact checkpoint-gradient equality for all 113 norms, exact chunked/full hidden
gradients, and a 2.0e-7 objective difference from chunk reduction. The complete
2047-target trial update had finite gradients and no overflow. Its optimizer
state was discarded before the declared training run.

## Export, storage and memory

- All 394 frozen GPU tensors have identical before/after hashes. All 113 selected
  norm tensors changed after FP16 rounding; the other 280 small tensors are exact.
- The final checkpoint's 113 FP32 master tensors round bitwise to the exported
  FP16 values. Reloaded output is bitwise equal to the functional training forward.
- The evaluator independently audited all 507 parent tensors / 8,236,999,680
  parameters and verified the final replacement and inherited identities.
- `other_fp16.pt` remains **7,283,930B**. Logical raw model data remains
  **2,886,482,462B**, with no serialized payload-byte increase.
- Overlay plus manifest occupies 7,346,899B. Actual parent-plus-overlay directories
  occupy **2,894,033,233B**. This experimental layout keeps the original parent.
  Training and smoke work directories add **1,096,057,126B** separately; all
  these binaries remain on the GPU host. See the [storage receipt](../reports/norm_compensation_v1_storage.json)
  for included/excluded directories.
- Training peak GPU allocation was **36,043,514,880B**, including the teacher,
  student and backward workspaces. Paired evaluation peak was 22,062,521,856B.
  Both use expanded FP16 weights, not compact inference residency.
- No new Huffman container was built. The first candidate's 2.666GB complete
  distribution is not a measured distribution size for this trained overlay.

## Interpretation and next decision

Existing norm capacity can recover some development PPL without growing the raw
base. This particular prose-only CE/KL recipe did not preserve every recall case.
The per-layer cause of the lost case has not been isolated. Training objectives
or parameter constraints that explicitly protect recall are plausible follow-up
hypotheses, not validated repairs. The failed combined result is retained; no
threshold or checkpoint choice was changed afterward.

The source-relative full-test failure of the original candidate remains visible
in [RESULTS.md](RESULTS.md). This experiment does not replace that result or
establish a globally smallest/equal-quality model.

## Reproduction and identities

See [training implementation](NORM_TRAINING_IMPLEMENTATION.md),
[overlay contract](NORM_OVERLAY.md), and [evaluation commands](NORM_EVALUATION_RUNBOOK.md).
Choose fresh output paths for a new reproduction; existing reports are protected.

| Evidence | SHA-256 |
| --- | --- |
| Frozen protocol | `125cf066981e0b014c6cb1eb3cfd01608d63a9b7271185038b4dea30938308a3` |
| Smoke receipt | `c01afe9dd5690bc65f911ff47b0d7e0b82b23e12155331d7f6d9ca7847d4c115` |
| Final training receipt | `62f05f0abe6b1e254f32f133e630a35be11138af5c03e667e675bbe6ae37c9e8` |
| Final checkpoint 128 | `8fe3d4f8a271a28d1906d50a301028dc364cac3fd6d6ce5333c897fdf9268810` |
| Overlay manifest | `1fa9d35ff56a08e2f7ddf258bfa1fb61bc2e83ae2af404ab2f68a781821ac4d8` |
| Replacement small-tensor file | `8a98319ad513fceaf48673ff4199a8a22b5c1046c64b91bbf3bbbb09f5d84530` |
| Resolved candidate ledger | `ac2a47931c1b844d5ec15b039c4ec591031c56c3e53cc6eb5a38398d4a98cbf9` |
| Integrity receipt | `6fe1781e1bb4c3d8af94feffb15398aafd6080acddc8a619113f644f759fe476` |
| Paired development evaluation | `72350eaa1989601f4412ba9e31c178e22dba95ec0b85f5837f9f08b29b265dad` |
