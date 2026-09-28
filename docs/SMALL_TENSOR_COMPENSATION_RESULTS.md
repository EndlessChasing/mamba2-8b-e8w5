# All-small-tensor PPL compensation

**Status: complete. Full-validation PPL improved at unchanged raw capacity; the source-relative +5% PPL target remains unmet.**
Publication remains on hold. This is a PPL-priority experiment under the user's
[revised research scope](PPL_PRIORITY_CONTINUATION.md); recall is deferred and
no Resurface improvement is assumed.

## Complete validation result

All four models ran in one process on the same **264,764 targets / 130 windows**,
at most 2048 targets per window, fresh zero state, native SSD prefill, FP16 head
GEMM and FP32 cross entropy. The unchanged evaluator scored every target once.

| Model | Full-validation PPL | Versus original FP16 |
| --- | ---: | ---: |
| Original FP16 | 7.334175947 | — |
| Original E8/W5 | 8.836560440 | +20.4847% |
| E8/W5 + norm compensation | 8.589640668 | +17.1180% |
| E8/W5 + all-small compensation | 8.359867548 | +13.9851% |

The new candidate improves PPL **2.6750% versus norm-v1** and **5.3946% versus
original E8/W5**. Of 130 windows, 119 improve versus norm-v1 and 11 worsen, with
no ties. Mean NLL drops 0.027114320206 nats/target from norm-v1. The predeclared
1% full-validation improvement condition passes.

The remaining gap to the original source is **+13.9851% PPL** (0.130897523435
nats/target), so the +5% source target, PPL <=7.700884745, still fails. This is the
best **PPL** among the measured repaired candidates; it is not a combined-quality
acceptance. MK was not measured for this candidate under the user's PPL-first
instruction, and Resurface recovery remains a separate future experiment.

Validation has already been used for diagnosis and repair research. It is not an
untouched test set. No test split was used in this experiment. Finetuning on
training text also does not prove reconstruction of the original weights or
isolate how much an equivalently finetuned uncompressed model would improve.

Independent evaluation took 274.932 seconds including source loading and audits;
peak allocation was 22,062,521,856 bytes in the expanded FP16 quality runtime.
The [evaluation receipt](../reports/small_compensation_v1_eval.json) preserves
all four per-window losses, token identities, and actual loaded weight hashes.

## Fixed training and model capacity

The candidate starts from the **rounded norm-v1 FP16 export** and trains all
393 existing small tensors / 3,580,928 parameters: block/final norms, gated
norms, convolution weights/biases, A_log, dt_bias and D. Architecture stays pure
Mamba-2 with 56 blocks. All 112 E8 projection matrices and both W5 vocabulary
matrices remain exact; no adapter, state dimension or new inference tensor is
introduced. See the [predeclared protocol](SMALL_TENSOR_COMPENSATION_PROTOCOL.md).

The 256 train-only windows contain 524,288 stored token positions, internally
disjoint and with zero overlap with the 32 earlier calibration/norm-training
windows. Each pass scores 524,032 targets; four seeded passes produced exactly
**1024 successful updates, 1024 attempts, zero overflows and 2,096,128 additional
target exposures**. The earlier norm stage used 262,016 exposures, so this
candidate's two training stages total 2,358,144 target exposures. Original E8
covariance collection is separate calibration work. No validation, test or MK
inputs entered either training stage.

Training used fresh AdamW state, learning rate 1e-4, global gradient clip 1,
FP32 masters with native FP16 casts, and 0.5 next-token CE plus 0.5 full-vocabulary
KL from the original FP16 teacher. Only the final 1024-step checkpoint was
exported; no intermediate quality score selected it. All 393 small tensors
changed after FP16 rounding.

Training took 1038.07 seconds including loading and audits. Peak GPU allocation
was **36,040,458,752 bytes (33.57 GiB)** including teacher, student and backward
workspaces. These are training measurements, not compact inference residency
or isolated throughput benchmarks. All binaries remain on the GPU host.

The logical raw weight data remains **2,886,482,462 bytes**, and the replacement
small-tensor archive remains **7,283,930 bytes**, a zero-byte change.
The new overlay manifest is 95,371 bytes; the two-file overlay totals 7,379,301
bytes. Retaining the original parent plus the overlay uses 2,894,065,635 bytes.
Training work uses 1,396,860,886 bytes (32 checkpoints account for 1,395,495,712),
smoke work 7,284,011 bytes, and the new data directory 4,201,723 bytes. These
selected scopes total 4,302,412,255 apparent bytes; they exclude source weights,
original Hessians, prior experiments/releases, external reports, code and tokenizer.
See the [independent final audit](../reports/small_compensation_v1_final_audit.json).
No new Huffman container was generated; the original candidate's 2.666 GB complete
distribution is not a measured size for this trained candidate.

## Correctness evidence

- Three CPU training tests cover all parameter-family mapping, checkpoint replay,
  export parity, stability rejection and deterministic data selection.
- Eleven CPU boundary tests cover final-step accounting, same-window overflow
  retries, exact FP16 rounding, data provenance and inherited-file identity.
- The new GPU smoke verifies all 393 gradients finite and all seven parameter
  families nonzero. Checkpoint gradients and exported/native forward results are
  bitwise equal; all 114 frozen GPU tensor hashes remain exact. Its trial update
  was discarded before training.
- Final training independently verifies all 114 frozen GPU tensor hashes and all
  393 saved FP32 masters rounded to the actual FP16 export. The unchanged public
  runtime's output equals the functional training forward after export.
- The separate evaluator verifies the complete 507-tensor model, checkpoint and
  loaded small tensors, rehashes the actual 114 frozen values, and computes all
  full-validation arms using unchanged evaluation math.

## Online training diagnostics

These values were observed at changing parameters on shuffled training windows.
They are **not held-out PPL or fixed-checkpoint measurements**.

| Pass | Mean objective | Mean training CE | Mean teacher-to-student KL |
| --- | ---: | ---: | ---: |
| 1 | 1.156452 | 2.120330 | 0.192575 |
| 2 | 1.056788 | 1.948637 | 0.164939 |
| 3 | 0.992324 | 1.816532 | 0.168117 |
| 4 | 0.945488 | 1.707136 | 0.183840 |

The later passes lower CE while mean KL rises from pass two. This is a diagnostic
of the fixed mixed training objective; it does not establish validation overfitting
or improvement. No step was selected or recipe changed in response.

## Reproduction and identities

See [training implementation](SMALL_TENSOR_TRAINING_IMPLEMENTATION.md) and the
[independent evaluation runbook](SMALL_EVALUATION_RUNBOOK.md). Use fresh report
and work paths; preserve existing receipts. No public push or model upload is
part of this experiment.

| Evidence | SHA-256 |
| --- | --- |
| [Protocol](../docs/SMALL_TENSOR_COMPENSATION_PROTOCOL.md) | `f57229f979f8bc94def0780f87aa48ae0b83f7d30dc29fef527b46aaa2be0c31` |
| [Prepared training manifest](../reports/small_compensation_v1_data.json) | `88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709` |
| [GPU smoke](../reports/small_compensation_v1_smoke.json) | `cd71bb63d32d6c703dd8a0efc823836d2fb9110075d0d278de855b6dca7d46c7` |
| [Final training report](../reports/small_compensation_v1_train.json) | `2b0abbe2559ca4393af5d949fc7fa17dbb55cc164ca1ffc2c88347e3ebed0aba` |
| [Candidate manifest](../reports/small_compensation_v1_manifest.json) | `edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006` |
| [Export integrity](../reports/small_compensation_v1_integrity.json) | `8c06dc1b378b9146f7e770a6d3097f788e43658c2a0604f5aa1fd0660a160211` |
| Final checkpoint (GPU host only) | `8620ef7753cd3a2517158185d752acc727d7a52e03a4b181f04793e0184caff1` |
| Replacement small-tensor file (GPU host only) | `15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4` |
| [Independent full validation](../reports/small_compensation_v1_eval.json) | `957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206` |
| [Independent final audit](../reports/small_compensation_v1_final_audit.json) | `c16d6d382deb46f1dfa9eb589cba4ca86d2dab85660d639a891bd0eb90250e51` |

## Next PPL work

The fixed recipe is complete. The next promising hypothesis should change the
E8 quantization approximation itself, such as codeword assignment or a calibrated
codebook, under a new bounded protocol. Simply training decoder input balances
mostly repeats the existing norm gains in exact arithmetic; see the
[balance/norm audit](BALANCE_NORM_EQUIVALENCE.md). None of those further repairs
has been measured here. The original release package is unchanged and unpublished.
