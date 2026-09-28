# Rank-4 compensation: completed negative result

**Both declared PPL gates failed.** The fixed 1,024-update experiment,
export checks and independent full validation completed successfully, but
candidate PPL worsened. Stop this recipe without extra epochs, intermediate
checkpoint selection or a rank sweep. The accepted all-small model and its
offline distribution remain unchanged; publication remains on hold.

## Fixed experiment

The [predeclared protocol](LOW_RANK_COMPENSATION_PROTOCOL.md) added rank-4
residual factors to all 112 input/output projections of the accepted all-small
base. The 224 FP32 master factors contain 7,827,456 values; each forward and
checkpoint replay casts them to FP16 before forming the native merged weight:
`half(W0.float() + B16.float() @ A16.float())`. All 507 base tensors remain frozen.

Training used 256 fresh, disjoint TRAIN windows of 2,048 tokens, excluding
prior calibration and repair windows, and four seeded passes. The fixed
objective was 0.5 next-token CE plus 0.5 full-vocabulary original-FP16-teacher
KL at temperature one, with AdamW learning rate 3e-4. The run completed
**1,024 successful updates, 1,024 attempts, zero overflow retries and
2,096,128 target exposures**. Only the final checkpoint was evaluated.

## Independent full validation

All three models used the same pinned validation token stream in one process:
**264,764 next-token targets across 130 fresh-state windows**, with up to
2,048 targets per window. Evaluation used the frozen native runtime after
independent factor loading and merging, without the training bank forward.

| Model | Full-validation PPL |
| --- | ---: |
| Original source cast to FP16 | 7.334175947318572 |
| Accepted all-small E8/W5 base | 8.359867548009992 |
| Final rank-4 candidate | 8.630189440194565 |

The candidate regressed **3.2335667%** versus the base and remained
**17.6708808%** above source. Mean NLL increased by **0.031823873 nats per
target** versus the base; **4 windows improved and 126 worsened**.

| Predeclared gate | Required candidate PPL | Result |
| --- | ---: | --- |
| At least 1% improvement over accepted base | ≤ 8.276268872529892 | **FAIL** |
| At most 5% above original source | ≤ 7.700884744684501 | **FAIL** |

These validation data have informed development and are not untouched test
evidence. No test-set result or new MK result is claimed. MK remains deferred
under the [PPL-first direction](PPL_PRIORITY_CONTINUATION.md).

## Integrity and storage

All 224 rounded final master factors matched their 112 serialized files.
The 128-token export check produced bitwise-equal native hidden states and
last-eight-token full-vocabulary logits. All 507 frozen base tensor hashes
were unchanged. The independent evaluator verified all 112 merged projection
identities and 395 inherited tensors, including the final 507-tensor content
audit. Training and evaluation each exited with code zero; see the
[completed job record](LOW_RANK_RUNNING_JOB.md).

| Storage scope | Bytes |
| --- | ---: |
| Additional FP16 factor values | 15,654,912 |
| 112 factor headers, 32 bytes each | 3,584 |
| Complete factor files | 15,658,496 |
| Candidate overlay manifest | 198,517 |
| Factor files plus overlay manifest | **15,857,013** |
| Existing logical base data | 2,886,482,462 |
| Logical base data plus factor files, excluding manifests | **2,902,140,958** |

The logical-data count excludes tokenizer, software, licenses, parent metadata,
reports and optimizer checkpoints. No new entropy-coded container or complete
distribution was built for this candidate, and these file sizes do not establish
compressed inference residency. The accepted all-small complete distribution
remains **2,671,176,471 bytes**, with no low-rank factors added.

## Training trend and its limits

The following are means of online training measurements over each successive
256-update pass. Parameters changed between measurements; these are **not
epoch-end checkpoint evaluations or final fixed-model TRAIN PPL**.

| Online pass | Mean objective | Mean CE | Mean teacher→student KL |
| --- | ---: | ---: | ---: |
| 1 | 1.18354 | 2.14268 | 0.22439 |
| 2 | 1.09814 | 2.00581 | 0.19047 |
| 3 | 1.00045 | 1.80820 | 0.19271 |
| 4 | 0.92048 | 1.61726 | 0.22370 |

The completed [fixed-model TRAIN diagnostic](LOW_RANK_GENERALIZATION_RESULTS.md)
now measures fitting PPL8.714097→4.247930 (−51.2522%) but fresh PPL8.510325→8.691613
(+2.1302%). All64 fresh windows also worsen in teacher KL. This strongly supports
overfitting in this recipe, while different text subsets and multiple changed
recipe components prevent a causal attribution to one mechanism. The diagnostic
did not train or promote a model. A separate [pure teacher-KL protocol](TEACHER_KL_COMPENSATION_PROTOCOL.md)
resets to accepted all-small, uses fresh factors and a reserved-window gate;
it has not run. This negative result applies to the declared recipe, not all
possible low-rank repairs.

[Prototype compensation](PROTOTYPE_COMPENSATION_RESULTS.md) still has the
lowest measured compact-candidate PPL, **8.306585062**, but also failed its
1% advancement gate and was not promoted. The accepted all-small base remains
the basis for the existing distribution.

## Immutable receipts

The following local files were checked against their actual sizes and SHA256s.
Large factor and optimizer binaries remain on the GPU host.

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
| [Training](../reports/low_rank_compensation_v1_train.json) | 65,507,208 | `b7fb6bf26d174189dbbe6d37298adb8e556412ee282c16bc996743044940694a` |
| [Candidate manifest](../reports/low_rank_compensation_v1_manifest.json) | 198,517 | `6549fac5e2531ff73bfcca54e4615ae35389f2a4e8f39b32fac3c583515f018a` |
| [Independent full validation](../reports/low_rank_compensation_v1_eval.json) | 878,430 | `efb4264bc65efded5fa9b05f9ea689542f9feaf23d2108a1f7c335bee6c67979` |
| [Completed job](../reports/low_rank_compensation_v1_job.json) | 6,914 | `cbe7e550e31a7b5780cf5c8a7fb682d0350c4e0c0b0dd87b4ad8e1abc69b1e62` |
