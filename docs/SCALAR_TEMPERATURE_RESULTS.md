# Scalar-temperature experiment: completed negative screen

The fixed experiment completed with exit code 0. The fitted scalar improves
seven-window screen PPL by **0.0574603%**, below the declared **1%** requirement.
The screen fails; full validation was correctly skipped. Stop this recipe
without extra iterations, fitting expansion, scalar selection or promotion.
Accepted all-small and its existing distribution remain unchanged.

## Fixed fitting and serialized result

Follow the [frozen protocol](SCALAR_TEMPERATURE_PROTOCOL.md), SHA256
`ade965d8381a2365705eade045b67f1eb6d60bde3b87bfbd0d92ec61b0bcac6f`.
Only the accepted all-small model's FP32 inverse temperature was fitted; its
507 model tensors, E8/W5 values and recurrent-state format did not change.
The failed teacher-KL model was not a candidate in this experiment.

The run completed exactly **three clipped TRAIN Newton updates**, then the
specified fourth pass to score the final TRAIN CE. Each pass used the same
64 windows / 131,008 targets, for 524,032 scored target exposures in total;
only the first three passes drove updates. These fitting windows did not
train the accepted all-small model, although the rejected teacher-KL experiment
used them. The previously observed 64 diagnostic/reserved windows were not
used for this fit or scalar selection.

The final scalar was serialized and reloaded exactly:

- **beta = 0.9864529967308044**, hence **T = 1.013733044872986**.
- Four little-endian FP32 bytes: `2f887c3f`.
- `beta.bin` SHA256:
  `ee0fd2f9729a34f98d0c237f9d4a1efea21302b300e45941ce52774c9c21d902`.

TRAIN PPL decreased from **8.185296703079976** to **8.182022041947075**,
**0.0400066%**. The final TRAIN nonregression check passed, permitting the
screen. No intermediate scalar was selected; only the third update's stored
value was used subsequently.

## Separate seven-window screen

The seven predeclared, previously unused TRAIN windows contain **14,329
next-token targets**. They are disjoint from fitting and earlier observed
windows. This is a small screen, not sufficient generalization evidence or a
new full-validation/test result. Native FP16 head computation precedes FP32
scalar multiplication and full-vocabulary CE, with fresh state per window.

| Model | Screen PPL |
| --- | ---: |
| Original source FP16 |7.008538753993407 |
| Accepted all-small, beta = 1 |8.219889723075257 |
| All-small with final reloaded scalar |8.215166552827002 |

The measured reduction is **0.0574603%**; five windows improve and two worsen.
Passing required PPL at most **8.137690825844505**, so the **1% screen fails**.
Native argmax changes are **0 / 14,329 targets**, as expected for a positive
scalar that preserves logit order.

`full_validation.performed` is false: validation data were not loaded. The
scaled model's complete-validation PPL, its 1% improvement condition and its
source+5% full-validation target are **unmeasured**. A successful process exit
and successful integrity checks do not change the negative screen outcome.

## Integrity, storage and execution

CPU derivative/tail, serialization and input checks passed before fitting.
The same-process beta = 1 CE control is exact. Source/base identities and all
507 model-content hashes remain unchanged; final scalar readback and bound
file checks pass. No model weights were trained or replaced.

| Additional artifact scope | Bytes |
| --- | ---: |
| FP32 scalar payload |4 |
| Actual scalar manifest |96,550 |
| Scalar plus manifest |96,554 |
| New implementation source, separately counted |27,457 |

Existing logical weight data plus the scalar total **2,886,482,466 bytes**,
excluding manifests, tokenizer, software and reports. No new complete
container/distribution was composed; the existing 2,671,176,471-byte all-small
distribution does not contain this scalar or its new runtime step.

Internal runtime was **208.5978 seconds**, process wall time **210.7754 seconds**.
Peak allocated GPU memory was **33,662,906,368 bytes** for the decoded FP16
quality reference, not compressed inference residency or isolated throughput.

## Interpretation and retained outcome

This fitted global temperature provides a measurable but very small benefit
on the declared fit and screen sets. It is insufficient for this protocol's
advancement gate. It does not establish that every calibration approach is
ineffective, nor that calibration is the dominant source of compression loss.
Scalar temperature cannot recover lost exact-logit ranking information.

Accepted all-small **full-validation PPL8.359867548009992** and its verified
distribution remain unchanged. The earlier prototype's numerical low
**8.306585061749145** on that validation split still failed its 1% advancement
gate. Those values are separate from the fit/screen scores here. There is no
promotion, extra fitting, new MK/recall result or publication.

## Completed evidence

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
|[Fit and screen result](../reports/scalar_temperature_v1.json) |1,358,517 |`cc678de09046d6f1113c73ebe1fbdd5b4664c9878a6f2329a64917deb4c4a4ee` |
|[Process / exit](../reports/scalar_temperature_v1_process.json) |1167 |`57a6b4c8fafc1d8865a192b4f41f51a9c201cb2276e4148668c38feaa3c08c83` |
|[Scalar manifest](../reports/scalar_temperature_v1_manifest.json) |96,550 |`f08f5fb7a1026a18c8f12fb4fa077368facc28b60a92550fa311aa5ccdf06e7a` |
|[CPU checks and preflight](../reports/scalar_temperature_v1_cpu.json) |96,171 |`18ae64974942b057a0f47199bc6f0278e26366e4325958bb89f0df5a1fb2611e` |

Implementation: `scripts/fit_scalar_temperature.py`, SHA256
`5dc7d188b4bcac74614368bc37e8f28d756fbb99b953d599deb7bfe6db845892`.
