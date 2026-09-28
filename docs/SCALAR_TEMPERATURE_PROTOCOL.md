# Bounded scalar-temperature repair protocol

Declared before implementation or fitting. This separate experiment follows
the completed [fixed-beta diagnostic](OUTPUT_CALIBRATION_DIAGNOSTIC_RESULTS.md).
It fits only the accepted all-small model, whose complete distribution is
2,671,176,471 bytes. The failed teacher-KL export is not a candidate in this
experiment. No model weights, E8 indices, W5 values or recurrent state format
will change. MK and publication remain deferred.

## Data and fixed budget

Use the already verified WikiText-2 TRAIN stream and teacher-KL data manifest
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
Select 64 windows from its 448 `training_windows` in stored index order:
`floor(i * 447 / 63)` for `i = 0..63`. These windows did not train the accepted
all-small model; the rejected teacher-KL candidate used them. Preserve their
2048-token spans, 2047 targets, original stream starts and token hashes.
Fit on exactly 131,008 targets per pass, with fresh state for every window.

Do not use the 64 observed diagnostic/reserved windows for fitting, solver
stopping or choosing the scalar. Lock the seven unused TRAIN starts as the
separate small sanity screen:
`362496, 718848, 1083392, 1439744, 1802240, 2158592, 2510848`.
Verify their zero interval overlap with all earlier fitting and observed
windows using the original data manifests. Seven windows are only a bounded
screen, not sufficient independent generalization evidence.

## One fitted scalar

Let `beta = 1 / T`. Run the unchanged native FP16 backbone and FP16 language
head; cast logits to FP32, multiply by the stored FP32 beta, and compute
full-vocabulary FP32 target CE in 64-token chunks plus the 63-token tail.
No full-vocabulary logits are retained between chunks or serialized.

Initialize beta to exactly 1. Perform exactly three TRAIN updates, then one
final TRAIN evaluation. Every pass scores the same fixed 64 windows. For each
update compute the token-weighted CE derivative `g = E_p[z] - z_y` and
curvature `h = Var_p(z)` for `p = softmax(beta*z)`. Accumulate sums in FP64
after native FP32 per-token calculations. Require finite values and `h > 0`.

Use `step = clip(g / h, -0.05, +0.05)` and
`beta_next = clip(beta - step, 0.8, 1.2)`, rounded immediately to FP32.
There is no line search, temperature sweep, intermediate checkpoint selection,
extra iteration or solver tuning after outcomes. Only the third update's
serialized beta is eligible. If its final TRAIN CE exceeds the initial CE,
stop without evaluating the screen. Numerical errors also stop this recipe.
This protocol permits TRAIN Newton updates; the preceding diagnostic did not
compute or choose a Newton estimate on its observed windows.

## Controls and native export

Before GPU fitting, check the scalar derivative and curvature against an
independent CPU autograd calculation, including a short chunk tail. Verify
FP32 serialization roundtrip and invalid-scalar rejection. Reuse prior metric
tests rather than duplicating a model test framework.

Require beta=1 CE to match the unchanged native CE calculation in the same
process. Bind the actual 507 accepted parameter hashes, model/data/code
manifests and final content checks. Save exactly four little-endian FP32 bytes
for beta plus a manifest with its SHA256, inputs, solver history and target
counts. Reload the actual scalar file for all subsequent scoring. Record
metadata and implementation bytes separately; do not claim the complete
distribution is byte-identical or newly built. Log execution precision and
available autotuner settings; keep historical numerical drift separate from
same-process correctness.

## Final-only screen and conditional validation

Score unchanged source, accepted base and the final reloaded scalar on the
seven locked windows, using identical inputs and full-vocabulary losses.
Require at least 1% lower aggregate PPL than the uncalibrated base to continue.
If that rule fails, retain the negative result and stop; do not run full
validation, change the scalar or expand the fitting budget.

Only if the screen passes, run the established complete validation protocol:
130 windows / 264,764 targets for source, base and final scalar. Report both
the existing >=1% improvement requirement and the source+5% target. Validation
has repeatedly informed development and is not untouched test evidence.
No candidate is promoted merely by passing the small screen. Any eventual
promotion requires the declared complete-validation outcomes and an accurately
composed, verified runtime/package; no publication is authorized here.

Positive temperature changes probabilities but preserves exact-logit order.
Check native argmax equality as a numerical control, without presenting it as
an MK measurement. This route cannot recover lost ranking information. A
favorable local derivative provides motivation, not evidence of a useful
finite PPL improvement or of recall recovery.
