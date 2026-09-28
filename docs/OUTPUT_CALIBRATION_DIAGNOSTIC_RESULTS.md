# Output-calibration diagnostic: first attempt failed its execution control

The first fixed-beta diagnostic attempt stopped with exit code 1 at the exact
historical CE control for `source_fp16/0`, before accepting any window row.
The diagnostic is incomplete: there are no accepted diagnostic measurements
or conclusions. A bounded investigation of the first window's execution
control is underway. The protocol has not been weakened and model weights
remain unchanged.

## Failed attempt receipt

The error was `Original beta1 CE does not reproduce exactly: source_fp16/0`.
The report has `complete: false`, zero source rows and empty summaries. The
process supervisor completed recording the failed child; its own
`complete: true` does not mean the diagnostic succeeded.

| Evidence | Bytes | SHA256 |
| --- | ---: | --- |
| [Incomplete diagnostic report](../reports/output_calibration_v1.json) |245,283 |`38c31d728bc44dba9882b3e2344aeae1feec746a48d500dcbe27264b6b3a374c` |
| [Process / exit receipt](../reports/output_calibration_v1_process.json) |1077 |`2b311d9d729e176b9d2a03d04277cebce4ce8d36300ad6a259cfa78c03f77aee` |

Process wall time was 57.5164 seconds; the incomplete report records 55.1897
seconds inside the diagnostic. This is an execution-control failure, not a
measurement of entropy, temperature direction or a new model-quality outcome.

## Fixed scope

Follow the [frozen diagnostic protocol](OUTPUT_CALIBRATION_DIAGNOSTIC_PROTOCOL.md),
SHA256 `baa87fa1549e29bf757cca8e18f1ab0feadc18daebffa79402c75c87035bc05d`.
The three fixed native models are original source cast to FP16, accepted
all-small, and the final 448-step teacher-KL candidate. Analyze beta = 1 only
on the same already-observed 64 reserved TRAIN windows / 131,008 targets.
These are declared input counts, not a claim of completed scoring.

This is descriptive posthoc analysis. There is no temperature fitting, sweep,
Newton candidate, training, checkpoint selection or advancement gate. The
accepted model and distribution remain unchanged. Full validation, test, MK
and publication remain outside this diagnostic.

## Execution and identity controls

Pending final verification:

- Exact prior per-chunk CE controls and paired token/window identities.
- Actual 507 baseline and candidate tensor hashes, 112 native factor merges
  and 395 inherited tensors, followed by final content/file audits.
- Diagnostic script and protocol bindings, finite metric checks and bounded
  CPU derivative/offset/tail controls.
- Final report bytes/SHA256, elapsed time and terminal completion.

The expected report path is `reports/output_calibration_v1.json`; existence
or a partial report alone does not establish successful completion.

## Entropy, slope and curvature

Pending verified per-window and token-weighted aggregate measurements for
each arm: CE/PPL, entropy, `g = CE - H`, and centered-log-probability variance.
No temperature direction or effect size is inferred before those results.

## Rankings, ties and target difficulty

Pending native argmax accuracy, strict-greater target-rank bins and tie counts.
Pending paired base/candidate NLL deltas in the fixed source-surprisal bins
with counts reconciled to the complete target set.

## Interpretation and limits

Conclusions are pending. At beta = 1, positive aggregate slope would indicate
that an infinitesimal increase in temperature locally lowers CE; negative
slope would indicate the reverse direction. Neither sign measures a finite
benefit or authorizes fitting. High entropy alone does not establish
miscalibration, and this analysis cannot establish that global calibration
dominates compression loss or identify one causal mechanism.

The completed [teacher-KL result](TEACHER_KL_COMPENSATION_RESULTS.md) remains a
failed reserved advancement gate. This diagnostic does not change that result
or provide independent fresh quality evidence. Any later correction would
require a separate TRAIN protocol and independent gate.
