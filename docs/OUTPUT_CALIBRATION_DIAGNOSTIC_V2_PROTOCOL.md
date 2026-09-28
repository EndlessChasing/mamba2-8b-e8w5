# Output-calibration diagnostic v2: numerical controls

Declared before the v2 diagnostic. This supersedes only the **cross-process
historical CE equality gate** in the [v1 protocol](OUTPUT_CALIBRATION_DIAGNOSTIC_PROTOCOL.md)
for a new, separately recorded diagnostic. Preserve the failed v1 report and
protocol unchanged. All v1 metrics, bins, derivative mathematics, CPU controls,
model identities and scope limits remain in force, subject to the replacement
execution controls below.

## Reason and fixed identities

The first attempt failed its historical CE equality control before accepting
any window. The [bounded window-0 probe](../reports/output_calibration_anchor_probe_v1.json),
SHA256 `b83fab9ae854e5b3622ee0960fcdc51e3d4e12210651cc5718ea2ea2be067e40`,
then completed with all nine same-process old/new/repeat comparisons exact,
while historical chunks differed. That probe covers source and accepted base
on one window; it does not establish a cause or verify the complete v2 run.
The cause of the cross-process drift remains unproved.

Retain these anchors and all upstream bindings from the v1 protocol:

| Input | SHA256 |
| --- | --- |
| V1 diagnostic protocol | `baa87fa1549e29bf757cca8e18f1ab0feadc18daebffa79402c75c87035bc05d` |
| Historical teacher-KL evaluation | `d8acf27dfa500d18a6d4b25dc6d95e3edaa10a0ccc27f49369cb02ef2649e59a` |
| Final teacher-KL overlay manifest | `cd1c260c3e92146618322431d845101ea7b84468c5fa759769f843a899445300` |
| Frozen independent evaluator | `b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071` |
| Frozen paired CE/KL scorer, `scripts/diagnose_low_rank_generalization.py` | `e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879` |

Bind this protocol and the separate v2 script before execution. Keep the same
already-observed 64 reserved TRAIN windows / 131,008 targets and the same
source, accepted all-small and final teacher-KL exports. This remains fixed
beta = 1 descriptive analysis, with no temperature fit or new candidate.

## Replacement execution controls

1. In one process score paired **source/base**, then paired
   **source/candidate**, using native FP16 head GEMMs and FP32 full-vocabulary
   arithmetic in 64-token chunks plus the 63-token tail. Use fresh state for
   each of the unchanged windows; pair starts, token hashes and counts.
2. On the first window, compare each new paired path against the frozen old
   paired scorer on the same loaded models. Require exact per-chunk source CE,
   student CE and teacher-to-student KL equality, including total target
   counts. These controls cover both student arms and the source control.
   Any mismatch stops the diagnostic; no tolerance replaces same-process
   equality.
3. Across all 64 windows require source per-chunk CE to repeat exactly between
   the base and candidate passes. A mismatch stops the diagnostic.
4. Retain strict export and actual-content identities: all 507 base/candidate
   hashes must match the historical evaluation, with the 112 independent
   native factor merges and 395 inherited tensors audited. Preserve source
   identity checks and final tensor/bound-file audits. No weight change is
   allowed to explain away a numerical-control mismatch.

Record every historical per-chunk CE difference for source, base and candidate,
plus each arm's summed NLL, target count, historical/current PPL and relative
PPL drift. Historical differences are disclosed measurements, not bitwise
matches. The exact same-process controls above remain independent of this
historical comparison.

## Historical numerical consistency ceiling

After all 64 windows, require separately for **each of the three arms**:

`abs(current_PPL / historical_PPL - 1) <= 0.001`

This is an absolute **0.1%** relative PPL drift ceiling. It is ten times smaller
than the existing 1% advancement condition and limits descriptive comparison
with the recorded endpoints. It is **not an advancement gate**, proof of
historical bitwise equality, evidence of model improvement or a basis for
relaxing the original teacher-KL quality result.

If any arm exceeds the ceiling, retain the complete failure evidence and stop
without interpreting aggregate entropy, slope, rank or difficulty statistics
as describing the historical endpoints. Do not adjust the ceiling after seeing
the result. Exact same-process or identity failures also stop interpretation.
Record computation completion separately from numerical-control acceptance.

## Measurements and limits

Keep v1's token-weighted CE, entropy, `g = CE - H`, centered-log-probability
variance, native argmax correctness, strict-greater ranks/ties and fixed source
surprisal bins unchanged. No new metric-driven selection or additional model
quality experiment is introduced. For any accepted descriptive comparison,
report both the new measurements and the historical drift controls.

A positive aggregate slope indicates only that increasing temperature
infinitesimally would locally lower CE; a negative slope indicates the reverse.
Do not fit beta, compute a Newton candidate, sweep temperatures or infer a
finite gain. High entropy alone is not evidence of miscalibration, and this
analysis cannot establish that global calibration dominates the loss.

The windows are already observed. A later correction would need a separate
TRAIN protocol and independent gate. The failed teacher-KL advancement result,
accepted all-small model and existing distribution remain unchanged. There is
no training, checkpoint selection, promotion, validation/test/MK computation
or publication in this diagnostic.
