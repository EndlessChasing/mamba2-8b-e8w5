# Projection-focused PPL repair

Publication remains held. Recall is deferred by the user's explicit direction;
Resurface recovery has not been measured. The best achieved compact candidate
remains the final all-small E8/W5 overlay; this page records the next diagnosis
and bounded same-bitrate repair pilot.

## Full-validation interaction: completed

Same process, fixed W5 embedding and output head in every arm,264,764 targets,
130 windows, up to2048 targets/window. Native FP16 quality runtime and original
evaluation math are unchanged. Source FP16 projections are a diagnostic
intervention requiring dense source weights.

|393 small tensors|112 projections|PPL|Mean NLL|
|---|---|---:|---:|
|Original|E8|8.836489701|2.178889705|
|Trained|E8|8.359912174|2.123447922|
|Original|Source FP16|7.415378090|2.003555964|
|Trained|Source FP16|7.412136849|2.003118771|

With the trained small tensors retained, restoring source projections lowers
PPL11.3371% and mean NLL0.120329150 nats/target; all130 windows improve. With
source projections, training the small tensors changes PPL by only−0.04371%
(65 windows improve,65 worsen). Thus there is substantial conditional projection
headroom and no large incompatibility between the adapted small parameters and
restored source projections.

The NLL interaction is+0.055004591 nats/target using the predeclared restoration
sign: restoring projections helps less after small-tensor adaptation. This is
consistent with the small tensors already compensating part of E8's error.
It does not show that a fixed-bit E8 code can reach the restored score.

Both direct-versus-swapped endpoint probes pass exact FP16 hidden/logit hashes.
The507 source and507 parent references were restored; final parent114 and
source112 large-tensor hashes remain exact. Complete loaded/export provenance,
window identities and target-weighted PPL checks pass. Runtime298.728s;
peak allocated33,662,853,632 bytes for this two-model FP16 diagnostic.

Tiny historical endpoint differences are retained in the receipt: parent
−0.0008005%, trained E8+0.0005338%, versus the earlier full-validation process.
Only current-process pairs determine the interaction. No fifth true-source arm
was run; the earlier source PPL7.334175947 and source+5% target7.700884745 are
historical references, not extra outcomes of this four-arm run.

Evidence:

- Protocol: `docs/SMALL_PROJECTION_INTERACTION_PROTOCOL.md`, SHA256
  `8000651017a92085c7221fb2a4203fc34564a467fd5b9f754ae7a2d7449501d4`.
- Script: `scripts/diagnose_small_projection_interaction.py`, SHA256
  `d502aee9790e0ec26423b76e3f488c3b6adfe8f58866662c6d40fa1f2364f203`.
- Receipt: `reports/small_projection_interaction_v1.json`, SHA256
  `540fb0048fce5d8ea3683f4b89aefc4c6ecf21ae4f3d236d6fa646ebf725dc03`.

## Candidate-input cross-moment pilot: running

The predeclared pilot compares current E8, original weights requantized under
the candidate's input covariance, and a teacher-output cross-moment target
requantized under that same covariance. Six fixed matrices: layers0/18/55,
in_proj/out_proj. Raw file capacity and codec settings remain unchanged.

32 fit and16 heldout TRAIN intervals are disjoint from the previous32+256
fitting windows. Candidate inputs and native FP16 teacher outputs are paired
at identical token positions. The target uses an anchored ridge solve; source
and projection roles, data selection, native-output error thresholds and joint
six-matrix TRAIN PPL gate were declared before outcomes.

Twelve CUDA-disabled CPU checks passed, including independent least-squares
math, rank-deficient inputs, actual teacher FP16 rounding, independent moment
collection and native replacement scoring. The runtime also performs a discarded
32-token GPU pairing check before collecting the full pilot statistics.

Protocol: `docs/PROJECTION_CROSSMOMENT_PILOT_PROTOCOL.md`, SHA256
`c330e52da1295eb9239dc30104c49ddefb43d8f76db87e368d5548a65d2aa511`.
No pilot quality result or larger expansion is claimed yet.
