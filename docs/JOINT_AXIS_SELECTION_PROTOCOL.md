# Same-capacity joint E8/axis selection protocol

Declared before implementation and measurements. The current best tested
candidate has full-validation PPL 7.977512009035770 versus source
7.334175947318572. Its completed evaluation is
`reports/output_axis_residual_v1_eval.json`, SHA256
`23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4`.
The source +5% target remains 7.700884744684501. PPL repair is the priority;
MK, Resurface and publication remain deferred.

## Fixed change and invariants

Change only offline selection of the existing 20-bit E8/axis representation.
For each eight-dimensional LDLQ query x:

1. Compute the frozen greedy E8-plus-axis result and retain its actual decoded
   vector and code as the incumbent.
2. Enumerate nibble 0 through 15, with offset s[n] equal to the existing signed
   coordinate correction of the fixed amplitude. Query the pinned E8 quantizer
   on x - s[n], form `(base_index << 4) | n`, and decode through the frozen
   decoder.
3. Compare FP32 sums of squared error against x. Replace the incumbent only
   for strictly smaller error. Ties retain the incumbent; ties among shifted
   candidates choose the first nibble in enumeration order.

Batch at most 4096 original query rows at a time, with all 16 offsets in fixed
nibble order. Preserve the incumbent computed by the original greedy routine
on the full query. Reject nonfinite inputs, candidates and errors. Both
`return_idx=True` and `False` must yield the same decoded values.

Use an explicit LDLQ callback adapter passed into the existing frozen
`codec.vector_quantize`. The callback receives the frozen AxisResidualCodebook,
keeps its base and amplitude, substitutes the joint selector and invokes the
same pinned LDLQ function. No monkeypatch, copied transform pipeline or change
to frozen codec/runtime/vendor files. Keep original source BF16 weights,
original TRAIN Hessians, seed `1000 + 2*layer + part` (input=0/output=1),
damping .01, scale .9, two tuning sweeps, buffer width 128 and feedback.

Prove stored amplitude, balance, signs, scale, shape/header bytes and raw file
length match the current corresponding axis file exactly. Only code-plane
contents change. Keep all 395 nonprojection tensors fixed: W4 embedding,
W5 output head and 393 accepted small tensors. No parameter fitting beyond
the unchanged original-weight amplitude rule, new data, layer selection,
temperature adjustment or checkpoint search.

Pinned current input/output manifests:

- Input: `0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85`.
- Output: `2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f`.
- Original calibration: `70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab`.
- Original checkpoint: `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.

## Limits of the local guarantee

Including the greedy incumbent prevents an increase in the measured FP32
Euclidean score for that same query. This is 16 shifted candidates obtained
from the pinned base search, not a proved global search over all 2^20 codes.
Changed choices alter later LDLQ queries; block curvature is not generally
identity, and final inverse transforms/FP16 rounding change the reporting
objective. Neither original-H error nor PPL is guaranteed to improve.

## Stage 1: fixed six-matrix TRAIN screen

Use layers 0, 18 and 55, each input then output: six matrices. First replay
the existing greedy recipe for all six. Require exact stored indices, metadata,
file bytes and decoded FP16 hashes against the pinned current files. Preserve
any failure and resolve it before scoring the new selector.

Then run joint selection for all six, with no sub-selection. Save actual
files and independent disk readback, finite FP16 hashes, original-H errors,
source/Hessian/code bindings, timing and peak allocation. Every source Hessian
must match both original calibration and original parent identities. CPU
tests cover a small independent exhaustive-codebook oracle, ties, row batching,
return modes, actual E8 boundary-code decoding, malformed input and disk layout.

Use final decoded FP16 weights to compute

`J = tr((W_hat - W) H_original (W_hat - W)^T)`.

The reduction for each matrix is `r = 1 - J_joint / J_greedy`, equivalently
`1 - (relative_RMS_joint / relative_RMS_greedy)^2` with the same denominator.
Do not compare percentages of RMS as percentages of squared error.

All three conditions are required to justify the more expensive full build:

- Median squared-error reduction across all six is at least **5%**.
- At least **four of six** improve by at least **2%**.
- No matrix worsens by more than **0.1%**.

This stronger expansion screen is chosen before results because a prior
approximately 1.9% reconstruction improvement produced only 0.46% PPL gain.
It is a compute-allocation rule, not a prediction of language quality.
Missing/nonfinite values, mismatched identities or failed roundtrip reject
the screen. Handle zero baseline error explicitly; it cannot provide a
positive relative reduction. If the screen fails, stop this fixed recipe;
do not lower thresholds, change layers, add tuning sweeps or combine matrices
from different selectors after seeing results.

## Stage 2: all 112 matrices, only after a screen pass

Create a separate complete overlay with all 112 joint-selected projections.
Reuse the six exact pilot files and their receipts; do not rerun or choose
between pilot checkpoints. Apply the same recipe to every other matrix.
Do not fall back to an old matrix because its original-H error worsens.
Record all errors and actual changed tensor identities, including any unchanged
decoded matrix, while proving all 112 replacement files are used.

Bind the source/current manifests, original calibration, all Hessians,
code/environment hashes, exact file and decoded identities, and the five
inherited data files (embedding, head, small tensors, configuration, codebook).
Use exclusive new paths, incremental receipts and final input rechecks.
Count actual metadata separately. The current raw data total is
3,138,928,792 bytes; exact header/payload lengths should keep that raw total
unchanged. A new manifest adds bytes. Huffman size may change and is unmeasured.
Reused pilot files must not be counted twice in resolved or physical storage.

## Stage 3: independent paired PPL

Only a complete verified overlay enables language evaluation. Evaluate the
current all-axis model, the joint candidate and a restored current-model
repeat, in that order, on the same 64 previously observed reserved TRAIN
windows from `teacher_kl_compensation_v1`. Pin its manifest
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
Use the established stored order, fresh state per window, 2048 stored tokens /
2047 targets, native FP16 backbone/head, FP32 full-vocabulary CE in chunks
of 64 plus a final 63, and FP64 host sums: 131,008 targets per arm.

Independently decode actual files. Verify all 507 tensor identities before
and after each arm and all 395 nonprojection identities remain fixed.
Restore the baseline by streaming stored original axis files, without a
second dense copy of all 112 matrices. Require exact baseline CE repeat.
Historical drift is descriptive; advancement uses same-process pairing.

Require candidate development PPL at most 99% of paired baseline. A failure
stops this recipe without full validation or extra tuning. Only after a pass,
evaluate source FP16, current all-axis baseline and freshly reloaded candidate
on the established 130-window /264,764-target full validation, including the
572-target final window and 60-token final CE chunk. Bind the exact existing
token/window plan. Report at least 1% paired gain and source +5% as separate
conditions. These datasets have informed development and are not fresh tests.

No automatic promotion or publication. Any stored-file gain is separate from
runtime residency: evaluation still decodes to FP16. Preserve every outcome.
