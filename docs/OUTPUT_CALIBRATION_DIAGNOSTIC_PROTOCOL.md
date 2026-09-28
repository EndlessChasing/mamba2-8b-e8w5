# Fixed output-calibration diagnostic protocol

Declared before this diagnostic's GPU execution. This is a posthoc,
descriptive analysis of the completed [teacher-KL experiment](TEACHER_KL_COMPENSATION_RESULTS.md).
Use the **same already-observed 64 reserved TRAIN windows / 131,008 targets**.
They are not fresh evidence, a temperature-fitting set or a new advancement
gate. Keep all three model exports fixed and evaluate only inverse temperature
**beta = 1**. There is no training, temperature sweep, Newton candidate,
checkpoint selection, promotion, full validation, test, MK or publication.

## Pinned inputs and native execution

| Input | SHA256 |
| --- | --- |
| Completed teacher-KL evaluation | `d8acf27dfa500d18a6d4b25dc6d95e3edaa10a0ccc27f49369cb02ef2649e59a` |
| Final teacher-KL overlay manifest | `cd1c260c3e92146618322431d845101ea7b84468c5fa759769f843a899445300` |
| Frozen independent evaluator | `b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071` |
| Reserved data manifest | `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d` |
| Heldout token file | `a3fc0803052890d973b74f91581a6bfc1d2f93ed28db5725f23c16eb8ec20546` |
| Original raw manifest | `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed` |
| Accepted all-small overlay manifest | `edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006` |
| Original teacher checkpoint | `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb` |
| Tokenizer | `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09` |

Bind the new script, this protocol and every reused code/input file by actual
hash before execution. Reuse the frozen evaluator's strict final-checkpoint,
factor-file and parent checks. Score original source cast to FP16, accepted
all-small, and the final 448-step teacher-KL candidate in one process. Load the
native models; do not use the training bank's functional forward.

Read all 112 strict factor files and independently merge their rounded FP16
values with the accepted base in FP32, then cast effective projections to FP16.
Audit all 507 actual base tensor hashes against the prior evaluation, the 112
merged projection hashes, and the 395 inherited tensors. The complete 507
candidate hashes must match that evaluation. At the end, rehash model contents
and all bound files. An identity mismatch makes the diagnostic incomplete.

Use the unchanged 64 windows in their pinned order. Each stores 2048 tokens,
uses `window[:-1]` and `window[1:]`, and starts with fresh state. Require matching
starts, token hashes, lengths and target counts. Use native FP16 head GEMMs,
the complete 256,000-token vocabulary, and FP32 logits/loss arithmetic in
64-token chunks plus the final 63-token chunk. Teacher and student forwards
have no gradients. Require finite hidden states, logits and metrics.

The original per-chunk CE sums are an execution control: reproduce the exact
source, all-small and candidate CE sums from the pinned prior evaluation,
including exact repeated source CE if two student passes are used. Keep the
same CE reduction for that control; do not substitute a differently ordered
diagnostic reduction. A mismatch fails the control and prevents presenting the
new statistics as describing the previously measured endpoints.

## Fixed per-token measurements

For the native FP16 head's logits cast to FP32, let `z` be the full logit
vector, `y` the observed target, and `p = softmax(z)` at beta = 1. Scaling by
beta is a mathematical derivative only; do not perform another model forward
or score another beta.

| Measurement | Definition |
| --- | --- |
| Target CE / surprisal | `CE = -log(p_y)` in nats |
| Predictive entropy | `H = -sum_j p_j log(p_j)` in nats |
| Local CE slope | `g = d CE(beta*z)/d beta at 1 = CE - H = E_p[z] - z_y` |
| Local CE curvature | `v = d² CE(beta*z)/d beta² at 1 = Var_p(z)` |
| Native top-1 correctness | `argmax(z) == y`, using the native lowest-index tie rule |
| Target rank | `1 + count_j(z_j > z_y)` |
| Target tie count | `count_j(z_j == z_y)`, including the target itself |

Compute curvature using centered log probabilities:
`mu = sum_j p_j log(p_j)` and
`v = sum_j p_j * (log(p_j) - mu)^2`.
This avoids subtracting two large raw-logit second moments and is nonnegative
by construction. Check finiteness and nonnegativity; do not hide invalid
results by clamping. Use full-vocabulary probabilities, not a top-k subset.
Log diagnostic reduction differences separately from the exact CE execution
control if floating-point summation order differs.

Report rank bins **1, 2–5, 6–10, 11–100, 101–1000, >1000**. These bins use
strict-greater rank. A rank-1 target can tie with another top logit and still
fail native argmax correctness. Report tie incidence and tie counts so this
distinction remains visible. Positive scalar beta preserves exact-logit ranks;
this diagnostic does not rerank or alter the vocabulary.

## Paired difficulty bins and aggregation

Use each token's **source** target surprisal to assign the same token to one of
six bins: **[0,1), [1,2), [2,4), [4,8), [8,16), [16,infinity)** nats. These are
fixed left-closed, right-open bins. No arm-specific binning, threshold fitting
or bin selection follows the measurements.

For each bin record the target count, source/base/candidate summed NLL and
the paired candidate-minus-base NLL sum and mean. Empty-bin means are null;
empty bins contribute zero to total counts and sums. Counts must partition
all 131,008 targets, and bin NLL/delta totals must reconcile with the unbinned
totals within the documented floating-point accumulation tolerance.

For every arm and window record token counts and sums/means of CE, entropy,
slope and curvature, argmax-correct counts, rank-bin counts and ties. Record
paired base/candidate NLL deltas per window. Aggregate with actual target
counts: global mean is the sum of token contributions divided by 131,008,
and PPL is `exp(total_NLL / 131008)`. Never average chunk PPL or give the
63-token tail the same weight as a 64-token chunk. Report the global slope's
value and sign plus its window distribution; a small signed value is not
evidence of a practically material correction.

## Correctness controls and completion

Before the once-only GPU diagnostic, run bounded CPU tests on synthetic
logits and labels:

- Compare the analytic slope and curvature with autograd's first and second
  derivatives of full-vocabulary CE at beta = 1.
- Add a common per-token logit offset and verify CE, entropy, slope,
  curvature, rank and argmax invariance within stated numerical tolerances.
- Compare direct totals with chunked totals including a short tail, and
  verify strict-greater rank, ties and fixed-bin boundaries.

Use float64 toy arithmetic for derivative identities and include the actual
FP32 metric path's tolerances in the CPU receipt. Record code hashes and actual
test outcomes. These tests verify the diagnostic calculation; they are not
another model-quality experiment.

The final report must record actual inputs, code/protocol hashes, the prior
evaluation identity, all CE controls, tensor/file audits, per-window and global
statistics, elapsed time and terminal completion. No new quality gate is
introduced. An execution/integrity failure is distinct from a successfully
completed descriptive result. Leave the accepted model and all existing
artifacts unchanged.

## Interpretation limits

With `beta = 1/T`, a positive global `g` means an infinitesimal increase in
temperature locally lowers aggregate CE; a negative `g` means an infinitesimal
decrease in temperature locally lowers it. A zero slope indicates a stationary
point of this scalar objective at the measured endpoint. This does not specify
a finite beneficial temperature, its gain, or behavior on unseen data.

High entropy alone does not establish miscalibration. The slope, curvature,
rank and difficulty breakdown may help distinguish confidence changes from
ranking changes, but cannot prove that global calibration dominates the
compression loss or identify one causal mechanism. The failed teacher-KL
candidate already improved teacher-distribution KL while worsening observed
target PPL; this analysis does not reverse that negative gate or qualify the
candidate for promotion.

Any later temperature fit or correction would require a separate TRAIN
protocol and an independent gate. Do not fit it on this already-observed set.
The diagnostic provides no full-validation/test/MK result, new package-size
claim, compressed-runtime-memory claim or publication authorization.
