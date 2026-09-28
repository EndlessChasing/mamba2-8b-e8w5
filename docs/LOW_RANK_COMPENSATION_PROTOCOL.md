# All-projection rank-4 PPL compensation protocol

Declared before this experiment's GPU smoke, training or candidate quality
measurements. Freeze implementation/data bindings before execution.
This is one bounded rank-4 experiment, not a rank sweep.
The completed prototype recipe failed its predeclared improvement gate and is
stopped; its prototype tables and optimizer are not inherited here. MK is deferred and
publication remains held.

## Fixed base and representation

Use the accepted all-small model: original two-sweep E8/W5 parent plus the final
1024-update small-tensor export. Parent manifest SHA256:
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
Small-overlay manifest SHA256:
`edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006`.
Small-tensor file SHA256:
`15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4`.
Freeze all 507 base parameter tensors, all E8 indices/balances/signs/scales,
both W5 vocabularies, the 393 small tensors and the native architecture.
Decode the base once, with no decoder gradients or repeated decoder training.

For each of all 112 projections, add factors `B[m,4]` and `A[4,n]`. The actual
shapes are 56 input projections `[18560,4096]` and 56 output projections
`[4096,8192]`. This gives 224 FP32 optimizer master tensors with **7,827,456
values**: 5,074,944 in B and 2,752,512 in A. Forward/export factors are FP16.
There is no separate alpha, bias, norm update or prototype-table correction.

Initialize with a dedicated CPU `torch.Generator().manual_seed(20260928)`.
Visit numerical layers 0 through 55, `in_proj` then `out_proj`. For each matrix,
draw exactly `A = torch.randn((4,n), generator=g, dtype=torch.float32) *
(1/math.sqrt(n))`; initialize B to FP32 zeros without consuming RNG. Preserve
these FP32 A values as masters. Canonical checkpoint names are
`layerN.in_proj.B`, `layerN.in_proj.A` and corresponding `out_proj` names.
Use fresh AdamW and scaler states, not previous experiment checkpoints.

## Native merge and differentiation

Inside every native forward and non-reentrant checkpoint replay, round both
masters to FP16 and materialize exactly:

```python
B16 = B_master.half()
A16 = A_master.half()
W_eff = (W0.detach().float() + B16.float() @ A16.float()).half()
```

The factor product and addition are FP32. Disable autocast and require
`float32_matmul_precision='highest'` with TF32 disabled. Use ordinary autograd
through these casts and products, with no custom gradient or base gradient.
Feed the materialized FP16 weight into the native projection. Do not substitute
a two-linear residual branch, fuse a differently rounded expression, train the
base, or change native norms/gates/state transitions/residual precision.

There is no zero-factor shortcut. A zero B gives zero correction and exactly
zero A gradients initially, while B can receive gradients. FP32 addition may
canonicalize a base negative-zero bit; initial merge checks distinguish signed
zero from a changed numerical value. The base tensor itself must retain every
bit and object identity. Final FP16 rounding can make the effective weight
difference higher rank than four; rank four describes the pre-round factor
product, not a guarantee about the rounded difference matrix.

## New, fixed TRAIN data

Use the original tokenizer and pinned WikiText-2 raw TRAIN stream, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. There are 2,533,678 tokens, joined
with two newlines and no automatic special tokens. Verify full dataset,
text/token hashes and tokenizer identity against the original calibration.
Do not read validation, test or MK inputs during fitting.

From the complete 2048-token grid, exclude every block whose half-open stored
token interval intersects any of these prior intervals:

| Provenance file | Intervals | SHA256 |
|---|---:|---|
| `calibration/v1/manifest.json` | 32 | `70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab` |
| `training_data/small_compensation_v1/manifest.json` | 256 | `88d4dcd9f468a50137f25f7afdc101fd2635f254bdd0d4b194ecf252c7116709` |
| `reports/projection_crossmoment_pilot_v1.json` | 32 fit + 16 held-out TRAIN | `034a161df40fa4dac5be99b25545f9087da6695accc414a852fafe0e47228f87` |
| `training_data/prototype_compensation_v1/manifest.json` | 32 | `a485424d1b7894eb1d5a6361ab4c25fdb4f6e5bedc56d7a15a7e4add2c454bcd` |

Expected counts: 1,237 grid blocks, 398 excluded, **839 eligible**. Choose
`eligible[floor(i*838/255)]`, `i=0..255`, in ascending order. Assert 256 unique
selected windows, zero internal/prior stored-token overlap, first start 8192,
last start 2523136 and minimum start gap 6144. Save CPU int64 `[256,2048]`
tokens and a manifest, including all starts and each window's token digest.

For each window, inputs are `window[:-1]` and targets `window[1:]`; reset state
and use 2047 targets. Four successive `torch.randperm(256)` calls from a
**separate** CPU generator seeded 20260928 give 1024 successful updates, batch
size one window and no gradient accumulation. Record the flat 1024-window-ID
schedule. Capacity/exposure totals are 524,288 stored tokens, 524,032 distinct
input positions and 524,032 distinct targets per pass, and **2,096,128 target
exposures** across four passes. Smoke and failed/overflow attempts are separate.

## Fixed objective and optimizer

Teacher: the original source checkpoint in the existing FP16 runtime, SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
Keep teacher no-grad and student eval-mode behavior with gradients enabled for
the factors. Use the existing native FP16 residual/state computation.

Objective: `0.5*CE + 0.5*KL(teacher || student)`, temperature 1, mean over all
2047 targets and sum over the complete 256,000-token vocabulary for KL.
Native FP16 head GEMMs precede FP32 losses. Reuse checked staged loss/hidden
backpropagation with at most 64-token chunks (last chunk 63), weighted by target
count, applying the loss scale exactly once.

AdamW: learning rate **3e-4**, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay
0; no scheduler. Clip the unscaled global factor-master gradient norm at 1.
GradScaler starts at 1024, growth factor 2, backoff 0.5, growth interval 2000.
Allow at most 8 total overflow retries, at most 1032 attempts. Retry the same
scheduled window with no master/optimizer mutation. Record actual attempted
target exposures and retry history. Nonfinite forward values, missing
gradients, nonfinite masters or frozen-base mutation fail immediately.
At zero-B initialization, zero A gradients are expected, not a failure.
No extra epochs, learning-rate changes, alpha, clamps or checkpoint selection.

## Discarded correctness and feasibility smoke

Before training, CPU tests and one independent discarded GPU smoke must prove:

1. Fresh initialization has the exact seed/order/shapes, all B zero and finite
   FP16 casts. All 112 merged weights numerically equal the base; record any
   signed-zero-only bit changes. Initial hidden states and head logits match
   the native base bitwise on a fixed short TRAIN probe. A failed match stops
   execution for diagnosis; do not silently loosen precision or tolerance.
2. On native blocks 0/18/55, compare checkpoint/non-checkpoint outputs and
   factor gradients with deterministic nonzero factor perturbations, restoring
   initialization afterward. Require bitwise equality on the same process and
   inputs; preserve failure receipts. Do not retain a full 56-block dense
   no-checkpoint training graph just to run this check.
3. One complete 2047-target update completes with finite forward/loss/hidden
   gradients and all 112 B-master gradients finite/nonzero; all 112 A gradients
   are present and exactly zero before that first update. On a subsequent
   short TRAIN backward using the updated factors, both families receive
   finite/nonzero gradients per projection. Report timing, peak allocated and
   reserved GPU bytes, scaler behavior and factor/gradient ranges.
4. Serialize all 112 actual rounded FP16 factor pairs, reload each through the
   strict CPU format, and prove exact master-rounding/readback equality. Merge
   them using the declared deployment arithmetic, install into a native model,
   and require hidden/head output equality with the functional training path.
   Verify all 507 original base objects and contents remain unchanged before
   and after smoke; only temporary effective projection weights are installed.

Discard every smoke update and optimizer state. Freeze passed smoke, code,
protocol and input hashes before real training. Any implementation fix requires
a new smoke and preserves previous receipts; it is not permission to tune
learning rate or select quality outcomes.

## Durable progress and final export

Keep per-attempt journals and atomic checkpoints every 32 successful updates,
on caught termination and at update 1024. Store FP32 masters, Adam/scaler/RNG
states, the full schedule, history and all input/code identities. A numerical
failure is terminal. This bounded driver has no resume or replay: require fresh
work/output paths, preserve interrupted runs as incomplete attempts, and do not
reuse their optimizer states or silently repeat exposures. Diagnostic durable
checkpoints do not authorize continuation or an increased exposure budget.

Evaluate only the final 1024-update export. Each projection file uses the
existing 32-byte versioned header followed by FP16 row-major B then A. Exactly
112 files contain **15,654,912 value bytes + 3,584 header bytes = 15,658,496 B**.
Count the candidate manifest, loader and packaging metadata separately; no
entropy savings or same-capacity claim is assumed. The frozen base is required.
Keep large optimizer/model artifacts on the GPU host, not the Mac.

The final manifest must bind the accepted raw/small parents, source/tokenizer,
data, protocol/code, passed smoke, final checkpoint and training receipt.
Independently inspect every final master's FP16 value against its file and
verify expected dimensions/rank, exact file inventory/length/hashes/finiteness,
507 base content hashes, and the 112 effective weights. Native evaluation must
load the base plus factor files independently, without using the training bank
as its inference implementation.

Avoid hash cycles: this protocol declares algorithms and names the preparation
script, but does not contain that script's hash or future data-manifest hash.
`scripts/prepare_low_rank_training_data.py` records its own hash, this protocol's
hash and immutable upstream identities in the data manifest. The training
binding then records those actual hashes. No governing input changes once the
passed smoke and training bindings are frozen.

## Independent PPL outcome

In one process, score original source, unchanged accepted all-small base and
the final independently reloaded candidate on the same frozen full validation:
264,764 targets, 130 fresh-state windows, up to 2048 targets/window, native FP16
head and FP32 loss, chunk size 64. Require paired token hashes, finite outputs,
per-window NLLs and target-weighted aggregate NLL/PPL. No test split.

Report two separate gates: meaningful repair requires
`candidate_PPL <= 0.99 * same_process_all_small_PPL`; the original quality
target requires `candidate_PPL <= 1.05 * same_process_source_PPL`. The first
does not imply the second. Preserve negative results and stop this fixed recipe
after final evaluation; no silent prototype promotion or automatic rank sweep.
MK remains deferred/nonblocking. PPL gain would not establish recall recovery,
minimum model size, compact GPU residency, an accepted release or permission
to publish.
