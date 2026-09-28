# Resurface readout repair: fixed first training experiment

Declared 2026-09-28 before preparing these training examples or fitting the
adapter. This protocol supplements the immutable MK_RESURFACE_PROTOCOL.md.
No publication is authorized. A failed candidate is retained as an experiment;
the accepted readapted base is not replaced unless all joint gates pass.

## Base and insertion

Use the actual final448 readapted model/export and all provenance checks from
Stage1: manifest SHA256 3ede80b2883d8b8987cb3e2198fd8badc2dbbb5af03c8fe6cac7f5f9f5b62047,
full PPL report SHA256 3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4.
Freeze all507 parameters of the pure56-layer Mamba2 model. A separately loaded,
identical current compressed/readapted model supplies the prose KL teacher.
The teacher is not the original FP16 source. Audit actual507 tensor hashes in
both models before/after fitting and parameter identity/storage/version during
fitting. Native FP16 computation and existing SSD kernels stay unchanged.

At each layer's explicit gated RMSNorm input, apply
`y' = y + sigmoid(w dot u + b) * g * (V @ y)` across128 heads of64 channels.
`u` is the actual4096-wide mixer input. The native scan has already added D*x;
this is a Resurface-inspired **post-D** variant, not an exact port of the
original pre-D adapter. Keep native grouped gated normalization and out_proj.
Apply the same hook during prefill and cached decode. Use no EMA or additional
recurrent cache. Downstream recurrent state values may change.

All56 layers have V[128,128], g[128], w[4096], b[]: 224 tensors and1,154,104
parameters. FP32 masters use4,616,416 bytes; the FP16 inference payload uses
2,308,208 bytes before container headers. Casting occurs inside every forward
and checkpoint replay. Keep the external bank outside the base model's named
parameter inventory. Cold initialization is V=0, g=1, w=0, b=-4. No old2.7B
adapter or pretrained readout is used. Gates remain soft sigmoid in training
and every MK/PPL evaluation. Hard thresholds, task flags and switching off
the adapter for PPL are outside this experiment.

## Data and labels

Native SentencePiece tokenizer SHA256
5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09,
without automatic BOS/EOS. Three frozen public-v1 templates, N16 and64.
The templates are shared deliberately; there is no claim of unseen-template
generalization. All intervals below are half open.

- TRAIN:1536 normal cases,256 per N/template cell, Python random.Random seed
  2026092801, keys[300000,380000). Sample unique values uniformly from the union
  [681000,690000),[781000,790000),[881000,890000),[981000,990000).
  Query position uses an independent rng.randrange(N) after keys and values.
- DEV: frozen public validation seed834901,384 normal+384 target-removed.
  This is already observed development data and overlaps earlier small probes.
- CONFIRM: seed2026092802,384 normal+384 target-removed,64 per cell,
  keys[400000,480000); value union[691000,700000),[791000,800000),
  [891000,900000),[991000,1000000). Query position=floor(sample*(N-1)/63).
  Removal replaces only the queried record with key490000+sample and
  value990000+sample, keeping the query and absent expected answer fixed.

Each split owns an independent RNG stream, files and manifest. Check complete
record/query rendering, unique IDs/token hashes, no split ID/prompt/token
overlap, numeric-range/control disjointness and actual file hashes. Training
must open only the literal TRAIN split; it must verify the manifest's protocol
SHA as well as actual data and helper hashes. Confirmation files/scores are
loaded only in the final confirmation stage. Manifest generation itself does
not authorize fitting on evaluation inputs.

TRAIN full text is prompt + one space + six-digit answer. Require encoded
prompt to be the exact prefix of encoded full text, and decoded suffix to equal
the full answer after whitespace stripping. Train all answer tokens using the
full256K vocabulary. If prompt length is P and full length L, supervise hidden
positions P-1 through L-2 against target token positions P through L-1.
Do not fit prompt tokens, EOS, only the first answer token, or restricted
answer-choice logits. Require prompt+12 and full length within4096.

Prose uses only the existing448 TRAIN windows from
training_data/teacher_kl_compensation_v1. Manifest SHA256
facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d;
train token SHA256 e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233.
These windows were already used by earlier compensation research; this reuse
is disclosed. Heldout64 and complete-validation130 windows are never fitted.

## Fixed optimization

Use torch CPU randperm(1536), seed2026092803, once. At successful update index
j=0..1535, take that MK example and a512-token prose segment. Prose window is
the existing seed20260928 train_schedule()[j mod448]; segment start is
512*floor(j/448) mod2048. Each prose segment has511 next-token targets.
There are784,896 successful prose target exposures. The exact MK answer-token
exposures are counted from the prepared data and schedule.

Per-step objective:

`mean MK answer CE + 0.5 mean prose CE + 0.5 mean KL(teacher || student) + 3 C`.

Temperature1. For prose only, C is the mean over56 layers of
`mean_tokens BCE(router_logits,0) + 10*relu(mean_tokens sigmoid(router_logits)-0.006)`.
No closure or opening loss on MK. Stage full-vocabulary prose logits in chunks
of64 tokens, divide all sums by the total511 targets, and backpropagate the
combined hidden and gate derivatives through each retained backbone once.
MK and prose gradients accumulate into one optimizer update.

AdamW: V/g initial LR1e-4, w/b3e-4; betas(.9,.999), epsilon1e-8,
weight_decay0, global gradient clip norm1. At update j multiply both LRs by
`0.1 + 0.9*(1+cos(pi*j/1535))/2`. Use FP32 masters, native FP16 forward,
model.eval(), block checkpointing, and no autocast over the native model.
GradScaler starts1024, growth factor2, backoff0.5, growth interval2000.
Require1536 successful updates; at most8 overflow retries/1544 attempts.
An overflow retries exactly the same MK/prose pair with unchanged masters and
optimizer state. Nonfinite forward values fail immediately. No automatic
resume or silent shortening. Save checkpoints every384 successful updates
and at terminal success; final1536 is the only candidate. No intermediate
checkpoint selection or development inspection during fitting.

## Preflight and artifact verification

Before formal training: CPU input/label/schedule checks with CUDA hidden;
native GPU zero-adapter prefill/cached-step parity including FP16 caches;
perturbed-adapter native/checkpoint forward and all-family gradient checks;
correct staggered zero-V gradients; full-vocabulary multi-token MK CE and
chunked prose CE/KL/closure parity; actual serialized FP16 export/reload parity.
Smoke updates are discarded. Formal training starts in a fresh process.
Record all source/protocol/data/model hashes and process identity. Recheck
bound inputs at terminal completion. Export the actual FP16 adapter, verify
its bytes/hash, tensors and master-to-FP16 rounding, and test native reload.

## Fixed evaluation and joint gates

Use the actual serialized candidate, with identical soft gate policy for all
tasks and no fitting. DEV MK has384 normal+384 controls and unchanged full256K
greedy max12-token scoring. DEV prose has all64 existing heldout windows.
Evaluate paired current, active candidate, and restored-current controls.
Require positive normal MK gain, no increase in target-removed false recall,
and aggregate PPL no greater than the paired current baseline. The historical
DEV reference is7.6242504268035765; exact current per-window CE/output replay
must agree with the prior accepted measurements and this run's controls.

If DEV passes, run complete paired130-window/264,764-target PPL. Require PPL
no greater than the paired current baseline (historical7.622396587826496),
with current repeat/control validation. No source+5% allowance or epsilon is
added to the quality threshold. Report every window, finite checks and totals.
If full PPL passes, run the independent CONFIRM MK set once on current and
candidate, with fresh FP16 caches, same prompts/IDs and target-removed controls.

Final recall gate: gain>0, exact two-sided McNemar p<0.05, and positive
conservative95% lower bound on paired accuracy improvement. Compute the bound
as the one-sided97.5% Clopper-Pearson lower bound on gained/n minus the
one-sided97.5% upper bound on lost/n (Bonferroni coverage at least95%).
Require no increase in target-removed false recalls. Report all counts,
per-cell results and intervals. Full PPL and recall are separate gates and
both must pass. No universal claim about prose, recall or exact bitwise PPL
equality follows from these finite tests.

If any stage fails, preserve and report it without promoting the adapter.
Further recipes require a new declared experiment and honest disclosure of
already observed development/confirmation data. This is the first bounded
recipe, not authorization to silently tune on the confirmation set.
