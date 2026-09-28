# Existing-norm compensation protocol

This is a new, bounded experiment following the failed eight-sweep repair.
Publication, uploads and pushes remain on hold. The original two-sweep candidate
is the parent; the eight-sweep candidate is not used. This protocol is declared
before training or inspecting a trained candidate's validation scores.

## Hypothesis and fixed representation

E8 projection errors dominate the first candidate's language-quality loss.
Training the model's existing normalization gains may compensate for part of
that error without additional inference parameters or a new weight format.
This is a hypothesis, not an established repair or a claim of identical archive
size. Normalization gains cannot independently reconstruct arbitrary weight errors.

Train exactly these **113 existing tensors / 692,224 scalar parameters**:

- 56 block `backbone.layers.{i}.norm.weight` tensors: 229,376 values.
- 56 gated `backbone.layers.{i}.mixer.norm.weight` tensors: 458,752 values.
- `backbone.norm_f.weight`: 4,096 values.

Keep the other 394 parameter tensors frozen: 112 E8 projections, two separate
W5 vocabulary matrices, and 280 small FP16 tensors. Do not change the state
dimensions, architecture, decoder, quantizer, vocabulary, precision or bit allocation.
Export all 393 small tensors in a new `other_fp16.pt`, with exactly the parent's
keys/shapes and FP16 dtype; the other 280 must be bitwise unchanged. Some selected
norms may export identically if their updates remain below FP16 resolution.

Use the original quantized parent `artifacts/e8w5_v1`, manifest SHA-256
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
Use the pinned official source as an FP16 teacher, source checkpoint SHA-256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
Keep the frozen runtime/codec/evaluation sources unchanged and record their hashes.

## Training data and budget

Use only the existing 32 training calibration windows, each 2048 tokens:

- Calibration manifest SHA-256:
  `70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab`.
- `calibration/v1/calibration_tokens.pt` SHA-256:
  `ebeb62135d074ba41fcb44cc64f8fe8aa09f452dfd28183c1b28f84645e6f577`.
- Flattened selected token digest, int64 little endian:
  `893bf7df37895a98e00116db9cb7a3189e5ee2e8f991e35594b970087958db34`.

For each window, inputs are `window[:-1]` and targets are `window[1:]`:
2047 input tokens and 2047 next-token targets. Start from zero state each time.
Do not concatenate windows or append tokens. There are 65,504 scored target
positions in one pass; four passes give **128 successful optimizer updates and
262,016 scored target exposures**. Microbatch is one window, no gradient accumulation.

Seed a CPU `torch.Generator` with 20260927 and use successive `randperm(32)`
calls for the four pass orders. Record the complete schedule. Keep evaluation
mode semantics with gradients enabled for the student; the teacher uses no-grad.
No validation, test or MK examples enter training. No intermediate checkpoint
is selected using quality scores: evaluate only the final 128-update export.

## Fixed optimization objective

Use FP32 master parameters initialized from the parent's FP16 norms. Each
student forward and checkpoint recomputation must cast those masters to FP16;
this is the same precision exported to the existing decoder/runtime.

The per-window objective is

`L = 0.5 * CE(target, student) + 0.5 * KL(teacher || student)`, temperature 1.

Both terms are means over the 2047 target positions. KL sums over all 256,000
vocabulary entries before taking the token mean; do not divide by vocabulary size.
CE uses the target token's full-vocabulary log probability.
Use the frozen FP16 head matmul as in the public runtime, then convert its logits
to FP32 for softmax/log-softmax and loss reductions. Process the
frozen teacher/student heads in chunks of at most 64 tokens; accumulate loss
sums divided by 2047. The last chunk has 63 targets and must not receive the
same weight as a full 64-target chunk. Do not use top-k teacher targets.

Teacher and student hidden states may be computed once per window. For memory,
the head loss may backpropagate into a detached student-hidden leaf per token
chunk, then pass its collected gradient through the student backbone once.
Apply loss scaling exactly once: backpropagate the scaled chunk losses into the
detached hidden leaf, pass its already-scaled gradient through the backbone,
then unscale the norm optimizer gradients before clipping. Do not multiply the
staged hidden gradient by the scale again. The full-vocabulary loss must remain
algebraically equivalent to the objective.
Its student-logit derivative is
`(p_student - 0.5 * (one_hot_target + p_teacher)) / 2047`.

Optimizer: AdamW, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`,
weight decay `0`, no learning-rate schedule. Clip the unscaled FP32 master
gradient global norm to `1.0`. Do not optimize any frozen tensor.

Use CUDA GradScaler with initial scale 1024, growth factor 2, backoff factor 0.5
and growth interval 2000. FP32 leaf gradients do not imply that the installed
native norm backward is FP32 throughout; it returns some local weight gradients
in FP16. Track finite values, gradient coverage and loss scale explicitly.

For a detected gradient overflow, skip the optimizer update, reduce the scale,
and retry the same scheduled window without changing masters or optimizer state.
Allow at most **eight overflow retries across the entire run** (at most 136
attempts); otherwise terminate with a failed receipt and no complete candidate.
Count attempted target exposures and successful updates separately. Nonfinite
forward outputs/losses, missing gradients, mutation of frozen parameters or
incorrect coverage fail immediately. Do not silently restart with different
hyperparameters or export a partial run as complete.

## Correctness and continuation gates

Before full training, perform an explicitly labeled smoke on training tokens:

1. Verify source, parent, calibration, protocol and implementation bindings.
2. Check that the training forward with initial rounded masters matches the
   unmodified parent's hidden/output computation in the same process.
3. Use non-reentrant checkpointing (`use_reentrant=False`), because the first
   block's frozen embedding activations need not require gradients. Compare
   checkpointed and uncheckpointed execution/gradients on a manageable
   training-only slice, ensuring recomputation re-enters the functional parameter
   mapping and does not replay stale norm values. Record numerical tolerances.
4. Verify the chunked full-vocabulary loss and hidden gradient against the
   unchunked expression on a short slice; verify every selected master gets a
   finite gradient and no frozen weight gets one.
5. Verify a changed rounded-norm export reloads into the unchanged public runtime
   with matching output, unchanged other parameters and exact tensor coverage.

The smoke is not a quality-selection stage. Discard its trial optimizer state
and begin the declared schedule from the parent. Preserve failures and receipts.
Load training tensors with ordinary autograd semantics, and use `no_grad` for
the teacher; do not create inference-mode tensors that the student backward
may need to save.
Implementation repairs are permitted before training starts; record final code
hashes and the successful smoke's identity. Once training starts, do not change
the protocol or code governing its math; stop and document any required repair.

Save small atomic resumable checkpoints at least every eight successful updates,
including FP32 masters, optimizer/scaler state, completed schedule position,
overflow count and exact input/code bindings. Resume the same live job only after
confirming it has terminated; a polling timeout is not evidence of termination.
On resume, reject mismatched provenance and preserve corrupt/incomplete files
for inspection. Original source, parent and prior reports remain immutable.

## Export and quality evaluation

Use a distinct overlay schema for replacing only `other_fp16.pt`. Bind the full
parent manifest and inherited-file ledger, selected parameter list, actual changed
keys, source/tokenizer/calibration identities, training schedule and optimizer
recipe, code hashes, this protocol, smoke and training receipts, and every export
hash/byte count. CPU verification must reject unsafe or additional paths and
changed non-norm tensors. Loading must resolve all 507 tensors / 8,236,999,680
parameters. The source teacher is not a candidate inference dependency.

After reloading the actual FP16 export, evaluate the parent and candidate in the
same process on the existing four validation windows (4096 targets) and all
12 normal plus 12 target-removed public MK cases. Require all of:

- Candidate PPL at most `0.99 * same_process_parent_PPL`.
- Normal MK correct count no lower than the parent.
- Target-removed matches no higher than the parent.
- Exact input/parameter identities and finite outputs throughout.

If any condition fails, retain the negative result and stop this fixed training
recipe; do not run full validation, lower the threshold or select another step.
If all pass, run the complete validation split (264,764 targets, windows of at
most 2048 targets) for original FP16 source, parent and candidate in the same
process. This confirmation is separate from the original source-relative +5%
PPL / recall quality target. Report both conclusions without conflating them.
No test split is used in this experiment; earlier test results have already
been inspected and remain visible. Twelve normal MK cases do not establish
full-test recall preservation. This experiment does not authorize publication.

## Storage and memory reporting

No new tensor fields or inference parameters are introduced; the 1,384,448-byte
norm tensor payload already existed. Measure the actual replacement archive
size, all manifest/provenance bytes and logical resolved-model size. Changed
values may change entropy-coded size. A new Huffman package must be measured
before claiming the first candidate's 2.666GB distribution size for this export.

Report physical parent-plus-overlay experiment storage separately, with optional
optimizer checkpoints counted as training artifacts. Teacher, FP32 masters,
optimizer moments and activation workspaces are training memory. The current
quality runtime expands weights to FP16; none of these measurements establishes
compact inference residency. Keep model/calibration binaries on the GPU host.
