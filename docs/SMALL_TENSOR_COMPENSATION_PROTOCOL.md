# All-small-tensor PPL compensation protocol

This protocol is declared before training or candidate quality measurement.
It follows the user's [PPL-priority instruction](PPL_PRIORITY_CONTINUATION.md).
It is a new bounded experiment, with no publication or test-set evaluation.

## Fixed representation and initialization

Train exactly all **393 existing small FP16 tensors / 3,580,928 parameters**:
113 normalization tensors, 56 depthwise convolution weights, 56 convolution
biases, and 56 each of `dt_bias`, `A_log`, and `D`. Shapes and architecture remain
unchanged. Initialize from the actual final norm_compensation_v1 FP16 export,
not its unrounded training masters. Use a fresh optimizer and scaler.

The immutable original E8/W5 parent manifest SHA-256 is
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
Initialization norm-overlay manifest SHA-256 is
`1fa9d35ff56a08e2f7ddf258bfa1fb61bc2e83ae2af404ab2f68a781821ac4d8`;
its small-tensor export SHA-256 is
`8a98319ad513fceaf48673ff4199a8a22b5c1046c64b91bbf3bbbb09f5d84530`.
All 112 E8 projections and both W5 vocabulary matrices remain frozen. No
indices, scales, balances, signs, dimensions, precision or bitrate change.
Teacher is the pinned original source in the existing FP16 runtime, checkpoint
SHA-256 `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.

## Deterministic training data

Use the same pinned WikiText-2 raw **train** split and native SentencePiece
stream as the original calibration (revision
`b08601e04326c79dfdd32d625aee71d232d685c3`). Verify text, tokenizer and full token
stream hashes against the original calibration manifest. There are 2,533,678
train tokens. Do not read validation, test or MK inputs during training.

Create the complete 2048-token grid at starts `2048*j`, dropping the final
incomplete block. Exclude any grid block whose half-open token interval
intersects any of the original 32 calibration windows, which also supplied the
norm-only training inputs. From the remaining eligible starts in ascending
order, choose indices `floor(i*(N-1)/255)` for `i=0..255`. Expected counts:
1,237 grid blocks, 62 excluded, 1,175 eligible, 256 selected. Assert these
counts, internal nonoverlap and zero overlap with the old 32 windows. Save the
new token file and manifest with actual hashes, starts and overlap audit;
no new Hessians are required.

Each selected window has 2048 tokens. Inputs are `window[:-1]` and targets are
`window[1:]`, with zero initial state and 2047 targets. Use four successive
`torch.randperm(256)` draws from one CPU generator seeded 20260927. This gives
**1024 successful updates and 2,096,128 target exposures** over 524,288 stored training-window tokens (524,032 distinct input
positions and 524,032 distinct target positions per pass). Batch size is one window, no gradient accumulation. Save the
complete schedule; only the final 1024-update checkpoint is evaluated.

## Fixed optimization and numerical behavior

Use FP32 master parameters, with FP16 casts inside every native forward and
checkpoint recomputation. Keep model eval-mode semantics with gradients enabled
for the student, no-grad original teacher, the existing unfused FP16 residual
path, and non-reentrant block checkpointing. Preserve exact native computation
for convolution, gates, state transitions, skip paths and norms.

Loss is `0.5*CE(target,student) + 0.5*KL(teacher || student)`, temperature 1,
mean over all 2047 targets, with the complete 256,000-token vocabulary. Use
native FP16 head products followed by FP32 losses, at most 64 tokens per chunk
(last chunk 63). Apply loss scaling once, collect the hidden gradient, then
backpropagate through the backbone once, as checked by the norm experiment.

AdamW: learning rate 1e-4, betas (0.9,0.999), epsilon 1e-8, weight decay 0,
no scheduler. Clip the unscaled master gradient global norm at 1.0. GradScaler:
initial scale 1024, growth 2, backoff 0.5, growth interval 2000. Allow at most
8 total overflow retries (1032 attempts), reusing the scheduled window without
changing masters or optimizer state. Nonfinite forward values, missing gradients,
nonfinite masters or mutation of frozen values fail immediately.

Do not add undeclared parameter clamps, rescale the state, or switch learning
rates. Check finite effective `A=-exp(A_log.float())`, positive decay magnitude,
and finite dt-bias/softplus diagnostics. These checks diagnose numerical safety;
they do not establish long-context recurrence or recall preservation. Save
actual parameter and gradient ranges in the smoke and final reports.

## Correctness, export and resumption

Before training, a new training-only GPU smoke must establish:

1. Initial functional forward equals the native runtime with the norm-v1 export.
2. Checkpointed and uncheckpointed forwards/gradients agree after perturbing
   selected masters, with finite gradient coverage for all 393 tensors, including
   A_log, dt_bias, D and convolution. Declare/check numeric tolerances explicitly.
3. Chunked full-vocabulary loss and hidden gradients match an unchunked short
   reference; a full 2047-target trial update has finite forward/backward values.
4. A changed FP16 export reloads exactly through the unchanged runtime, while all
   114 frozen projection/vocabulary tensor hashes remain exact.

Discard smoke optimizer state. Freeze final training/helper/export/protocol/data
hashes before the real run. Do not modify governing math after training begins.
Save atomic resumable checkpoints every 32 successful updates, on termination,
and at the final step, with masters, optimizer/scaler/RNG, schedule and provenance.
Confirm the prior process has exited before any resume; reject mismatched inputs.

Export only the final rounded masters, keeping precisely the parent's 393 small
keys/shapes and FP16 dtype. Independently verify all 507 parameters, unchanged
E8/W5 inherited files, actual changed small keys, final checkpoint-to-export
rounding, and exported/native forward parity. Use a distinct overlay and report
schema; retain all original receipts unchanged. Invalid/incomplete exports do
not proceed to quality evaluation.

## PPL evaluation and interpretation

Evaluate actual reloaded exports on complete validation: 264,764 next-token
targets in 130 windows of at most 2048 targets. Compare original FP16 source,
original E8/W5, norm-v1 and the new candidate in one process if practical; at
minimum compare norm-v1 and the new candidate in one process and clearly label
any historical source comparison. Require exact matched input hashes and finite
outputs; report target-weighted aggregate NLL, exp(mean NLL), per-window paired
changes and source-relative gaps. A four-window diagnostic may run first but
does not select a checkpoint or replace full validation for a valid final export.

The PPL target remains at most 1.05 times the original source PPL. Any measured
recovery is reported separately from achieving that target. A >=1% decrease
against norm-v1 on full validation is the predeclared threshold for a meaningful
next-stage improvement; smaller changes are reported without pretending the
threshold passed. Stop this fixed recipe after its final evaluation. MK is not
a continuation gate under the user's new instruction; do not infer recall from
PPL or claim Resurface has repaired it. No test split, new container promotion,
public push or model upload is part of this protocol.

## Storage and memory

Count actual replacement archive and manifest bytes, logical resolved-model raw
bytes, and physical parent-plus-overlay storage. The original 393-tensor FP16
payload is 7,161,856 bytes and already exists; fitting it adds no tensor capacity.
Changed values may change Huffman size; no old package size applies automatically.
Measure training peaks separately from expanded-FP16 evaluation residency. Keep
source, calibration, teacher, model binaries and optimizer checkpoints on the
GPU host; retain only code, documentation and small receipts on the Mac.
