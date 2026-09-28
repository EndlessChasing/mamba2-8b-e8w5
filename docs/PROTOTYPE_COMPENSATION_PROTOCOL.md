# Language-loss compensation through learned E8-derived prototypes

Declared before GPU training or candidate-quality outcomes. The previous
cross-moment pilot failed despite lower local MSE; this experiment optimizes
end-to-end language loss directly. Publication remains held; MK/Resurface and
test-set scoring are deferred.

## Frozen baseline and representation

Baseline: original two-sweep E8/W5 parent plus final1024 all-small overlay.
Parent manifest SHA256:
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
Small manifest SHA256:
`edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006`.
Keep all507 baseline parameter tensors, all112 E8 index arrays, signs, balance
vectors, scales, rotations, both W5 vocabularies and393 small tensors frozen.

Learn one256x8 correction table for each of112 projections:229,376 FP32 optimizer
masters, initialized to zero. Forward and serialized tables are FP16. Codeword
c retains its prototype ID c>>8 and original sign/parity/coset mapping. Correct
the original FP32 expanded codeword by the signed prototype correction; reuse
the exact frozen inverse transformations and final FP16 weight rounding.
Arbitrary corrections make this an E8-derived learned codebook, rather than a
strict E8 lattice. There are no new per-weight indices or per-token state fields.

Actual extra files:112 x(32-byte header+4096-byte FP16 table)=462,336 bytes,
including458,752 bytes of table values. Count the export manifest separately.
This is a small positive storage increase, not a same-byte compression result.
No new entropy-container or compact-runtime residency claim is made.

## Data and fixed optimization

Use pinned WikiText-2 TRAIN and the native tokenizer. From2048-token grid
intervals exclude all token overlap with original32 calibration intervals,
all-small256 training intervals, and cross-moment pilot48 intervals. The871
remaining grid intervals are ordered; choose32 ranks floor(i*870/31),i=0..31.
Store int64[32,2048] and bind exact token/data/source hashes before training.
This is65,536 stored input positions,65,504 next-token targets/pass and262,016
target exposures across four passes. No validation/test inputs enter fitting.

Fixed recipe:

- Four successive torch.randperm(32) draws from one CPU generator seed20260927.
- Exactly128 successful updates. Final checkpoint only; no intermediate PPL
  measurement, checkpoint selection or adaptive hyperparameter search.
- Native FP16 source teacher; full256,000-token vocabulary, temperature1.
- Objective0.5 next-token CE+0.5 teacher-to-student KL;64-token logit chunks,
  native FP16 head GEMM then FP32 loss. Reuse the frozen norm-training loss code.
- AdamW, learning rate0.001, betas(0.9,0.999), epsilon1e-8, weight decay0;
  global master-gradient norm clip1. Learning rate is an experimental fixed
  choice for prototype-coordinate units, not a claimed optimal value.
- FP32 masters cast to FP16 inside every non-reentrant block-checkpoint replay.
  Ordinary decoder autograd, including the FP16 casts, without a custom backward.
- Dynamic loss scale starts1024, growth interval2000, factor2/backoff0.5.
  At most8 overflow attempts; retry the same scheduled window after reducing
  scale, with optimizer/masters unchanged. Any nonfinite forward aborts.
- No new clamps, residual tensor, codeword reassignment, small-tensor update,
  vocabulary change or stochastic weight recovery.

Keep journals and checkpoints every32 successful updates. This bounded driver
does not resume: existing work/output directories are rejected; any interrupted
run remains an incomplete attempt and must not be silently replayed.

## Correctness and feasibility smoke before training

First pass the CPU codec/mapping and bank/checkpoint tests. Then a separate GPU
smoke, using the declared TRAIN windows only, must establish:

1. All112 zero-table decoded weights reproduce the loaded baseline exactly.
   Initial128-token functional/native hidden states and logits are bitwise equal.
2. Perturb all112 tables with the same FP32 linspace(-0.002,0.002,2048) pattern;
   reshape to256x8 and use the declared FP16 forward casts. Check
   checkpoint/non-checkpoint gradients and outputs on native blocks0/18/55.
   Full56-block non-checkpoint training is not required because it retains all
   decoded-weight graphs. Restore every master afterward.
3. One complete2047-target CE/KL update with checkpointing produces finite,
   nonzero gradients across all112 tables, with every baseline tensor unchanged.
   Record seconds and peak GPU allocation. Reject a run that exceeds available
   memory or cannot complete its finite-value/export checks.
4. Serialize every rounded FP16 table using the strict4128-byte format. Reload
   and decode all112, install only the resulting FP16 weights temporarily, and
   require native output equality with the functional path. Verify actual
   table bytes, rounded master values and every restored baseline reference/hash.

Discard smoke optimizer/master updates, temporary tables and statistics from
the training initialization. Count smoke targets separately from128-update
training exposures. Bind the passed smoke receipt into the training run.

## Final export and independent quality check

Export exactly112 final FP16 prototype files plus an immutable manifest binding
the parent, small overlay, protocol, software, data, final checkpoint and training
receipt. Rehash actual frozen model tensors, prove master.half()==readback tables,
and verify the reloaded decoder/native forward independently. Account both the
new file bytes and logical combined model bytes; training checkpoints are extra
experiment storage and not inference payload.

Only after fixed training/export completes, independently score the source,
unchanged best all-small candidate and the reloaded prototype candidate on the
same entire validation split:264,764 targets,130 windows, up to2048 targets per
window, fresh state, frozen FP16-head/FP32-loss evaluation math, chunk64.

Meaningful repair criterion: candidate PPL at least1% below the same-process
all-small baseline. Also report the gap to source and the original source+5%
target separately. Failure stops this fixed recipe without extra epochs or
checkpoint selection. A gain does not imply recall recovery, an equal-quality
claim, compact GPU residency, minimum possible model size or permission to publish.
