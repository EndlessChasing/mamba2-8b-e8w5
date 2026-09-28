# Norm compensation implementation

This experiment is governed by `NORM_COMPENSATION_PROTOCOL.md`. It uses the
original two-sweep parent. Implementation and smoke correctness do not imply a
quality improvement. No validation or recall data enters this trainer.

`mamba_e8w5/norm_training.py` holds 113 FP32 master tensors separately from the
507 frozen FP16 runtime tensors. Every native block call, including every
non-reentrant checkpoint replay, constructs an ordinary `functional_call`
mapping with freshly cast FP16 master values. The final residual sum and final
norm follow the unfused installed Mamba backbone. No runtime or quantizer source
is changed.

The frozen FP16 vocabulary heads process at most 64 positions at a time. The
full-vocabulary CE and teacher-to-student KL use FP32 logits and global target
normalization. Each chunk's loss is scaled once. Detached hidden gradients are
collected and supplied to the backbone in one backward call. Master gradients
are unscaled before clipping. Both staged hidden gradients and master gradients
are checked for overflow; an overflow calls no optimizer step, halves the scaler
through its public API and retries the same schedule entry. Missing gradients,
nonfinite forward values or all-zero master gradients fail immediately.

The CPU tests use the same orchestration with a small two-block model. They
check changed-master checkpoint recomputation, frozen parameter gradient state,
FP32 master gradients, exact changed FP16 export/readback forward identity, and
chunk-tail normalization with a non-unit loss scale. The GPU smoke additionally
uses the actual architecture, full vocabulary, 128-token checkpoint comparison
and one full 2047-target trial update. Smoke state is discarded.

The fixed smoke numerical tolerances are in `SMOKE_TOLERANCES`: native/export
forward outputs must match bitwise; checkpoint gradient comparison uses rtol
0.005 and atol0.00003; different head chunk shapes use rtol0.005 and atol0.001,
with objective absolute difference at most0.00001. All measured differences are
recorded, and failed receipts are preserved before implementation repairs.

Training writes a distinct durable checkpoint after every attempted update.
It contains only FP32 norms, optimizer/scaler/RNG state and provenance, never
copies of the 8B model. A resumed process checks the prior PID is dead and binds
the exact schedule, inputs, code, successful smoke, and initial frozen GPU
weight hashes. Existing receipts are archived before resume. All394 frozen GPU
tensor values are hashed in bounded chunks before and after the run. Only the
final128-update checkpoint is exported, and the overlay verifier compares its
rounded masters directly to the exported FP16 norms. Container bytes are measured;
fixed tensor capacity does not imply identical entropy-coded distribution size.

Run smoke and training as separate coordinated GPU jobs. Example paths:

```sh
python scripts/train_norm_compensation.py --mode smoke \
  --work-dir artifacts/norm_compensation_v1_smoke_work \
  --report reports/norm_compensation_v1_smoke.json
python scripts/train_norm_compensation.py --mode train \
  --work-dir artifacts/norm_compensation_v1_work \
  --report reports/norm_compensation_v1_train.json \
  --smoke-report reports/norm_compensation_v1_smoke.json
```

Publication, pushes and uploads remain on hold.
