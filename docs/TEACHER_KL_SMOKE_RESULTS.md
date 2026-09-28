# Teacher-KL discarded GPU smoke: passed

The training-only smoke completed with exit code0 in**176.9986 seconds**.
All trial factors and optimizer state were discarded. This establishes
correctness and feasibility for the [fixed protocol](TEACHER_KL_COMPENSATION_PROTOCOL.md),
not a quality improvement. No formal448-update export or reserved/full-validation
result is established by this smoke. The accepted all-small distribution remains
unchanged; publication is held and MK deferred.

## Checks completed

- Fresh zero-B factors preserve all112 merged projection numerical values.
  The initial128-token hidden states and last-eight full-vocabulary logits
  match native accepted all-small bitwise; base tensors remain unchanged.
- On real blocks0/18/55, checkpoint/non-checkpoint outputs and factor gradients
  match exactly under nonzero perturbation; maximum gradient difference0.
  Initialization was restored before the complete training-window trial.
- One complete**2047-target pure-KL update** succeeds in**2.194529 seconds**,
  with zero overflow retries and loss scale1024 unchanged. Recorded loss equals
  teacher→student KL,**0.2574918809**; CE1.9106958834 is logging only, with
  coefficient0 in the gradient. These are one-window diagnostics, not PPL gates.
- All224 factor gradients are finite. All112 B gradients are nonzero and all112
  A gradients initially zero, as expected from B=0. The subsequent128-token
  backward has nonzero finite gradients in both families, with no second
  optimizer step.
- All112 strict factor files round-trip to the actual FP16 factors:
  **15,654,912 value bytes +3584 header bytes =15,658,496 bytes**. This excludes
  the required frozen base and metadata and is not a distribution size.
- After disk readback and native installation,128-token hidden states and
  last-eight full-vocabulary logits are bitwise equal to the functional path,
  with maximum absolute difference0. All507 original baseline tensor hashes
  are unchanged before/after; the trial state is discarded.

## Measured memory and scope

| Scope | Peak allocated GPU bytes | Peak reserved GPU bytes |
| --- | ---: | ---: |
| Complete training-window update, including resident teacher/base |36,211,180,032 |36,834,377,728 |
| Entire smoke, including temporary native export verification |46,416,051,200 |48,146,415,616 |

The176.9986-second total includes model loading, initialization, replay,
serialization and export checks. One update's timing is not a guaranteed
448-update completion time or isolated throughput benchmark. Dense FP16 model
weights are used; compact file size does not imply compressed GPU residency.
Reserved windows were not used for fitting or smoke forwards.

## Evidence and next status

- [Smoke receipt](../reports/teacher_kl_compensation_v1_smoke.json):641,149 bytes,
  SHA256 `69f07ca855b383424f713de4095a0fac25785e3fec02c75193fdc2b84ce19c00`.
- [Actual process/exit receipt](../reports/teacher_kl_compensation_v1_smoke_process.json):2634 bytes,
  SHA256 `82216c19414381ac87bfb31b90b8a8af0dd545f2b77419a6e09ff82ad06b3324`.
- [Readiness and frozen source identities](TEACHER_KL_READINESS.md).

Formal training must start from fresh factors and optimizer state, complete
exactly448 successful updates and export only the final checkpoint. Its64-window
reserved gate requires both>=1% PPL improvement and lower teacher KL; a failed
gate stops the experiment before full validation. Passing this smoke does not
relax that rule or authorize extra epochs, checkpoint selection or publication.
