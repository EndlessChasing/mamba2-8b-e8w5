# Current-axis small-tensor readaptation

Experiment declared on 2026-09-28 in the [fixed protocol](AXIS_SMALL_READAPTATION_PROTOCOL.md).
The user selected readaptation of existing small parameters. PPL has priority;
recall/Resurface and publication remain deferred.

## Descriptive diagnostic: complete

All112 current nominal2.5-bit axis projections, W4 embedding and W5 head remain
fixed. Only the393 FP16 small tensors are exchanged. Each arm uses the same64
previously observed reserved TRAIN windows /131,008 targets, native FP16
backbone/head and FP32 full-vocabulary CE with FP64 host accumulation.

| Small parameters | Development PPL | Relative to current |
|---|---:|---:|
| Current, previously trained393 | 8.013059657 | reference |
| Original source393 | 8.190893923 | +2.21931% |
| Restored current393 | 8.013059657 | exact repeat |

Original source393 improves16 windows and worsens48; mean NLL rises by
0.0219503722 nats per target. All507 actual tensor identities are checked before
and after each arm;114 large tensor identities, storage/version and contents
remain fixed. All64 baseline windows /2,048 CE chunks repeat exactly, and final
current-model restoration and bound-file rechecks pass.

The prior adaptation remains useful on the current axis weights. Reverting it
does not repair PPL. This diagnostic does not establish that readaptation will
help, and it does not select training initialization. The fixed training recipe
still starts from current old-trained393 values.

Terminal diagnostic: exit0,360.4870 seconds; report complete and correctness
passed. Report SHA256:
`7a4d9ad7793773e2b09859eafd201189344d529c0c6d7bf4995af6370dfcdc09`.
See [report](../reports/axis_small_diagnostic_v1.json) and
[process receipt](../reports/axis_small_diagnostic_v1_process.json).

## Training and independent quality: pending

The declared recipe adjusts393 existing tensors /3,580,928 parameters for448
successful updates /917,056 targets, one pass over the prepared TRAIN schedule.
AdamW lr3e-5;0.5 CE +0.5 original-FP16-teacher KL. All114 large tensors stay fixed.
Only the final448 checkpoint is eligible. Native evaluation reloads actual FP16
exports, checks all507 tensors and exact baseline repetition, and requires at
least1% paired development PPL improvement before complete validation.

No training or candidate-quality result is claimed by this pending section.

## Integrity preparation

Shared-loader/diagnostic CPU preflight and trainer CPU preflight pass with CUDA
uninitialized. The first diagnostic CPU attempt stopped because the newly
declared protocol file had not yet been copied remotely; its failure receipt
is preserved. Copying that exact file resolved the setup issue without a code
change. The independent evaluator CPU self-test and separate code review pass.

The current raw model data are3,138,928,792 bytes. A new replacement file and its
manifest must be measured after training; unchanged parameter count alone does
not establish unchanged serialized bytes. Execution decodes weights to FP16;
these tests do not establish compact GPU residency. Development data have been
used previously, and any later complete validation is also observed research
data, not an untouched test set.
