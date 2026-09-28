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

## GPU training smoke: complete

The discarded smoke passes: native/functional hidden and last-eight-position
logit hashes agree, checkpoint gradient maximum difference is0, and all seven
small-parameter families have finite nonzero gradients. Full-vocabulary chunked
loss differs from the direct five-target objective by1.13249e-7, with identical
hidden gradients. One2047-target update succeeds without overflow; loss scale
remains1024. Actual serialized FP16 tensors equal rounded masters, and native
export hidden/logit hashes match. All underlying/teacher507 audits and fixed114
checks pass, and every trial master is reset to its original value.

Smoke terminal exit0;226.9593 seconds including loading/audits. Peak allocated
GPU memory is35,771,876,864 bytes for decoded student+teacher quality reference.
The smoke export is discarded and is not an eligible trained candidate.
See [smoke report](../reports/axis_small_readaptation_v1_smoke.json) and
[process receipt](../reports/axis_small_readaptation_v1_smoke_process.json).

## Fixed readaptation training: complete

The declared recipe adjusts393 existing tensors /3,580,928 parameters for448
successful updates /917,056 targets, one pass over the prepared TRAIN schedule.
AdamW lr3e-5;0.5 CE +0.5 original-FP16-teacher KL. All114 large tensors stay fixed.
Only the final448 checkpoint is eligible. Native evaluation reloads actual FP16
exports, checks all507 tensors and exact baseline repetition, and requires at
least1% paired development PPL improvement before complete validation.

Formal training started in a fresh process from current old-trained393 with
fresh optimizer, scaler and fixed RNG. It completed448 successful updates in448
attempts, zero overflows and917,056 targets. All393 exported tensors changed;
every exported FP16 value matches its final448 FP32 master after rounding.
The frozen underlying507, teacher507 and fixed114 content checks pass, as does
native/functional export parity. The final checkpoint is43,741,489 bytes.

Whole-process wall time is629.6042 seconds; recorded update times sum to381.2600
seconds (median0.808873 seconds). Peak allocated memory is36,041,105,920 bytes.
These timings include a single fixed recipe, not a performance benchmark.

The replacement `other_fp16.pt` is7,283,930 bytes, exactly the old file size.
Raw inference data remain3,138,928,792 bytes: zero replacement-byte delta.
The new provenance manifest is239,782 bytes, counted separately; no complete
new distribution or entropy-coded package has been built.

See [training report](../reports/axis_small_readaptation_v1_train.json) and
[process receipt](../reports/axis_small_readaptation_v1_train_process.json).
Training report SHA256:
`d5cc4e8d6a9c7d85c806014fc53ab7e90cd0e2af7fcc7b406218046df2acec75`.

## Independent native quality: pending

Independent actual-checkpoint/export CPU verification passes with CUDA
uninitialized: all393 rounded masters, the448-step schedule/attempt history,
code/data/receipt identities,116 inherited files and storage counts agree.
See [verification report](../reports/axis_small_readaptation_v1_verify.json) and
[process receipt](../reports/axis_small_readaptation_v1_verify_process.json).
The final candidate is awaiting native PPL scoring; training completion and
integrity checks are not evidence of a quality gain.

## Integrity preparation

Shared-loader/diagnostic CPU preflight and trainer CPU preflight pass with CUDA
uninitialized. The first diagnostic CPU attempt stopped because the newly
declared protocol file had not yet been copied remotely; its failure receipt
is preserved. Copying that exact file resolved the setup issue without a code
change. The independent evaluator CPU self-test and separate code review pass.

Both current and readapted raw model data are3,138,928,792 bytes. The replacement
file and new manifest are measured separately above. Execution decodes weights to FP16;
these tests do not establish compact GPU residency. Development data have been
used previously, and any later complete validation is also observed research
data, not an untouched test set.
