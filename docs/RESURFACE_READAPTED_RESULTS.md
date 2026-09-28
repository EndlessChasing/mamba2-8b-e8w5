# Readapted Mamba2-8B MK validation and Resurface repair

Status on2026-09-28: the source/current MK baseline, native correctness checks,
final1536 training/export and all DEV measurements are complete. The original
strict DEV run failed its historical cross-process MK replay prerequisite.
The separate terminal audit passed. Prospective full PPL and independent
confirmation are pending. No promotion or publication.

## Completed baseline

Same384 normal prompts and384 paired target-removed controls per arm, three
templates, N16/64, full256K greedy generation up to12 tokens. The score is the
first standalone six-digit integer matching the queried binding. This public
development set is distinct from the earlier2.7B word-target Resurface metric.

| Model | Normal MK | Removed-target false recall | Prior complete PPL |
|---|---:|---:|---:|
| Original source FP16 |167/384 =43.4896%|0/384|7.334175947|
| Current112-axis/W4 embedding/W5 head/readapted393 |111/384 =28.9063%|0/384|7.622396588|

The paired loss is14.5833 percentage points:99 correct in both,205 wrong in
both,12 gained and68 lost after compression/readaptation. Exact two-sided
McNemar p=1.201648635082027e-10. These synthetic measurements establish a
substantial current recall gap; the PPL improvement alone did not restore it.
PPL values above come from the already completed full130-window validation,
not a new PPL run inside this MK baseline.

| Records N | Template | Source correct/64 | Current correct/64 |
|---:|---:|---:|---:|
|16|0: colon records|26|18|
|16|1: sentence records|30|17|
|16|2: arrow dictionary|64|55|
|64|0: colon records|3|1|
|64|1: sentence records|7|3|
|64|2: arrow dictionary|37|17|

All507 native tensor contents and parameter identity/storage/version checks
passed before/after each arm. Twelve prompts per arm replayed exactly,
including generated token IDs. Outputs and nativeFP16 cache checks remained
finite. Each cache contained112 tensors/122,028,032 bytes (116.375MiB).
Execution used native SSD prefill followed by recurrent decode; prompt states
were not rounded after every input token. Weights were expanded toFP16 for
this reference execution, so this is not a compact-residency result.

Baseline report: `reports/readapted_mk_baseline_v1.json`,4,444,177 bytes,
SHA256 `a88f8fe75e82b992b8fc9cebeb6f681847e87aa36ea8bb1771edf979c47d551d`.
Process3517107 exited0 in993.0244 seconds; GPU process absence was verified.
The initial CPU Wilson endpoint test failure and corrected passing attempt2
are retained in separate reports; no model or scoring rule changed.

## Declared repair experiment

See [the frozen training and joint-quality protocol](RESURFACE_READAPTED_TRAINING_PROTOCOL.md).
The first candidate adds memoryless, soft gated cross-head readout at all56
layers, after native D*x and before grouped gated RMSNorm. It is explicitly a
post-D Resurface-inspired variant. All507 base parameters remain frozen.
224 adapter tensors contain1,154,104 parameters:2,308,208 FP16 payload bytes;
actual serialized file and metadata bytes will be reported after export.

CPU adapter checks passed7/7; staged loss checks passed4/4. The actual1536
TRAIN examples passed exact rendering, re-tokenization and full-answer prefix
checks:10,752 supervised answer tokens, seven per answer, maximum prompt1236
tokens. TRAIN values cover first digits6/7/8/9 in disjoint bands. The trainer
CPU preflight passed with CUDA uninitialized and exact protocol/model/data
bindings. These are correctness checks and do not establish recall/PPL quality.

- [x] Source/current384+384 MK baseline and integrity/replay audit.
- [x] Frozen bounded1536-step recipe, data and CPU preflight.
- [x] Native8B adapter zero/step/gradient/export correctness smoke. All zero
  prefill/three-step/cache comparisons are exact; checkpoint gradients match
  exactly across224 tensors. Full-vocabulary MK/prose objective differences
  are3.41e-8/6.55e-8. One complete MK+511-prose trial succeeds without overflow,
  then is discarded. Actual FP16 export/reload and hook-removal restoration
  are exact; both507-tensor models remain unchanged. Smoke exits0 in333.2057s,
  report SHA256 `3e34033ec70f2827b674d1f953afc318758868c51791529f49f33be8a8b04a89`.
- [x] Final1536 training and actualFP16 export verification:1541 attempts,
  five same-pair overflow retries,10,752 MK answer targets and784,896 prose
  targets. Runtime1943.0175 seconds, exit0. All224 exported tensors equal the
  saved final1536 masters rounded toFP16, with exact native export output.
  Base/teacher507 content audits pass. Independent actual-file CPU verification
  also passes (report SHA256 `545693a88a8b02eda34f80a1e3b41805340e2e6556cd0c355253d2259ecda8b9`).
  Adapter file2,539,647 bytes, SHA256
  `1e9857feaf80ede6525cce839726883f6230c98a067cb226eb2f095693ea803e`;
  its manifest is294,171 bytes. Base plus adapter data are3,141,468,439 bytes,
  excluding manifests/tokenizer/software; no complete distribution built.
  Training report SHA256
  `e64effbceda78b82488fd69608d6338fd2bb6e4b535c1d8eca12ddee789f70b6`;
  final checkpoint15,850,619 bytes, SHA256
  `03ed8da1d7c49fde4b70c053719eff19b715393d7b96dc41427f3cfc62af3fba`.
- [x] Same-enabled-adapter DEV MK and64-window PPL paired audit under the
  explicitly amended continuation protocol; original strict replay failure
  remains preserved. CPU audit12.6830 seconds, exit0, CUDA uninitialized,
  report SHA256 `6e89f9bcdaacabcbc9613713256874a7aa741b71ba289ed57fe745deb1a99074`.
- [ ] Conditional complete130-window PPL no greater than paired current.
- [ ] Conditional independent384+384 confirmation MK with positive paired
  gain/confidence and no increased removed-target false recall.

The candidate must pass both recall and PPL gates. Improvement over the current
model is not automatically full recovery to source recall or broad recall
equivalence. No tested candidate has yet passed those gates.

## Completed DEV measurements and preserved replay failure

| Same-process arm | Normal MK | Removed-target matches |64-window PPL |
|---|---:|---:|---:|
| Current readapted |112/384 (29.1667%)|0/384|7.6242504268035765|
| Final1536 active soft Resurface |346/384 (90.1042%)|0/384|7.59503251999968|
| Restored current |112/384 (29.1667%)|0/384|7.6242504268035765|

The observed paired gain is60.9375 percentage points; PPL decreases0.3832233%.
All768 restored MK rows and all64 restored prose rows/CE chunks match the
initial current arm exactly. Final507 base and224 adapter audits agree, with
169 bound files rechecked. This is DEV evidence, not fresh confirmation.

The historical current baseline was111/384. Seven generated sequences changed
between processes despite identical inputs and weight hashes; one changed
correctness. The cause remains unresolved. The original strict run correctly
retains `complete:false`, exit1 and its historical-MK mismatch error. Report
SHA256 `12f8ba2cdaf19ab929aa7c1eac14a20033f0f31223b5ada7311f6014a2e81781`,
6,930,157 bytes; runtime1698.7724 seconds; process absence verified. See the
[bounded drift audit](RESURFACE_MK_BASELINE_DRIFT.md).

The separately declared [soft continuation protocol](RESURFACE_SOFT_CONTINUATION_PROTOCOL.md)
explicitly removes only historical cross-process MK equality, after observing
that discrepancy. It preserves the original failure and all paired MK/PPL
quality thresholds. A CPU audit must first establish complete restored controls
and actual-file integrity, followed by prospective full130-window PPL and the
previously unopened CONFIRM set. Candidate, soft gates and numerical profile
remain identical. The prepared hard0 fallback is not triggered or evaluated.
