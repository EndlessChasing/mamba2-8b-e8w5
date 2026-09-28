# Readapted Mamba2-8B MK validation and Resurface repair

Status on2026-09-28: complete. The fixed final1536 soft adapter passes the
declared paired MK/PPL quality gates, including independent confirmation and
independent receipt audits. Original strict DEV failure from historical
cross-process MK replay remains preserved; the transparent continuation
changes only that prerequisite. No model publication or new distribution.

| Final measurement | Current compressed base | Same base + soft adapter |
|---|---:|---:|
| Independent normal MK |91/384 (23.6979%)|340/384 (88.5417%)|
| Independent target-removed false recall |0/384|0/384|
| Complete WikiText-2 validation PPL |7.622396588|7.593163114|

The independent MK improvement is64.84375 percentage points, with a
conservative95% lower bound of58.502343 points. Full PPL improves0.3835208%,
with all130 windows improving. This meets PPL non-regression, not exact
unchanged candidate PPL. Both tasks use the identical enabled adapter/policy.

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
- [x] Complete130-window PPL no greater than paired current:7.622396588→
  7.593163114,130/130 windows improve,264,764 targets/4,137 CE chunks per arm.
  Current/restored/historical chunks match exactly;507/224 and167 bound-file
  checks pass. Independent receipt audit passes. GPU exit0 in287.4773 seconds;
  report SHA256 `06a71c11fc0a12a52add6e7bf5d28b8a8eb9f832b2ec1cb846d9acaf30afe961`.
- [x] Independent384+384 confirmation MK with positive paired gain/confidence
  and no increased removed-target false recall. Normal91→340,251 gained/2
  lost, conservative95% lower bound+58.502343pp, exact McNemar
  p4.439957888196715e-72. All768 restored rows and12 within-arm replays match;
  final507/224 and170 bound-file checks pass. GPU exit0 in1532.3547 seconds,
  evaluator process absent; independent terminal receipt audit passes.

This candidate passes both gates under the explicit continuation protocol.
The finite numeric-binding test does not establish broad recall equivalence
or performance on unseen prompt families. Original source full PPL remains
7.334175947, so the repaired compressed candidate is still3.5312374% higher.

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

## Complete PPL and fresh confirmation preparation

Full same-process PPL is7.622396587826496 (current),7.593163113563457 (soft
adapter), and7.622396587826496 (restored). The candidate improves all130
windows by0.3835207723% in aggregate. Total NLL falls1017.3771133423 over
264,764 targets. There are129 full2048-target windows and one572-target
tail; the last CE chunk has60 targets. Every baseline/restored/historical
CE chunk, window NLL and aggregate matches exactly.

The terminal report has2,703,975 bytes, exit0, with evaluator process absence
verified. Independent receipt-only audit:
`reports/resurface_soft_continuation_v1_full_independent_audit.json`, SHA256
`14a9bb85a95dcdd6429e12409c73336765e546a9ae09b43b2c0bd28bdcc2d515`.
The corpus has informed previous development; this is complete validation
measurement, not untouched natural-language generalization evidence.

Only after full PPL passed, the previously absent CONFIRM directory was
created using the original frozen split contract. Preparation exit0 in
2.8924 seconds, CUDA uninitialized,768 unique prompts with exact raw/token
roundtrip and normal384/removed384 coverage. Actual manifest SHA256
`0abb3e1ef2994e2088b8da0f0f605371e980f5afa3ea9df5335ff19d3f5b3dec`.
The new ranges separate numeric instances, while the three template families
remain shared. No confirmation fitting or prior score inspection occurred.

## Completed independent confirmation

| Records N | Template | Current correct/64 | Active soft correct/64 |
|---:|---:|---:|---:|
|16|0: colon records|16|61|
|16|1: sentence records|14|63|
|16|2: arrow dictionary|44|64|
|64|0: colon records|0|47|
|64|1: sentence records|2|48|
|64|2: arrow dictionary|15|57|

N16 improves74→188/192 (38.5417%→97.9167%); N64 improves17→152/192
(8.8542%→79.1667%). All six cells improve. Across384 paired normal prompts,
89 are correct in both models,42 wrong in both,251 newly correct and2 newly
wrong. Target-removed matches remain0/384. The exact two-sided McNemar
p-value is4.439957888196715e-72. The conservative95% improvement lower bound
is0.5850234300333482, using the predeclared Bonferroni/Clopper–Pearson rule.

The report is5,171,089 bytes, SHA256
`306ae9e8ed5e78756f7c8ea39c8db40dbebf3895ced3c5c279722be5100a344b`.
Runtime1532.3547 seconds, exit0; process3797841 absence verified. All three
arms have complete768-prompt coverage and12 exact fresh-cache replays. The
restored current rows equal the initial current rows in full. Actual507 base
and224 adapter contents/identities, same native environment and170 final
bound-file checks pass.

Independent CPU audit:
`reports/resurface_soft_continuation_v1_confirm_independent_audit.json`,
4,523 bytes, SHA256
`37e3c9e0e5498335fcbf0a30fbc0cfaee34418c80cf5ea0a1cfd99adad6b56da`.
It recomputes the paired statistics and confidence bound independently and
checks complete receipt identities/restoration/chronology. Its direct-binomial
inversion agrees within1e-15; SciPy was unavailable and was not claimed used.
This audit reads reports/manifests, not model binaries; actual binary/tensor
audits were performed by the completed native evaluator. Preparation started
34.4859 seconds after complete PPL finished successfully.

## Validated artifact and scope

The measured candidate is the unchanged current readapted axis/W4-embedding/
W5-head base plus the actual final1536 soft adapter at remote path
`/home/horde/Mamba2-8B-E8W5/artifacts/resurface_readapted_v1/adapter_fp16.pt`.
Its2,539,647 serialized bytes add0.0809081% to base raw data; FP16 tensor payload
is2,308,208 bytes and the separate manifest is294,171 bytes. Combined raw base
and adapter data are3,141,468,439 bytes, excluding manifests/tokenizer/software.
No new complete entropy-coded distribution was built or published.

All507 base tensors stay frozen; only1,154,104 external readout/router
parameters were trained. Native FP16 recurrent/conv cache capacity stays
122,028,032 bytes per tested batch-one prompt. Execution expands weights to
FP16: these results do not measure compressed GPU residency, latency or ASIC
energy. This is the documented post-D Resurface-inspired native variant,
not an exact reproduction of the original paper's placement. Soft gates stay
enabled for both MK and PPL. The hard0 fallback and deterministic-profile
probe remain untriggered and unexecuted on the model.

The scope is independent numeric bindings within three shared prompt families,
N16/64, and the complete previously used WikiText-2 validation protocol. No
arbitrary-prompt, longer-context, original test-set or global-smallest claim
follows. The original historical MK replay failure remains unchanged in Git.
