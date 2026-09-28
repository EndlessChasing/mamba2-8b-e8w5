# Current-axis small-tensor diagnosis and readaptation

Declared before implementation and measurements on 2026-09-28, following the
user's selection of option1. PPL has priority; recall/Resurface and publication
remain deferred. Frozen recipes and their previous negative outcomes remain
unchanged.

## Fixed baseline and hypothesis

Current baseline is all112 nominal2.5-bit axis projections, W4 embedding,
W5 head and393 trained FP16 small tensors. Its complete-validation PPL is
7.977512009035770 versus source7.334175947318572. The source+5% reference is
7.700884744684501. The small tensors were trained with the earlier E8/W5
projections, not readapted after the axis changes. A mismatch is a hypothesis,
not an established cause: the earlier source-projection intervention did not
show substantial incompatibility with those trained small tensors.

Authoritative identities:

- Current evaluation: `23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4`.
- Input axis manifest: `0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85`.
- Output axis manifest: `2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f`.
- Original parent manifest: `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
- Trained-small manifest: `edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006`.
- Trained-small file: `15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4`.
- Original source: `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
- Prepared TRAIN data: `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.

Only393 existing tensors /3,580,928 parameters are eligible: norms, conv
weights/biases, A_log, dt_bias and D, exactly the frozen SmallMasters inventory.
All114 large tensors (112 projections and two vocabulary matrices), model
geometry and evaluation math are fixed. No new clamp, state size, adapter or
inference tensor is introduced. Preserve original frozen files; use explicit
new wrappers around reusable loading, training and native-scoring functions.
No monkeypatch or global replacement of old hyperparameters.

## Stage 1: descriptive small-tensor diagnostic

Use the existing64 reserved TRAIN windows from teacher_kl_compensation_v1 in
their stored order:131,008 targets. These windows have already informed earlier
experiments; they are not fresh test evidence. They never enter fitting.

In one process score current axis + old trained393, then the same large
weights + original source393, then restore and score old trained393 again.
Obtain original393 through the verified source-to-public-runtime mapping,
cast to FP16 and check every shape/key. Verify all507 actual tensor hashes
before/after every arm; all114 large tensors must remain fixed. Restore current
small parameters in finally and require exact baseline CE/window repetition.

Native FP16 backbone/head, fresh state for each window, FP32 full-vocabulary
CE chunks64 plus63 final targets, FP64 host sums. Report PPL/NLL differences
and window signs. No validation data, training, initialization selection,
new package or promotion in this diagnostic. Historical drift is descriptive;
only same-process pairing defines the comparison. A correctness failure must
be resolved before training. Neither direction of the diagnostic alone proves
that training cannot help or selects a different initialization.

## Stage 2: fixed short readaptation

After a valid diagnostic and a passed training smoke, always initialize from
the current old-trained393 FP16 values. The original-small arm is diagnostic
only. Optimize the same393 FP32 masters with native FP16 casts; keep the whole
underlying baseline model frozen. Reuse frozen SmallMasters/native-block
checkpointing and the established chunked full-vocabulary loss implementation.

Fixed recipe:

- Existing448 TRAIN windows from the pinned data manifest, visited once in its
  seed20260928 torch.randperm schedule;448 successful updates/917,056 targets.
- These TRAIN tokens were used by an earlier rank4 experiment. Reuse is declared;
  no claim of previously unseen training data. All64 reserved windows remain
  disjoint from fitting, including smoke and failed attempts.
- Objective:0.5 next-token CE +0.5 original-FP16-teacher-to-student KL, T=1.
- Fresh AdamW:lr3e-5, betas(.9,.999), eps1e-8, weight_decay0; global grad clip1.
- Batch1,2048 stored tokens/2047 targets, native eval mode with gradients,
  checkpoint each block, full256K-vocabulary loss chunks64 plus63.
- GradScaler:init1024, growth2, backoff.5, growth_interval2000. A finite-forward
  gradient overflow retries the same window with halved loss scale and no
  master/optimizer update. At most8 overflow retries;456 attempts maximum.
  Nonfinite forward/master/FP16 export or changed frozen weights aborts.
- Final448 successful updates only; no validation during fitting, checkpoint
  selection, extra epochs, learning-rate search or inherited optimizer state.

The smoke uses only training windows, records and discards every trial update,
then formal training starts from exact original masters/fresh optimizer/scaler
and fixed RNG. Check native-vs-functional FP16 hidden/logit parity, checkpoint
forward/gradient parity across all seven parameter families, chunked objective/
hidden-gradient parity, a full2047-target update, finite native A/dt behavior,
and independent FP16 export/native parity. Existing documented numeric smoke
tolerances may be reused unchanged and must be recorded.

Use fresh work/report/artifact paths and durable per-attempt journals.
Write final FP32 masters and independent FP16 other_fp16.pt plus manifest.
Check serialized tensors equal final master.half(), exact393 coverage, finite
values and unchanged114 large/underlying507 baseline content. Record actual
changed small tensor identities; unchanged individual tensors are permitted.
Native export must match functional hidden states and logits on a recorded
training-only probe. Preserve failed work; no implicit resume or overwrite.

## Stage 3: independent native quality

A separate evaluator verifies actual files/final448 checkpoint, complete
attempt/schedule/data/code bindings, final rounded masters, and all393 exported
identities. It never uses the training forward to compute PPL. Score current
baseline, readapted393 candidate and restored-current repeat on the same64
reserved TRAIN windows, using the Stage1 math. Require all507 before/after
hashes,114 unchanged large tensors, exact baseline repeat and final restoration.

Only candidate PPL <=99% of the paired baseline allows complete validation.
If this gate fails, stop the fixed recipe with no extra steps/selection or full
validation. If it passes, freshly load actual candidate and score source,
current baseline and candidate on the established130 windows/264,764 targets,
including572-target final window and60-target final CE chunk. Pin the prior
window/token plan. Report >=1% paired improvement and source+5% as separate
conditions. Development and validation are already observed, not untouched tests.

## Capacity and integrity scope

Existing raw model data are3,138,928,792 bytes. Replacing393 values preserves
parameter count and FP16 payload count; measure actual serialized byte delta
and new manifest separately. A different Huffman size or compact runtime
residency is not inferred. No full package or publication is authorized here.

Use one deduplicated actual-model input ledger:current112 axis files, W4/W5
vocabulary, small/config/codebook, pinned manifests/source/report, prepared
training data/provenance and relevant frozen/new code. Do not recursively
rehash unrelated historical weights/Hessians. Verify real source/current file
identities before work and bound inputs finally; label historical provenance
as such. CPU preflights precede GPU work. Record process identity, terminal
exit and complete reports. CPU checks are integrity evidence, not PPL results.
