# Current readapted 8B MK baseline and Resurface work plan

Declared on 2026-09-28 before new MK measurements. The user requests MK
validation, Resurface repair, and preservation of the latest PPL. Publication
remains on hold. This document freezes Stage 1 only; a separate adapter recipe
will be declared before training or candidate selection.

## Current model and PPL requirement

The pure 56-layer Mamba-2 8B uses 112 nominal 2.5-bit axis projections, W4
embedding, W5 output head, and the final448 readapted393 small tensors. Its
complete-validation PPL is 7.622396587826496 on 130 windows /264,764 targets.
Raw inference data are 3,138,928,792 bytes. Existing model and evaluator files
remain immutable.

- Final native PPL report: `3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4`.
- Readapted manifest: `3ede80b2883d8b8987cb3e2198fd8badc2dbbb5af03c8fe6cac7f5f9f5b62047`.
- Final checkpoint: `c464dbb99c80d99851f7543ed319bab0fab4f00e6aafebf25b5ba41bbfc04af6`.
- Actual export CPU proof: `2a48af2d28d1f2859ea717feffae4ab6e0adae0ee0b0b4a81e7865c7a24c6ad6`.
- Original source: `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.

The working interpretation of preserved PPL is no regression: the same enabled
adapter and inference policy must have complete PPL no greater than the paired
current model, with no relaxed source+5% allowance. Exact equality under a
separate user-selected recall mode is a different claim and would require
reporting that recall mode's PPL as well. No unconditional quality guarantee
is made before measurement.

## Stage 1: fixed baseline measurement

Reuse the immutable `mamba_e8w5.evaluation.synthetic_mk_cases` public-v1 renderer
with `split="validation"`, `samples_per_cell=64`, and its default N=(16,64).
There are three templates and 64 samples per cell: 384 normal prompts plus
their 384 target-removed controls. Keys/values, seed, query positions, rendering
and first-six-digit matching retain the frozen generator's semantics. Prior
12-case development measurements overlap this generator family; these data
are observed development evidence, not a fresh confirmation test.

Score original FP16 source, then the actual current readapted model, on exactly
the same 768 prompts/token hashes in generator order. Native SentencePiece has
no automatic BOS/EOS. Greedy generation uses the complete 256K vocabulary,
at most12 new tokens, stops at EOS, and scores the first standalone six-digit
integer against the target. This is one queried binding per prompt; it is not
interchangeable with the old 2.7B Resurface word-target MK metric.

Execution is the established native SSD prefill followed by recurrent decode.
Each prompt starts from fresh FP16 convolution/SSM caches. Prefill writes FP16
cache at its end; decode writes it every generated token. Do not describe this
as tokenwise FP16 rounding of every prompt token or as compact weight residency.
Reject empty inputs, nonfinite computation, and prompt+12 above4096 tokens.

Before GPU scoring, CPU verification must establish actual current model/export
provenance, exact507 identities, prompt/control bindings, deterministic prompt
generation, unique IDs/token digests and tokenizer/length bounds. Reuse the
final448 export verification; do not accidentally score the older393 values
loaded initially by the shared axis loader.

Audit all507 native tensor contents before/after each arm, and parameter
identity/storage/version immutability within each arm. After scoring, replay
sample0 for every N/template cell and both conditions (12 prompts per arm),
requiring exact generated IDs, decoded text and score agreement with that arm's
recorded rows. No parameters are fitted or selected by this stage.

Persist progress at least every16 prompts, retain every outcome and exact input
identity, and record process identity, terminal exit, finite state/cache details
and final bound-input rechecks. Report aggregate/per-cell normal/control counts,
paired gain/loss and uncertainty separately. A larger sample is still a
synthetic engineering measurement; it does not prove broad recall equivalence.

## Subsequent repair work

After the baseline, implement a no-EMA Resurface readout compatible with the
actual grouped Mamba-2 runtime. Freeze the compressed backbone and count every
adapter parameter/file. Verify zero-adapter parity, prefill and recurrent-step
injection, checkpoint gradients, and multi-token numeric-answer loss before
training. The old 2.7B adapter checkpoint is not geometrically compatible and
its results are not an 8B measurement.

Declare disjoint TRAIN/development/confirmation inputs, a bounded training
recipe and candidate-selection rule before fitting. Confirmation MK must use
separate seeds and key/value ranges, with exact input/token overlap checks;
never fit on the baseline or final confirmation prompts/answers. Evaluate the
same active adapter on MK and complete PPL. Include deleted-binding controls,
all data/storage costs and failure outcomes. No model publication is authorized.

## Checklist

- [ ] Independently verify actual current inputs and the fixed384+384 baseline.
- [ ] Complete paired source/current MK with finite outputs,507 checks and12
  exact prompt replays per arm.
- [ ] Declare and validate the native Resurface adapter and training recipe.
- [ ] Train the bounded candidate with frozen backbone and prose preservation.
- [ ] Independently verify exported adapter, MK improvement/control behavior,
  and complete PPL no worse than the same-process current model.

Checklist completion will be tracked in `PLAN.md`; this protocol file remains
unchanged after its identity is pinned.
