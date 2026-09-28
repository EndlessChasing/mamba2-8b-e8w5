# Conditional low-rank residual repair — draft

**Conditional preparation, pending the current prototype-training result.**
No low-rank model experiment has been selected or run. An isolated
[CPU factor format and merge helper](LOW_RANK_RESIDUAL_IMPLEMENTATION.md) is
implemented and checked; a training recipe, GPU path and model loader are not
implemented. No expected PPL gain is assumed. Publication remains on hold.

## Representation and exact value capacity

For a frozen decoded projection `W0[m,n]`, store FP16 factors `B[m,r]` and
`A[r,n]` representing an additive rank-at-most-r correction `BA`. The factors
add `r(m+n)` values, or exactly `2r(m+n)` bytes. No bias or separately stored
scaling parameter is assumed; any fixed scaling is incorporated into factors.
Low rank describes the factor product before the final FP16 merge; rounding
the merged weight can make its effective difference from `W0` higher rank.

The actual resolved manifest contains 56 `in_proj[18560,4096]` and 56
`out_proj[4096,8192]` matrices. Therefore:

- All 112 projections: `56r(18560+4096+4096+8192)` = `1,956,864r` values.
- Only 56 output projections: `56r(4096+8192)` = `688,128r` values.

Percentages below use the current complete offline bundle **2,671,176,471 B**,
including tokenizer, software, licenses and metadata. These are additional
uncompressed factor-value bytes; the frozen base remains required.

| Scope | Rank | New FP16 values | Extra value bytes | MiB | Increase over current bundle |
|---|---:|---:|---:|---:|---:|
| All 112 | 4 | 7,827,456 | 15,654,912 | 14.929688 | 0.586068% |
| All 112 | 8 | 15,654,912 | 31,309,824 | 29.859375 | 1.172136% |
| All 112 | 16 | 31,309,824 | 62,619,648 | 59.718750 | 2.344272% |
| Output 56 | 4 | 2,752,512 | 5,505,024 | 5.250000 | 0.206090% |
| Output 56 | 8 | 5,505,024 | 11,010,048 | 10.500000 | 0.412180% |
| Output 56 | 16 | 11,010,048 | 22,020,096 | 21.000000 | 0.824359% |

The isolated CPU format stores both factors and a 32-byte header in one file
per projection. If used by a future model experiment, these headers would add
another 3,584 B for all 112 or 1,792 B for output 56. No model factor set has
been trained or packaged. A manifest, loader source and any revised container
metadata are additional and must be measured after packaging.
No entropy-compression benefit is credited. For example, rank-8 output factors
plus these headers add 11,011,840 B; the value-only percentages exclude headers.
If a future baseline includes prototype files, count those as well and update
the denominator. This is added capacity, not a same-byte result.

Shape evidence: [resolved manifest](../reports/all_small_resolved_v1_manifest.json),
SHA256 `bc554db936b13cbbeaeef5267d96b8d6ba7c183bf740c7e64e5f26ac7c478af8`.
Bundle evidence: [distribution manifest](../reports/all_small_distribution_v2_manifest.json),
SHA256 `5f5445803f4f0b12ae88c74fec0b00d331a536e12cc424ddfe2093e538d9b5e6`.

## Why local error fitting is insufficient

The completed [cross-moment pilot](../reports/projection_crossmoment_pilot_v1.json)
reduced median native projection-output MSE by **17.433952%**, improving five
of six projections. Installing the six repaired projections nevertheless
worsened held-out TRAIN PPL from **8.423374233 to 9.147628791**; the input-moment
refresh control scored 8.424745839. Layers interact through nonlinear gates,
normalization and recurrent dynamics, and their inputs change after upstream
weights change. Lower local squared error therefore cannot establish lower
language loss. This does not prove low-rank repair fails; it rules out treating
a low-rank weight/output fit as a PPL guarantee.

## Scope evidence from completed diagnostics

The earlier draft considered output-only rank 8 as a small-capacity example.
Existing quality evidence supports considering **both input and output
projections** in a subsequent single pilot. It does not establish the best rank.

On the original E8/W5 candidate, the same four validation windows / 4,096
targets scored PPL **9.237163951**. Each separate intervention restored the
named weights fully to source FP16 while keeping W5 and original small tensors:

| Source weights restored | PPL | Mean NLL reduction from original E8/W5 |
|---|---:|---:|
| All input projections | 8.294958073 | 0.107587038 |
| All output projections | 8.671019498 | 0.063248534 |
| Input z rows | 8.746508676 | 0.054580295 |
| Input x rows | 8.775621406 | 0.051257324 |

Evidence: `row_rescues.*.ppl` in
[the original role diagnostic](../reports/ppl_components_dev.json), SHA256
`1b01b3c844358998a4c41c603d54f9bc87bf8e451672b342ccf142ae7c073c36`.
These effects are conditional and cannot be added. Dense source restoration
is neither a low-rank candidate nor an upper bound on a learned correction.

The later [all-small interaction diagnostic](../reports/small_projection_interaction_v1.json)
restored all 112 projections together: PPL **8.359912174 to 7.412136849** on
264,764 targets, with W5 and trained small tensors fixed. That supports further
projection repair, but does not re-establish the in/out ordering on this newer
base. The running prototype candidate has no role-specific quality evidence.

All-projection rank 4 adds **15,654,912 value bytes**, versus **11,010,048** for
output-only rank 8: a difference of **4,644,864 B / 4.4296875 MiB**, plus
1,792 extra header bytes and unmeasured manifest differences. Broader coverage
halves each output correction's rank, so storage arithmetic cannot rank their
PPL. A shared x/z-only rank-4 input basis plus rank-4 outputs would save only
974,848 value bytes versus full in/out rank 4; it also requires explicit row
mapping and model-loader support. No row-masked model format is implemented.

## Conditional bounded pilot for later review

If low-rank repair is pursued after the prototype result, prioritize reviewing
a single scope that covers input and output projections. The preceding table
compares capacities; no rank, training recipe or sweep has been selected.

If this route is selected after the current result:

1. Bind an immutable, explicitly chosen base and its same-process comparator.
   Freeze all base E8/W5 bytes and existing small tensors; freeze prototype
   tables too if they are part of that base. Use the original FP16 source
   teacher. Initialize one factor to zero and the other to a deterministic
   nonzero random matrix; initializing both to zero gives zero gradients for
   both factors. Record initialization scale and RNG before training.
2. Use FP32 optimizer masters rounded to FP16 inside every checkpoint replay.
   A proposed native deployment contract is
   `W_eff = half(W0.float() + B16.float() @ A16.float())`, with TF32 disabled.
   Training and independent export evaluation must execute this same contract.
   Decode the frozen E8 base once; gradients need not traverse its rotations.
   A separate two-GEMM residual branch can round differently and must not be
   substituted without a new parity/quality check. Materialized weights retain
   the current FP16 residency; small factor files do not imply compact GPU RAM.
3. Predeclare a single fixed budget, for example 32 new disjoint TRAIN windows
   of 2048 tokens, four passes, **128 successful updates / 262,016 targets**.
   Bind data exclusions and order before fitting. Reuse full-vocabulary
   `0.5 CE + 0.5 KL(teacher || student)`, temperature 1, 64-token loss chunks,
   FP32 masters, checkpointing and bounded loss-scale retries. Fix learning
   rate and factor initialization together in a future protocol. Those values
   are not chosen yet. The scope discussion above explicitly uses already
   observed validation diagnostics; it is not an untouched evaluation.
4. First run a discarded correctness/feasibility smoke: zero-residual native
   parity, checkpoint gradient parity, finite full-window loss/gradients,
   a real optimizer update, exact factor readback and native export parity,
   frozen-base content hashes, wall time and peak memory. At initialization,
   a zero factor can legitimately make the other factor's gradient zero;
   require both factor families to receive gradients after the first update.
5. Evaluate only the final export, independently loaded, with source and the
   unchanged chosen base on the frozen full-validation plan: 264,764 targets,
   130 windows, FP16 native projections/head and FP32 loss, identical token
   hashes and state resets. A proposed meaningful-repair gate is
   `PPL_candidate <= 0.99 * PPL_base`. Report NLL deltas, per-window results,
   actual complete bytes and source gap. The source-plus-5% target is separate.
   Preserve failure without extra epochs, intermediate checkpoint selection
   or an automatic rank sweep. MK stays deferred/nonblocking; no recall
   recovery, test result or publication claim follows from this gate.

FP32 masters, gradients and two Adam moments occupy **88,080,768 B** for
rank-8 outputs, or **125,239,296 B** for rank-4 input and output projections,
before FP16 casts and activations. Full model/teacher memory, temporary dense
residuals and checkpoint activations still need the smoke on the 96 GB GPU.
No speed, memory-fit or accuracy result has been measured for either design.

For scale only, the last completed full-validation best PPL 8.359867548 would
need about **7.882694%** reduction to reach source-plus-5% (7.700884745).
That is a planning gap, not a predicted low-rank gain; any future gate must
recompute source and baseline together in the same process.
