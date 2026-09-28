# Fixed teacher-KL rank-4 compensation protocol

Declared before this experiment's smoke, training or candidate measurements.
This is one bounded experiment following the completed negative rank-4 result
and its [posthoc diagnostic](LOW_RANK_GENERALIZATION_RESULTS.md). It resets to
the accepted all-small base. It does not continue either failed optimizer,
select an intermediate checkpoint, change rank or authorize publication. MK
remains deferred. Several recipe components change together; any outcome will
not isolate a single cause or guarantee a repair.

## Fixed base, initialization and arithmetic

Use original two-sweep E8/W5 plus the final accepted393-small-tensor export:

| Input | SHA256 |
| --- | --- |
| Original raw manifest | `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed` |
| All-small overlay manifest | `edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006` |
| All-small FP16 values | `15de8deba5335a1e3961143b562b03aee91f62ae2224627ed12f42ce5a4865a4` |
| Original teacher checkpoint | `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb` |
| Tokenizer | `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09` |

Freeze all507 baseline tensors and all original E8/W5 indices, signs, balances
and scales. Do not inherit prototype tables or learned low-rank factors.
Use the unchanged rank-4 bank, SHA
`1948c89100750477e2bacda8013c97ff3e2ccbde0e444b961092f28c96c7f503`,
and strict factor format, SHA
`050bc995c14477189b5b919d8d75c31e6c18dd4815d0de54cb72da39f6e083d6`.

All112 projections receive B[m,4] and A[4,n]:56 input matrices18560×4096
and56 output matrices4096×8192. The224 FP32 masters contain7,827,456 values.
With a dedicated CPU generator seeded20260928, visit numeric layers0..55,
in_proj then out_proj. Draw each A as CPU FP32 normal/sqrt(n); B is zero and
does not consume RNG. Preserve the FP32 A masters. Use fresh optimizer/scaler.

Inside every block forward and checkpoint replay, compute exactly
`half(W0.detach().float() + B_master.half().float() @ A_master.half().float())`.
Require highest FP32 matmul precision, TF32 disabled and autocast disabled.
Use native FP16 projection/head/residual computation and ordinary cast
autograd; no decoder gradients or separate two-linear residual branch.
Rank four describes the product before final rounding. Zero-B addition may
change signed-zero bits; the base storage itself must stay bitwise unchanged.

## Fresh data with a reserved gate

Use the pinned WikiText-2 raw TRAIN stream, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`:2,533,678 tokens, two-newline join,
no automatic special tokens. Match full dataset/text/token/tokenizer identities
to original calibration. Exclude stored2048-token intervals intersecting:

| Existing scope | Intervals | Provenance |
| --- | ---: | --- |
| Original calibration |32 |`calibration/v1/manifest.json` |
| All-small training |256 |`training_data/small_compensation_v1/manifest.json` |
| Cross-moment fitting and diagnostic |48 |`reports/projection_crossmoment_pilot_v1.json` |
| Prototype training |32 |`training_data/prototype_compensation_v1/manifest.json` |
| Previous rank-4 training |256 |`training_data/low_rank_compensation_v1/manifest.json` |
| Newly observed diagnostic windows |64 |`reports/low_rank_generalization_v1.json` |

The previous rank-4 data-manifest SHA is
`e168e1d90ea7cd9c14c8210f13b5e6b569612c82dcaf832100c250cb72d24bef`;
it binds the first five scopes. The completed diagnostic SHA is
`2f75a73ad675f563b035f02ee3fab3522f011785eff3459b5c0caf098347da25`.
Resolve starts from these pinned receipts, not an unverified supplied list.

From the1237 complete2048-token grid blocks,519 remain eligible. First reserve
64 windows at `eligible[floor(i*518/63)]`, i=0..63. They are never training
inputs. Remove them, leaving455 entries; select448 training windows at
`remaining[floor(i*454/447)]`, i=0..447. Seven remain unused. Assert:

- Reserved first/last starts14336/2516992, minimum gap30720.
- Training first/last starts18432/2514944, minimum gap2048.
- Unused starts362496,718848,1083392,1439744,1802240,2158592,2510848.
- Every split and prior scope has zero half-open stored-token overlap.

Save separate CPU int64 token files and starts/per-window hashes for both
sets. Training stores917,504 tokens and has**917,056 next-token targets**;
reserved stores131,072 tokens and has**131,008 targets**. Each window uses
`window[:-1]` as input and `window[1:]` as targets,2047 targets with fresh state.
Train once: a single `torch.randperm(448)` using a separate CPU generator
seeded20260928 gives exactly448 successful updates. Batch one, no accumulation.
Reserved data may be prepared and integrity-checked before training, but must
not enter training/smoke forward passes, gradient computation or tuning.

## Objective and fixed optimizer

Teacher is the original checkpoint in the existing FP16 native runtime with
no teacher gradients. Student remains in eval-mode behavior with factor
gradients enabled. Temperature is1. The sole optimization loss is the mean
**KL(teacher || student)**, summed over the full256,000-token vocabulary and
all2047 targets, then divided by2047. Log next-token CE for diagnosis only:
its coefficient is0 and it must contribute no hidden/factor gradients.

Use FP16 native head GEMMs followed by FP32 log-softmax/loss. Sum64-token
chunks plus the63-token tail and normalize by total targets, not chunk count.
Stage hidden backpropagation with the scaler applied exactly once. Implement
the new loss helper separately; do not modify previous frozen loss utilities.

AdamW learning rate**1e-4**, betas(.9,.999), epsilon1e-8, weight decay0;
no scheduler. Clip unscaled global factor gradient norm at1. GradScaler starts
at1024, growth factor2, backoff.5, growth interval2000. Allow at most8 overflow
retries total, hence at most456 attempts. Retry the same scheduled window with
no factor/optimizer mutation. Record successful and attempted target exposures
separately. Nonfinite forward/loss, missing gradients, nonfinite masters or
frozen-base mutation fail immediately. Initially A gradients are exactly zero
because B=0; all B gradients must be present. No loss-mix/LR/epoch/rank sweep.

## Discarded smoke and durable final export

Before real training, freeze input/code/protocol/data bindings and perform one
discarded smoke on training inputs only:

1. CPU proof that staged pure KL gradients match an independent full-vocabulary
   KL calculation, including a short tail and loss scale; altering CE labels
   changes logged CE without changing optimization gradients.
2. Fresh initialization and zero-merge/native hidden/logit checks, recording
   signed-zero-only changes. Three native blocks0/18/55 must have matching
   checkpoint/non-checkpoint outputs and factor gradients under nonzero
   perturbation. Restore initialization afterward.
3. One complete2047-target update with finite gradients for all224 factors,
   initially112 nonzero B gradients and112 zero A gradients; a subsequent
   short TRAIN backward must give finite nonzero gradients to both families.
   Record time, memory, scale and gradient ranges, then discard trial state.
4. Write/read all112 strict FP16 factor files, match rounded masters and native
   merged hidden/last-eight logits exactly. Rehash all507 frozen base tensors
   before/after. Stop on any mismatch; code fixes require a new smoke.

Require fresh work/output paths and no resume/replay. Journal each attempt and
save atomic diagnostic checkpoints every64 successes, on caught termination
and at final448. Save FP32 masters, optimizer/scaler/RNG, flat schedule/history
and all binding hashes. A stopped/failed run does not authorize continuation or
more exposures. Evaluate only the final448 export.

Strict112 factor files contain15,654,912 value bytes plus3584 header bytes,
**15,658,496 bytes** total. Count the actual overlay manifest and any package
metadata separately. Base data are required; there is no same-capacity or
entropy-size claim. Bind source, raw/small parents, data, protocol/code, passed
smoke, final checkpoint, training report and all file hashes without cycles.
The protocol omits future preparation/data hashes; preparation records its
code/protocol/upstream hashes, then training binds the actual receipts.

An independent evaluator must read final224 masters and all112 factor files,
prove rounded equality, load the native accepted base, and independently merge
FP16 factors in FP32 then cast to FP16. No training-bank inference. Audit actual
507 baseline hashes,112 merged projections and395 unchanged tensors, plus
the final candidate507 contents and bound-file identities after scoring.

## Two-stage final-only evaluation

First evaluate source, accepted all-small and final candidate on the64 reserved
TRAIN windows in one process:131,008 targets, fresh state, native FP16 head,
FP32 full-vocabulary CE and teacher-to-student KL,64-token chunks. Pair all
window/token hashes; calculate PPL from summed NLL/targets and KL from summed
KL/targets. If source scoring repeats across passes, require exact per-chunk
source CE repetition. Reserved advancement requires both:

- Candidate PPL <=0.99 × same-process accepted-base PPL.
- Candidate mean teacher-to-student KL < same-process accepted-base mean KL.

A failed reserved gate is a complete negative experiment: stop and skip full
validation. Do not relax thresholds, select earlier checkpoints or promote it.
Only if both pass, score the same source/base/candidate on the existing full
validation split:130 fresh-state windows,264,764 targets, at most2048 targets
per window, native FP16 head and FP32 loss, chunk64. Report separately the
>=1% PPL improvement condition and source+5% quality target. The former does
not imply the latter. Do not compute test/MK metrics. Full validation has
already informed development and is not untouched test evidence.

No result implies recall recovery, a global minimum size, compressed GPU
residency or publication permission. The fixed recipe ends after its specified
gate and optional full validation; retain all negative outcomes and the existing
accepted all-small distribution unless a later explicit decision changes it.
