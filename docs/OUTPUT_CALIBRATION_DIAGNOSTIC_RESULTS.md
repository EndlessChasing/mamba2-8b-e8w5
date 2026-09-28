# Output-calibration diagnostic: completed v2 result

The separately declared v2 diagnostic completed with exit code 0 and all
numerical and identity controls accepted. It measures **opposite local
temperature directions** for accepted all-small and the failed teacher-KL
candidate, alongside lower top-1 accuracy for both than the source. No
correction was fitted or applied, so this is **not an actual PPL repair**.
The failed v1 attempt and bounded execution probe remain recorded below.

## Scope and controls

Follow the [frozen v2 protocol](OUTPUT_CALIBRATION_DIAGNOSTIC_V2_PROTOCOL.md),
SHA256 `57058d5f78777a5cf12127bef628bde49f9e2a589a862b2cc459db6da63b6ecb`.
The three fixed native models are original source cast to FP16, accepted
all-small, and the final 448-step teacher-KL candidate. Beta is exactly 1.
All **64 already-observed reserved TRAIN windows / 131,008 targets** completed
with fresh state, native FP16 head GEMMs and FP32 full-vocabulary metrics in
64-token chunks plus the 63-token tail. This set is not fresh quality evidence.

Verified controls:

- First-window new/frozen paired scorer CE and teacher KL match exactly for
  both student arms. Source CE repeats exactly across both passes on all 64
  windows; the repeated source diagnostic statistics also match.
- Starts, token hashes and counts match the pinned prior evaluation.
- Actual 507 base/candidate tensor identities, all 112 native factor merges
  and 395 inherited tensors pass, with final content/file audits.
- Historical per-chunk CE differences are recorded. In this v2 run all three
  arms have zero aggregate PPL drift, and every recorded historical CE chunk
  matches. The predeclared 0.1% descriptive consistency ceiling passes without
  using its tolerance. This does not explain the earlier failed run or promise
  bitwise reproducibility across future processes.
- Bounded CPU derivative, offset and tail controls passed before GPU execution.
  This report records computation completion separately from numerical-control
  acceptance; both are true.

The diagnostic took **260.0261 seconds** internally; process wall time was
**262.2935 seconds**. No weights, temperature or candidate status changed.

## Entropy, local slope and curvature

All means are weighted by target count. Entropy and CE are in nats. With
`beta = 1/T`, `g = dCE/d beta at 1 = CE - H`; curvature is the centered
log-probability variance. PPL uses the original chunk-summed CE control.
Diagnostic per-token sums retain their separately reported FP32 reduction
rounding and are not substituted for that control.

| Model | PPL | Mean entropy H | Mean slope g | Mean curvature | Top-1 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original source FP16 |7.3690860306 |1.990534522 |+0.0067591622 |3.910573976 |55.672936% |
| Accepted all-small |8.3972987489 |2.055041732 |+0.0728683461 |4.165482007 |53.471544% |
| Final teacher-KL |8.5716990714 |2.184432336 |-0.0359663658 |4.502776990 |52.927302% |

For accepted all-small, an **infinitesimal temperature increase** locally
lowers aggregate CE. For the teacher-KL candidate, an **infinitesimal temperature
decrease** locally lowers it. Source has a smaller positive slope. Positive
window-mean slopes occur in 41/64 source windows and 59/64 base windows;
candidate window-mean slopes are negative in 52/64. These are local derivative
statements, not measured finite-temperature improvements or fitted optima.

The teacher-KL candidate raises entropy while losing another **0.54424
percentage points** of top-1 accuracy versus all-small. All-small itself loses
2.20139 points versus source. Teacher KL still improves from 0.2480998971 to
0.1943251635, while PPL worsens 2.0768622%. Thus better teacher-distribution
matching coexists with worse observed-token prediction and lower top-1 accuracy.
High entropy alone does not establish miscalibration.

## Rankings and ties

Rank is `1 + count(logit > target_logit)`. Native argmax uses its lowest-index
tie rule; a tied top target can have rank 1 without being the argmax output.
The table reports target counts, not percentages.

| Strict-greater rank | Source | All-small | Teacher-KL |
| --- | ---: | ---: | ---: |
|1 |73,057 |70,173 |69,481 |
|2–5 |30,616 |31,066 |31,400 |
|6–10 |8412 |8916 |9138 |
|11–100 |13,430 |14,540 |14,655 |
|101–1000 |4272 |4990 |5022 |
|>1000 |1221 |1323 |1312 |
|Targets tied with another logit |7417 |9122 |9366 |
|Native top-1 correct |72,936 |70,052 |69,339 |

Each model's six rank bins sum to 131,008. Both compressed models show top-1
ranking loss relative to source; the candidate is also worse than the accepted
base on this measure. A positive scalar temperature preserves exact-logit
ranks, so scalar calibration cannot repair these ranking differences. This
does not quantify how much of the NLL gap is attributable to ranking.

## Loss by source-token difficulty

Bins are fixed by source surprisal and contain the same target positions for
all arms. Delta is teacher-KL candidate minus all-small, using paired per-token
NLL sums. All six bin totals worsen; this does not mean every individual token
worsens. Across windows, six improve and 58 worsen in NLL.

| Source surprisal, nats | Targets | Summed NLL delta | Mean NLL delta per target |
| --- | ---: | ---: | ---: |
|[0,1) |63,975 |+250.603782 |+0.003917214 |
|[1,2) |20,661 |+230.366072 |+0.011149803 |
|[2,4) |24,061 |+497.432909 |+0.020673825 |
|[4,8) |17,635 |+457.133274 |+0.025921932 |
|[8,16) |4599 |+1034.349986 |+0.224907586 |
|[16,infinity) |77 |+223.100373 |+2.897407445 |

The 4676 targets with source surprisal at least 8 nats are **3.56925%** of the
set and contribute **1257.4503595 / 2692.9863958 = 46.69353%** of the total
paired per-token NLL increase. This is a descriptive concentration of loss on
this observed set, not a causal attribution. Per-token sums differ slightly
from chunk-reduced NLL totals because of FP32 reduction order.

## Preserved v1 failure and execution probe

The first fixed-beta attempt stopped with exit code 1 at
`Original beta1 CE does not reproduce exactly: source_fp16/0`, before accepting
any window row. Its report remains `complete: false`, with zero source rows
and empty summaries. The process supervisor's `complete: true` means only that
it finished recording the failed child. Wall time was 57.5164 seconds and
internal report time 55.1897 seconds. It provides no diagnostic metric result.

The subsequent bounded window-0 probe completed with all nine same-process
old/new/repeat comparisons exact across all 32 chunks and source/base 507-tensor
audits passing. Its historical chunks differed: source total NLL by
-0.06638336181640625 / 2047 targets and base by +0.222503662109375 / 2047.
The cause remains unproved. No tolerance reclassified v1 as a pass. V2 used
its separately frozen same-process controls and disclosed historical drift.

## Conclusions and limits

There is evidence of differing **local confidence directions** and of ranking
loss. No beta was fitted, no Newton step or sweep was evaluated, and no finite
gain was measured. These observations cannot establish that global calibration
dominates compression loss, identify one causal mechanism, or predict a repair
on unseen data.

The [teacher-KL candidate](TEACHER_KL_COMPENSATION_RESULTS.md) still fails its
reserved advancement gate. Its full-validation/source+5% target remains
unmeasured. Accepted all-small full-validation PPL **8.359867548009992** and
its existing distribution remain unchanged. The prior prototype's numerical
full-validation low **8.306585061749145** failed its 1% advancement gate.
These validation values are from a different split than this diagnostic.
Any later correction requires a separate TRAIN protocol and independent gate.
There is no promotion, new full-validation/test/MK result or publication.

## Evidence

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
|[Completed v2 diagnostic](../reports/output_calibration_v2.json) |2,637,918 |`e401ea26a84fcf2d91ed053e8995b0ddf78e44cf0b613c7897f5e72c7c8a37f0` |
|[V2 process / exit](../reports/output_calibration_v2_process.json) |1026 |`7245ca7834c25d09837d0bf91db19d3e080a343114db2835eb36b7ba74d1550f` |
|[Preserved incomplete v1](../reports/output_calibration_v1.json) |245,283 |`38c31d728bc44dba9882b3e2344aeae1feec746a48d500dcbe27264b6b3a374c` |
|[V1 process / exit](../reports/output_calibration_v1_process.json) |1077 |`2b311d9d729e176b9d2a03d04277cebce4ce8d36300ad6a259cfa78c03f77aee` |
|[Bounded execution probe](../reports/output_calibration_anchor_probe_v1.json) |933,066 |`b83fab9ae854e5b3622ee0960fcdc51e3d4e12210651cc5718ea2ea2be067e40` |
