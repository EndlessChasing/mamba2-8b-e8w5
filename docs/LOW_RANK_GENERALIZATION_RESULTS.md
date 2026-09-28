# Final rank-4 fit-versus-fresh TRAIN diagnostic

The posthoc diagnostic completed successfully on the same fixed final1024
rank-4 export that failed [full validation](LOW_RANK_COMPENSATION_RESULTS.md).
It performed no training, checkpoint selection, validation/test computation
or promotion. The accepted all-small distribution remains unchanged.

## Fixed-model scores

| Window set | Targets | Source PPL | All-small PPL | Final rank-4 PPL | Candidate vs all-small |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original256 fitting TRAIN windows |524,032 |7.671760022 |8.714096599 |4.247929608 |−51.2522% |
|64 new disjoint TRAIN windows |131,008 |7.434059360 |8.510325047 |8.691613391 |+2.1302% |

| Window set | All-small mean teacher KL | Rank-4 mean teacher KL | Candidate change |
| --- | ---: | ---: | ---: |
| Fitting |0.2480272731 |0.2364773575 |−0.0115499156 |
| Fresh |0.2483324246 |0.2921350857 |+0.0438026611 |

KL is teacher-source→student, temperature1, summed over the complete256,000
token vocabulary and divided by the target count. NLL/PPL uses the same
target weighting. Source-to-source KL is0 by definition.

All256 fitting windows improve in NLL;170 improve and86 worsen in teacher KL.
On fresh windows,12 improve and52 worsen in NLL;**all64 worsen in teacher KL**.
The candidate's change in mean NLL versus all-small is−0.7185102999 on fitting
data and+0.0210784450 on fresh data: a difference of0.7395887449 nats/target.

These are final fixed-model scores, unlike the earlier online training losses
measured while parameters changed. The large fitting benefit with worse fresh
PPL and teacher agreement strongly supports overfitting in this recipe.
Different text subsets can have different difficulty; this diagnostic does
not isolate a causal mechanism or prove that CE alone caused the failure.

## Data and identity checks

- The pinned WikiText-2 raw TRAIN stream and original tokenizer are unchanged.
  Every window stores2048 tokens and scores2047 targets with fresh state.
- The256 fitting windows exactly match the previous training input. The64
  fresh windows are spread deterministically over583 remaining complete-grid
  blocks using `eligible[floor(i*582/63)]`. Their stored-token intervals have
  zero overlap with fitting data or any prior calibration/repair/diagnostic
  intervals; they are now observed diagnostic data.
- Native FP16 head GEMMs precede FP32 full-vocabulary CE/KL in64-token chunks
  with a63-token tail. No teacher/student gradients are constructed.
- Source CE repeats exactly, per chunk and per window, across both student
  passes for all320 windows. All paired starts/token hashes and target counts
  match. Report aggregates were independently recomputed from the rows.
- All507 loaded baseline tensors match the accepted all-small identity.
  Independently merged112 projections and the complete507 candidate content
  hashes match the already evaluated final export before and after scoring.
  Bound file identities were rechecked. This adds no new candidate.

The diagnostic elapsed399.5803 seconds. Its scope is descriptive TRAIN
generalization, not an acceptance gate, untouched test score, recall result,
compressed-runtime benchmark or publication permission.

## Evidence and next bounded experiment

[Diagnostic receipt](../reports/low_rank_generalization_v1.json):3,052,355 bytes,
SHA256 `2f75a73ad675f563b035f02ee3fab3522f011785eff3459b5c0caf098347da25`.
The [diagnostic script](../scripts/diagnose_low_rank_generalization.py) has SHA256
`e2879c439738ca1ff2e920d4f41cc183d7383fc7c918c88a2bcdc0fa18dbf879`.
The fixed final evaluation SHA is
`efb4264bc65efded5fa9b05f9ea689542f9feaf23d2108a1f7c335bee6c67979`.

A separate [teacher-KL protocol](TEACHER_KL_COMPENSATION_PROTOCOL.md) resets to
accepted all-small with fresh rank-4 factors. It declares pure teacher KL,
CE logging only, learning rate1e-4 and a single448-window pass, with64 new
reserved windows. The reserved gate must show both>=1% PPL improvement and
lower teacher KL before any full-validation run. This changes several recipe
components together and cannot isolate one cause. The new experiment has not
run or shown a repair; there are no extra epochs for the failed candidate.
MK remains deferred and publication remains on hold.
