# Completed prototype training and automatic full validation

**Terminal status:** training and independent full validation both exited0.
The job finished at2026-09-28 09:59:49 UTC (02:59:49 PDT). Prototype PPL is
**8.306585061749145** versus all-small**8.359867548009992** and source
**7.334175947318572**. Both declared quality gates failed: gain0.6374% is
below1%, and the source gap13.2586% exceeds5%. Retain the accepted all-small
model/distribution and stop this fixed recipe with no extra epochs. See
[complete results](PROTOTYPE_COMPENSATION_RESULTS.md).

The original validated decoder is used without a numerical substitution.
The faster explicit matrix multiplication probe failed decoded FP16 bitwise
parity, so it was not adopted for this run.

The fixed job was launched on the GPU host on 2026-09-28 at approximately
04:21 UTC (September27,21:21 PDT). Runner PID1379083; initial training PID1379084.
These PIDs identify the historical launch, not current liveness. The following
command records the original one-shot invocation:

```bash
/home/horde/.venvs/lodram/bin/python -u scripts/run_prototype_compensation_job.py \
  --report reports/prototype_compensation_v1_job.json
```

The runner rejects existing outputs. Do not repeat this command to monitor it.
Read the receipts and logs instead:

```bash
python scripts/prototype_job_status.py
```

This read-only Linux helper checks actual process command lines and states in
`/proc` as well as receipts. A receipt alone is not evidence that a job is live.

- `reports/prototype_compensation_v1_job.json`: process IDs, stage and terminal status.
- `reports/prototype_compensation_v1_train.json`: training binding and successful update count.
- `reports/prototype_compensation_v1_train.log`: detailed training output.
- `artifacts/prototype_compensation_v1_train_work/`: durable attempt journal and checkpoints.
- `reports/prototype_compensation_v1_eval.json`: completed independent final full-validation result.

After128 successful TRAIN updates and export, a fresh process verified the
serialized tables/checkpoint/baseline and scored source FP16, accepted all-small
E8/W5 and the prototype candidate over all264,764 validation targets. Both
stages completed successfully; all128 attempts succeeded with zero overflows
and262,016 target exposures. No intermediate PPL, checkpoint selection, MK,
test evaluation, upload or publication was performed.

The smoke update took154.61seconds. At launch, its naive128-update
extrapolation was5.50hours before startup/export/validation; that was a planning
estimate. The completed training report now measures19,981.123seconds
(about5.55hours, including load/export checks), followed by312.222seconds for
independent evaluation. The separate bounded decoder probe confirmed substantial
backward cost. All507 baseline tensors and original E8/W5 payloads stayed frozen;
only229,376 prototype master values were trained.

The completed [job receipt](../reports/prototype_compensation_v1_job.json) has
SHA256 `12b8786fb270c4a539f8c71fa0d7369f4ab3c9809539d6ed75c6edb1fb035733`.
It binds both exit0 statuses and the final evaluation SHA. The prototype is
numerically lowest among measured compact candidates, but did not meet its
advancement condition. Publication stays on hold and recall remains unmeasured.

## First periodic checkpoint: CPU audit completed

At 2026-09-28 05:47:58 UTC, the read-only checkpoint-32 audit passed. The
training job continued normally; this was neither an intermediate quality
measurement nor checkpoint selection.

- 32 successful updates, 32 attempts, zero overflows, 65,504 target exposures.
- The seeded schedule, live-report history prefix and all 32 journal pairs agree.
- All 112 FP32 tables (229,376 values) and their FP16 casts are finite. All 112
  tables are nonzero after rounding; this does not establish a PPL improvement.
- Adam state covers all 112 tables at optimizer step 32; loss scale is 1024.
- The 3,057,019-byte checkpoint matches its saved receipt before and after CPU
  loading. SHA: `d9b34363f36841684953dffc072d119e77a613ac57f97d90079ff99eeb7c88f2`.
- Bound source hashes still match. CUDA remained uninitialized in the auditor.

The 507 frozen tensor hashes in this checkpoint describe startup. Their
agreement with the passed smoke and training report is not a new step-32 GPU
content hash. Final export performed the separate frozen-content verification;
the independent evaluator then checked fresh native tensor content, as recorded
in the [final results](PROTOTYPE_COMPENSATION_RESULTS.md).

The audit retains exact copies of the single-read training report and committed
journal prefix, because those live files were changing during training:

- [CPU audit](../reports/prototype_compensation_v1_checkpoint32_audit.json), SHA
  `412d47138bed420d289d2de5ed072eabdf4a2ecab7a8776b52116d72b89a0e13`.
- [Training-report snapshot](../reports/prototype_compensation_v1_checkpoint32_audit.train_snapshot.json).
- [Completed journal prefix](../reports/prototype_compensation_v1_checkpoint32_audit.journal_prefix.jsonl).

No intermediate export, PPL, MK, resume or publication was performed. The fixed
128-update recipe and automatic final full-validation sequence are now complete.
The failed quality gates do not authorize additional epochs or publication.
