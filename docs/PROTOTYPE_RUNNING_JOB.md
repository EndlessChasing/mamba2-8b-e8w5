# Fixed prototype training and automatic full validation

The original validated decoder is used without a numerical substitution.
The faster explicit matrix multiplication probe failed decoded FP16 bitwise
parity, so it was not adopted for this run.

The fixed job was launched on the GPU host on 2026-09-28 at approximately
04:21 UTC (September27,21:21 PDT). Runner PID1379083; initial training PID1379084.
These PIDs identify this launch and are not a permanent indication of liveness.

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
- `reports/prototype_compensation_v1_eval.json`: independent final full-validation result, when available.

After128 successful TRAIN updates and export, a fresh process verifies the
serialized tables/checkpoint/baseline and scores source FP16, best all-small
E8/W5 and the prototype candidate over all264,764 validation targets. Training
or integrity failure stops the sequence. No intermediate PPL, checkpoint
selection, MK, test evaluation, upload or publication is performed.

The smoke update took154.61seconds. A naive128-update extrapolation is5.50hours
before startup/export/validation; this is a planning estimate, not a measured
job completion time. The separate bounded decoder probe confirmed substantial
steady-state backward cost. All507 baseline tensors and original E8/W5 payloads
remain frozen; only229,376 prototype master values are trained.

At launch, no prototype full-validation PPL exists. The best measured compact
candidate remains all-small E8/W5 at PPL8.359867548 versus source7.334175947.
Publication stays on hold and recall recovery remains unmeasured.

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
content hash. Final export performs the separate frozen-content verification.

The audit retains exact copies of the single-read training report and committed
journal prefix, because the live files continue to change:

- [CPU audit](../reports/prototype_compensation_v1_checkpoint32_audit.json), SHA
  `412d47138bed420d289d2de5ed072eabdf4a2ecab7a8776b52116d72b89a0e13`.
- [Training-report snapshot](../reports/prototype_compensation_v1_checkpoint32_audit.train_snapshot.json).
- [Completed journal prefix](../reports/prototype_compensation_v1_checkpoint32_audit.journal_prefix.jsonl).

No intermediate export, PPL, MK, resume or publication was performed. The fixed
128-update recipe and automatic final full-validation sequence remain unchanged.
