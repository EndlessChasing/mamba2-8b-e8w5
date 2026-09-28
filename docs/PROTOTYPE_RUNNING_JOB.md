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
