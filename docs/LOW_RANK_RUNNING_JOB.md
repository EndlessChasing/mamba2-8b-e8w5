# Completed rank-4 PPL repair job

**Execution completed; both quality gates failed.** Training and independent
full validation exited with code zero. The fixed recipe is stopped, and the
accepted all-small model/distribution remains unchanged. See the
[result and receipt ledger](LOW_RANK_COMPENSATION_RESULTS.md).

The job started **2026-09-28 10:28:31 UTC** on `horde-gpu`. Training finished
at **11:14:10 UTC**, and evaluation finished at **11:19:07 UTC**. Runner PID
2451867, trainer PID 2451868 and evaluator PID 2586102 are historical process
identities, not claims that those processes are still running.

The frozen [protocol](LOW_RANK_COMPENSATION_PROTOCOL.md) used the accepted
all-small baseline, all 112 input/output projections, rank four, 256 new TRAIN
windows and four seeded passes. The [discarded GPU smoke](LOW_RANK_SMOKE_RESULTS.md)
passed before launch. The completed run records **1,024 successful updates in
1,024 attempts, zero overflows and 2,096,128 target exposures**. No resume,
extra epochs or intermediate validation selection occurred.

All 224 rounded final masters matched the 112 serialized factor files.
Native export hidden states and last-eight-token logits were bitwise equal,
and all 507 frozen baseline tensor hashes were unchanged. Independent native
full validation then measured PPL **7.334175947 / 8.359867548 / 8.630189440**
for source / accepted all-small / low-rank candidate. Candidate PPL worsened
**3.2336%** versus the base; both the 1% advancement and source+5% gates failed.

## Read the completed job record

```sh
ssh horde-gpu 'cd /home/horde/Mamba2-8B-E8W5 && /home/horde/.venvs/lodram/bin/python scripts/check_low_rank_job.py'
```

The checker reads receipts and logs; it cannot restart or modify the job.
Its live-process checks require matching PID, process start ticks, boot ID
and exact command. No milestone wait is needed for this completed run.

Remote outputs:

- `reports/low_rank_compensation_v1_job.json`: completed stages and both exit codes.
- `reports/low_rank_compensation_v1_train.json` and `.log`: complete update/loss/overflow history.
- `artifacts/low_rank_compensation_v1_train_work/`: journals and checkpoints, including the final step 1,024.
- `artifacts/low_rank_compensation_v1/`: final 112 factor files and manifest.
- `reports/low_rank_compensation_v1_eval.json` and `.log`: completed three-arm full validation.

Model and optimizer binaries remain on the GPU host. Training/evaluation
receipts and a manifest copy are available locally; the source model and
accepted baseline were reused unchanged.

## Frozen launch identities

| Input | SHA256 |
| --- | --- |
| Training driver | `fdcb974e7abcd9379a6da900cd5801839e03973275010185f59235d67d938273` |
| Independent evaluator | `f0d1285276d1b4ea3bb132b98cb412a9718289b715d70ee1fa961c08bc3b0bd7` |
| Passed GPU smoke | `dacedf6401f082d172da31a5737c11d038998c939988999099f89b9620de0132` |
| Runner | `ad4becc8ffec91e71864b57909b4d68696d774f5af5a0904aea6a607332dad94` |
| Read-only checker | `46cca2177963e8cb673fb404860c4cf29f5823b98263d6b62d5d61947d17a7cb` |

The [launch receipt](../reports/low_rank_compensation_v1_launch.json) records
the initial command and process identity; the
[completed job receipt](../reports/low_rank_compensation_v1_job.json) records
both successful process exits. Completion and integrity do not constitute a
quality pass. No factor overlay was promoted or added to the accepted offline
distribution. MK remains deferred, and publication remains on hold.
