# Fixed rank-4 PPL repair job

Started **2026-09-28 10:28:31 UTC** on `horde-gpu`, with runner PID 2451867
and initial trainer PID 2451868. The launch and first live process identities
were verified through Linux `/proc`. A PID written in this document is historical;
use the checker below for current state.

The job follows the frozen [protocol](LOW_RANK_COMPENSATION_PROTOCOL.md):
accepted all-small baseline, all 112 input/output projections, rank four,
256 new TRAIN windows, four seeded passes and exactly 1,024 successful updates.
The [discarded GPU smoke](LOW_RANK_SMOKE_RESULTS.md) passed before launch.
Only the final export is eligible for independent full validation. No resume,
extra epochs, intermediate validation selection, MK or publication is scheduled.

## Read status

```sh
ssh horde-gpu 'cd /home/horde/Mamba2-8B-E8W5 && /home/horde/.venvs/lodram/bin/python scripts/check_low_rank_job.py'
```

An optional bounded wait uses `--milestone 128 --wait-seconds 45`. The checker
matches PID, process start ticks, boot ID and exact command before reporting
a live process. It reads receipts and logs; it cannot restart or modify the job.

Remote outputs:

- `reports/low_rank_compensation_v1_job.json`: process identities, stages and exit codes.
- `reports/low_rank_compensation_v1_train.json` and `.log`: update/loss/overflow history.
- `artifacts/low_rank_compensation_v1_train_work/`: journals and unique checkpoints.
- `artifacts/low_rank_compensation_v1/`: eventual final 112 factor files and manifest.
- `reports/low_rank_compensation_v1_eval.json` and `.log`: eventual three-arm validation.

Training and optimizer artifacts remain on the GPU host. The source model and
accepted baseline are reused unchanged. At launch the remote filesystem had
about 19 GiB available; this experiment writes factor checkpoints and reports,
not another full source-weight copy.

## Frozen launch identities

| Input | SHA256 |
| --- | --- |
| Training driver | `fdcb974e7abcd9379a6da900cd5801839e03973275010185f59235d67d938273` |
| Independent evaluator | `f0d1285276d1b4ea3bb132b98cb412a9718289b715d70ee1fa961c08bc3b0bd7` |
| Passed GPU smoke | `dacedf6401f082d172da31a5737c11d038998c939988999099f89b9620de0132` |
| Runner | `ad4becc8ffec91e71864b57909b4d68696d774f5af5a0904aea6a607332dad94` |
| Read-only checker | `46cca2177963e8cb673fb404860c4cf29f5823b98263d6b62d5d61947d17a7cb` |

The [launch receipt](../reports/low_rank_compensation_v1_launch.json) records
the exact command, time and initial process identity. Launch does not establish
training completion, export correctness or improved PPL. The final report must
separately state the 1% improvement and source+5% outcomes. MK remains deferred,
and publication remains on hold.
