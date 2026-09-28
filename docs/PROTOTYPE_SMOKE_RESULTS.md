# Prototype compensation: correctness smoke passed

**The fixed 128-update training run has not started.** The first complete
smoke update took **154.61 seconds**. A bounded performance investigation must
distinguish first-use work from steady-state cost before starting that run.
Publication remains on hold; no PPL or recall improvement is established here.

## Verified results

The [smoke receipt](../reports/prototype_compensation_v1_smoke.json) records:

- All **112 zero-table decoded projections** match the current baseline exactly.
  Initial 128-token hidden states and last-eight-token full-vocabulary logits
  also match bitwise.
- After the declared perturbation of every table, blocks 0, 18 and 55 have
  identical checkpoint/non-checkpoint outputs and compared gradients;
  maximum gradient difference is **0**.
- One complete **2,047-target CE/KL update** succeeds with **zero overflows**.
  All 112 tables have finite, nonzero gradients; 229,375 of 229,376 scalar
  gradients are nonzero. One zero scalar does not mean a missing table gradient.
- All **507 baseline parameter hashes** remain unchanged and match the pinned
  original E8/W5 plus final all-small baseline.
- All 112 rounded tables change. Independent CPU readback verifies their
  headers, file and payload hashes, finite values and exact inventory:
  **462,336 file bytes**, including **458,752 FP16 value bytes**.
- Decoding the exported tables into native FP16 projection weights reproduces
  functional hidden states and last-eight-token logits **bitwise**, with maximum
  difference **0**. Original parameter references are restored afterward.

The smoke optimizer and master updates were discarded. They are not the
initialization of the formal training run.

## Timing and memory scope

| Measurement | Result |
| --- | ---: |
| First full update, CUDA-synchronized | 154.608914 s |
| Full-update peak allocated GPU memory | 36,566,708,224 B |
| Entire smoke, including loading and checks | 422.481729 s |
| Entire smoke peak allocated GPU memory | 46,710,476,800 B |

The update peak includes the resident teacher, frozen baseline and training
workspace. The higher overall peak includes temporary decoded FP16 projections
used for export verification. Neither is compressed inference residency.

Multiplying the first update by 128 gives about **5.50 hours**, but that is a
planning extrapolation, not a measured steady-state training duration. No
training recipe, forward arithmetic or checkpoint choice has been changed.

## Evidence

- [Frozen protocol](PROTOTYPE_COMPENSATION_PROTOCOL.md)
- [CPU driver checks](../reports/prototype_driver_cpu_audit_v1.json)
- Smoke receipt SHA-256:
  `05f41c5607b90da8b9cc68924c854655c12fe8ce1b76450c6ca5974dead85a6f`
- Training driver SHA-256:
  `bc6f59da1671d0e8973984f8ce24d176c364a128086f17b57b34c1f4022e1bc0`

This establishes smoke correctness and observed resource use only. Full
128-update export and independent full-validation quality remain outstanding.
