# Rank-4 compensation: discarded GPU smoke

The fixed all-projection experiment passed its discarded correctness and
feasibility smoke on the RTX PRO 6000 Blackwell Server GPU. This is not a PPL
result. Formal training must start from the original zero-B initialization,
with a new optimizer and scaler, following the unchanged
[protocol](LOW_RANK_COMPENSATION_PROTOCOL.md).

## Actual checks

- All 112 zero-B merges equal their decoded FP16 base numerically. The declared
  FP32 addition canonicalizes 140,175 signed-zero bits in temporary merged
  weights; the base itself remains bitwise unchanged. Initial native and
  functional hidden states and last-eight full-vocabulary logits match by hash.
- Native blocks 0, 18 and 55 have identical checkpoint/non-checkpoint forward
  values and gradient bits with deterministic nonzero factor perturbations.
  Initialization is restored after that test.
- One complete 2,047-target update succeeds with loss scale 1,024 and no
  overflow. All 112 B gradients are finite and nonzero; all 112 A gradients
  are present and zero, as required by B=0. A subsequent 128-target backward
  gives finite, nonzero gradients to all 224 factors, without another update.
- All 112 FP16 factor files read back exactly and match rounded FP32 masters.
  The files total **15,658,496 bytes**, including 3,584 header bytes.
- Installing the reloaded factors as native effective weights produces
  bitwise-identical 128-token hidden states and last-eight full-vocabulary
  logits, with maximum absolute difference zero.
- All 507 base tensor contents are rehashed and unchanged. The trial update,
  perturbed factors and optimizer state are discarded.

## Timing and memory

| Measured scope | Result |
| --- | ---: |
| One full update, including teacher and CE/KL backward | 2.062858 seconds |
| Full-update peak allocated GPU bytes | 36,211,180,032 |
| Full-update peak reserved GPU bytes | 36,834,377,728 |
| Whole smoke peak allocated, including temporary native export | 46,416,051,200 |
| Whole smoke peak reserved | 48,146,415,616 |
| Entire smoke, including load and integrity checks | 161.427867 seconds |

One update is a feasibility observation, not a sustained throughput benchmark.
The teacher and student use dense FP16 weights, with temporary dense merges and
gradients. These memory numbers do not demonstrate compressed GPU residency.

## Evidence

The [GPU receipt](../reports/low_rank_compensation_v1_smoke.json) is 577,732 bytes,
SHA256 `dacedf6401f082d172da31a5737c11d038998c939988999099f89b9620de0132`.
The process exited 0. Its binding includes:

- Trainer: `fdcb974e7abcd9379a6da900cd5801839e03973275010185f59235d67d938273`.
- Training bank: `1948c89100750477e2bacda8013c97ff3e2ccbde0e444b961092f28c96c7f503`.
- Protocol: `b732c24692ae5dc91c5c67c1a0e0a0c435be3402959bfe5db874c67246836cf2`.
- Fresh TRAIN manifest: `e168e1d90ea7cd9c14c8210f13b5e6b569612c82dcaf832100c250cb72d24bef`.

The [bank CPU checks](../reports/low_rank_training_cpu_tests.json) passed nine
tests, and the [driver preflight](../reports/low_rank_driver_cpu_preflight.json)
passed four accounting tests and verified 22 pinned input paths with CUDA hidden.

The next step is the fixed 1,024-update run and independent final-only paired
full validation. The accepted all-small baseline remains unchanged. MK stays
deferred, and publication remains on hold.
