# Independent rank-4 final-export evaluation

The evaluator is prepared and CPU checked. No quality run is implied by this
document. Run only after the fixed training recipe has completed, exported its
final 1024-update candidate, and handed off the GPU.

## Inputs and command

`scripts/evaluate_low_rank_compensation.py` requires:

- `--overlay-dir`: the final directory with exactly 112 `.lrf` files and its manifest.
- `--training-report`: the completed training JSON bound by that manifest.
- `--training-checkpoint`: the actual file named by the report's `final_checkpoint`.
- `--smoke-report`: the matching passed, discarded smoke receipt.
- `--report`: a fresh output path outside all immutable input directories.

The source, original E8/W5 parent, accepted all-small overlay, norm initialization,
fresh TRAIN data, protocol and baseline report default to their repository paths.
Use `--verify-only` with `CUDA_VISIBLE_DEVICES=''` for CPU integrity alone. An
actual full-quality run must use a different fresh report path.

```sh
cd /home/horde/Mamba2-8B-E8W5
/home/horde/.venvs/lodram/bin/python scripts/evaluate_low_rank_compensation.py \
  --overlay-dir artifacts/low_rank_compensation_v1 \
  --training-report reports/low_rank_compensation_v1_train.json \
  --training-checkpoint /absolute/path/from/final_checkpoint.file \
  --smoke-report reports/low_rank_compensation_v1_smoke.json \
  --report reports/low_rank_compensation_v1_eval.json
```

The checkpoint placeholder must be replaced with the actual completed receipt's
path. This evaluator has no training, checkpoint selection or resume option.

## Verification and native computation

The CPU stage verifies all hashes, strict rank-4 file geometry, all 224 FP32
masters rounded to the actual FP16 factor bytes, the fixed 1024-window schedule,
overflow history, smoke/report/checkpoint bindings and the 507-tensor base ledger.
It independently reconstructs the 256-window TRAIN selection from pinned prior
intervals and checks every stored window digest. It imports neither the trainer
nor the low-rank training bank.

Native quality uses three arms in one process: original source FP16, accepted
all-small E8/W5, and the independently reloaded low-rank candidate. Each projection
uses exactly `(W0.detach().float() + B16.float() @ A16.float()).half()` with local
autocast disabled, TF32 off and highest FP32 matmul precision. There is no
two-linear residual branch. All 112 merged tensor hashes are recorded; the 393
small tensors and two W5 vocabulary tensors retain their objects and contents.
The complete candidate's 507 tensor hashes are checked again after scoring.

The frozen evaluation math covers 264,764 validation targets in 130 paired,
fresh-state windows, with up to 2048 targets per window and logits chunks of 64.
The report records every window, target-weighted NLL/PPL and separate criteria:

- `comparison.meaningful_improvement_reference.met`: at least 1% lower PPL than
  the same-process accepted all-small baseline.
- `comparison.source_plus5_percent_reference.met`: at most 1.05 times the
  same-process original source PPL.

Validation has informed development, so these are not untouched test results.
MK is deferred; neither gate establishes recall recovery or publication approval.
The factors add 15,658,496 raw bytes plus the measured candidate manifest, and
evaluation expands weights to FP16. Timing is not compressed-runtime throughput.

## Prepared evidence

`reports/low_rank_evaluation_cpu_tests.json` records six passing CPU checks,
including actual fresh TRAIN data. The independent static review covers the
same evaluator SHA. A future final-export receipt remains necessary; synthetic
checkpoint tests do not validate an unproduced candidate.
