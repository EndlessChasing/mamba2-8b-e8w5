# All-small compensation: independent evaluation

This evaluates the sole final 1024-update candidate declared in
[the fixed training protocol](SMALL_TENSOR_COMPENSATION_PROTOCOL.md). The helper
is `scripts/evaluate_small_compensation.py`. Earlier reports and the original
norm combined-gate failure remain unchanged.

## Inputs and audit

The candidate contains only the replacement `other_fp16.pt` and its manifest.
Its 393 existing small FP16 tensors have the original 3,580,928 parameters.
All 112 E8 projection files, both W5 vocabulary files, config and codebook are
inherited from the original two-sweep parent. The small-overlay verifier checks
these identities, training data, initialization, fixed source hashes, smoke and
final checkpoint. The evaluator independently repeats the final FP32-master to
FP16-export hash comparison for every small tensor.

A fresh GPU parent load is audited against all 112 decoded E8 hashes, an
independent CPU W5 decoder for both vocabularies, and all 393 raw FP16 tensors.
After applying norm-v1 and the all-small candidate, the evaluator hashes all
393 actual loaded small tensors against their respective export. It also
rehashes every one of the 114 frozen projection/vocabulary tensors and checks
complete 507-tensor / 8,236,999,680-parameter coverage. These are content checks,
in addition to the overlay loader's parameter-object checks.

## Fixed validation

One process evaluates source FP16, original E8/W5, norm-v1 and the all-small
candidate, in that order. Source is freed before the compressed parent loads.
The same 130 windows cover all 264,764 next-token targets, at most 2048 targets
per window, with a final 572-target window. Token and dataset hashes must equal
the completed norm PPL continuation receipt. Evaluation uses frozen
`evaluation.py`: native prefill, fresh state per window, FP16 vocabulary GEMM,
FP32 cross entropy and 64-token logit chunks. This is expanded FP16 inference.

Every arm includes all per-window losses; the evaluator separately checks the
target-weighted aggregate NLL and `exp(mean NLL)`. The report includes paired
window differences, all source-relative PPL gaps, the predeclared 1% full-PPL
reduction threshold against norm-v1, and the separate `PPL <= 1.05 * source`
reference. Complete validation runs for every valid final export regardless of
those results. MK is not measured or a continuation gate in this experiment.
Validation has informed development; no untouched-test claim is supported.

## Invocation after GPU handoff

Run from `/home/horde/Mamba2-8B-E8W5`. Replace the final checkpoint filename only
with the exact `final_checkpoint.file` reported by the completed training run.
Do not infer it from an in-progress directory or choose a checkpoint by PPL.

```sh
/home/horde/.venvs/lodram/bin/python -u scripts/evaluate_small_compensation.py \
  --source-dir models/source \
  --parent-dir artifacts/e8w5_v1 \
  --initialization-dir artifacts/norm_compensation_v1 \
  --overlay-dir artifacts/small_compensation_v1 \
  --data-dir training_data/small_compensation_v1 \
  --training-report reports/small_compensation_v1_train.json \
  --training-checkpoint FINAL_CHECKPOINT_FROM_COMPLETED_TRAINING_REPORT \
  --smoke-report reports/small_compensation_v1_smoke.json \
  --report reports/small_compensation_v1_eval.json \
  > reports/small_compensation_v1_eval.log 2>&1
```

The helper refuses an existing report. A CPU-only `--verify-only` mode is
available for a distinct integrity receipt; it does not load datasets/models
or evaluate PPL. Use it only if an additional integrity receipt is needed.
Keep large model/training files on the GPU host. Copy only code, docs, receipts
and logs to the Mac. Do not run concurrently with training. No test split,
new container, upload or publication is authorized by this runbook.

## Size accounting

Report actual replacement archive/manifest bytes and logical resolved-model
raw bytes. Also report physical parent plus both overlay directories, explicitly
excluding source, tokenizer, software, reports, calibration and training files.
The fixed small-tensor capacity is 7,161,856 bytes; serialized ZIP metadata is
measured separately. Existing Huffman container sizes do not describe the
changed values. Training peaks and expanded-FP16 evaluation peaks are separate.
