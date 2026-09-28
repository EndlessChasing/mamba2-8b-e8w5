# Existing-norm compensation artifact and evaluation

This is a separate experiment on the **original two-sweep E8/W5 parent**.
It does not use the eight-sweep candidate. See the fixed
[training and quality protocol](NORM_COMPENSATION_PROTOCOL.md).

The overlay contains only `manifest.json` and a replacement `other_fp16.pt`.
The replacement retains all 393 original FP16 tensor names and shapes:

- 113 existing norm tensors, totaling 692,224 values, are selected for training.
- The other 280 tensors must remain bitwise identical, including signed zero.
- A selected norm may remain identical after FP16 rounding.
- All 112 E8 projection files, both independent W5 vocabulary files, the
  codebook and configuration are inherited with their original hashes.

The verifier checks exact inventories, all file hashes, the original parent
identity, tensor contents, fixed protocol/code/hyperparameter identities, and
full 507-tensor / 8,236,999,680-parameter coverage. It rejects the eight-sweep
parent, additional tensors, different precision, missing entries, unsafe paths
and nonfinite values. Training receipts identify the final 128 successful
updates, deterministic schedule, exposure accounting and successful smoke.

## Final development result

The completed 128-update candidate was reloaded from its exported FP16 file
and compared with the original two-sweep parent in the same process:

| Metric | Parent | Norm candidate |
|---|---:|---:|
| Development PPL | 9.2371639507 | 8.9653964898 |
| Normal MK | 10/12 | 9/12 |
| Target-removed MK | 0/12 | 0/12 |
| Logical raw model payload | 2,886,482,462 bytes | 2,886,482,462 bytes |

PPL improved **2.9421%**, exceeding the predeclared 1% improvement threshold.
Normal MK lost the case `validation-n64-t2-s0`; all other paired outcomes
were unchanged. **The combined development gate failed**, and the evaluator
correctly skipped full validation. No test split, source-relative full-validation
claim, alternate training step or further recipe was evaluated.

All 507 loaded parent tensors passed the fresh audit, including independent W5
readback. All 113 final trained norms matched the checkpoint's rounded FP32
masters and exported FP16 values. The other 280 small tensors stayed bitwise
unchanged, and the other 394 parameter objects were retained. The replacement
archive is 7,283,930 bytes, exactly the parent's serialized size in this run;
its FP16 tensor payload is 7,161,856 bytes. The physical parent-plus-overlay
directories total 2,894,033,233 bytes, including both manifests, excluding
tokenizer, software, training checkpoints and reports.

Evidence: [final evaluation report](../reports/norm_compensation_v1_eval.json),
SHA-256 `72350eaa1989601f4412ba9e31c178e22dba95ec0b85f5837f9f08b29b265dad`.
The GPU evaluation stage took 156.85 seconds and peaked at 22,062,521,856
Torch-allocated bytes. It used the expanded FP16 accuracy runtime.

## Loading the exported candidate

```python
from mamba_e8w5.norm_overlay import load_norm_model

model = load_norm_model(
    "artifacts/e8w5_v1",
    "artifacts/norm_compensation_v1",
    device="cuda",
)
```

This reloads the actual FP16 export into the existing frozen runtime. The
original floating-point checkpoint, optimizer state and calibration are not
candidate inference dependencies. This accuracy runtime still expands weights
to FP16; it is not a compressed-residency runtime. The parent remains on disk
as a declared dependency, including its superseded small-tensor file.

## Verification and paired quality evaluation

Use `scripts/evaluate_norm_compensation.py` with explicit paths to:

- The original parent directory and completed norm overlay.
- The pinned tokenizer/source directory.
- Calibration manifest and token file.
- Final training report, final checkpoint and successful smoke report.
- The frozen norm-compensation protocol and a new output report path.

`--verify-only` performs CPU checks. Evaluation additionally audits every
loaded parent tensor: 112 decoded E8 hashes, both vocabularies against an
independent W5 decoder, and all 393 small FP16 tensors. It then reloads only
the selected norms, checks the exported small tensors exactly, and retains
the other 394 parameter objects.

Parent and candidate run in the same process on the fixed four validation
windows and 24 public MK cases. The predeclared gate requires at least 1%
PPL improvement, no lower normal MK count, and no higher target-removed
count. Inputs and paired case identities must match exactly.

`--full-validation-if-improved` runs only after that gate passes. It evaluates
source, parent and candidate on all 264,764 validation targets in the same
process. The original source checkpoint is a declared reference-only input
for this stage. It separately reports the source-relative +5% PPL target;
the parent-relative improvement gate does not establish source-relative
quality or full recall preservation. No test split or intermediate training
checkpoint is used for selection.

## Storage scope

The 1,384,448-byte norm tensor payload already existed. Report actual serialized
replacement size, overlay metadata, all inherited payload bytes, and physical
parent-plus-overlay disk use. Optimizer checkpoints, smoke/training/evaluation
reports, calibration and source weights are separate experiment inputs or
outputs. Entropy-coded size must be measured for this export before claiming
the original package's compressed distribution size.

Boundary tests are in `tests/test_norm_overlay.py`; they exercise signed-zero
preservation, forbidden mutations, precision/shape/key changes, nonfinite values,
unsafe paths and the paired development gate.
