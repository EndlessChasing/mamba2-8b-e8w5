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
