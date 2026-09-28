# Final norm-compensation evaluation handoff

Training uses `artifacts/norm_compensation_v1_train_work`. Evaluate only the
completed candidate `artifacts/norm_compensation_v1` and the final checkpoint
named in `reports/norm_compensation_v1_train.json`. Do not guess the checkpoint
filename: overflow retries can change the attempt number.

The successful smoke receipt is `reports/norm_compensation_v1_smoke.json`, SHA
`c01afe9dd5690bc65f911ff47b0d7e0b82b23e12155331d7f6d9ca7847d4c115`.
The original two-sweep parent remains `artifacts/e8w5_v1`.

## Commands prepared for the GPU host

Run from `/home/horde/Mamba2-8B-E8W5` with
`/home/horde/.venvs/lodram/bin/python`. The following resolves the final
checkpoint from the completed report and builds a structured command without
shell interpolation. Initially select `phase = "verify"`; run this only after
training has completed. Select `phase = "evaluate"` only after explicit GPU
handoff. These are separate invocations, not an automatic chained workflow.

```python
import json
from pathlib import Path
import subprocess

phase = "verify"  # Use "evaluate" only after the GPU handoff.
root = Path("/home/horde/Mamba2-8B-E8W5")
python = "/home/horde/.venvs/lodram/bin/python"
train_path = root / "reports/norm_compensation_v1_train.json"
train = json.loads(train_path.read_text())
assert train["complete"] and train["final_step"] == 128
assert train["successful_updates"] == 128
checkpoint = Path(train["final_checkpoint"]["file"])
assert checkpoint.parent.resolve() == (root / "artifacts/norm_compensation_v1_train_work").resolve()
assert checkpoint.is_file()
assert (root / "artifacts/norm_compensation_v1/manifest.json").is_file()

command = [
    python, "scripts/evaluate_norm_compensation.py",
    "--source-dir", "models/source",
    "--parent-dir", "artifacts/e8w5_v1",
    "--overlay-dir", "artifacts/norm_compensation_v1",
    "--calibration-manifest", "calibration/v1/manifest.json",
    "--calibration-tokens", "calibration/v1/calibration_tokens.pt",
    "--protocol", "docs/NORM_COMPENSATION_PROTOCOL.md",
    "--training-report", str(train_path),
    "--training-checkpoint", str(checkpoint),
    "--smoke-report", "reports/norm_compensation_v1_smoke.json",
]
if phase == "verify":
    command += ["--report", "reports/norm_compensation_v1_integrity.json", "--verify-only"]
elif phase == "evaluate":
    command += ["--report", "reports/norm_compensation_v1_eval.json", "--full-validation-if-improved"]
else:
    raise ValueError(phase)
subprocess.run(command, cwd=root, check=True)
```

The evaluator refuses to overwrite an existing output report. Preserve a
failure and diagnose it before choosing another explicit report name; do not
rerun or modify the frozen training recipe automatically.

## Expected stages

CPU verification checks the complete parent and overlay inventory, every file
hash, all 393 FP16 tensors, the final checkpoint masters rounded to FP16,
the deterministic training schedule and exposure accounting, and the successful
smoke and training receipts. It does not allocate a model on the GPU.

After GPU handoff, evaluation first audits all 507 loaded parent tensors. It
then compares the original parent and reloaded norm candidate on the fixed four
development windows and all 24 public MK cases. A failed development gate stops
the route and skips full validation.

Only a passed development gate allows complete validation of candidate,
parent and original source in the same process. All three use the same
264,764 validation targets. The source checkpoint is loaded solely as the
declared reference at this stage. No test split is used.

Report the parent-relative improvement gate and source-relative +5% PPL target
separately. No result in this workflow authorizes publication, and the small MK
development set does not establish full recall preservation.
