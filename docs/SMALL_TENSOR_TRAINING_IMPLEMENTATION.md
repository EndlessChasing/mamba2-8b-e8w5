# All-small-tensor training implementation

This implementation follows `SMALL_TENSOR_COMPENSATION_PROTOCOL.md`. Quality
results are separate; implementing or passing a smoke does not establish PPL
recovery or recall preservation. No old norm-v1 code or receipts are modified.

The new helper maps all seven parameter families in every block: block norm,
gated norm, convolution weight/bias, dt_bias, A_log and D, plus the final norm.
There are 393 FP32 master tensors with 3,580,928 values. Every native block call
and checkpoint replay casts the full mapping to FP16. Native convolution,
FP32 exponential of FP16-rounded A_log, softplus and skip computations remain
unchanged. The 114 E8/W5 parameter objects have no gradient and their tensor
bytes are hashed before and after the run.

The original norm-v1 helper supplies unchanged full-vocabulary staged CE/KL,
FP16 export/readback parity and gradient-coverage checks. The old trainer supplies
unchanged optimizer/scaler constructors and atomic/checksum utilities. Their
source hashes are included in the new experiment binding. The new trainer uses
a fresh optimizer and scaler, initialized from the actual norm-v1 FP16 export.

`prepare_small_training_data.py` reads only the pinned train split. Its 2048-token
grid excludes any overlap with the original32 calibration windows, then selects
256 entries by the fixed index formula. The saved input contains524,288 stored
positions;524,032 positions are passed to the model per pass and the same number
are scored as targets. Four passes score2,096,128 target exposures. No Hessians
or vocabulary/projection quantization files are changed.

The GPU smoke checks native initialization, changed-master checkpoint replay,
all393 gradient tensors and all seven gradient families, full-vocabulary chunk
equivalence, a full2047-target trial update, and a changed FP16 export. It records
parameter/gradient ranges, negative finite effective A, and positive finite
bias-only softplus diagnostics. These do not measure token-dependent recurrence
or long-context recall. Numerical tolerances are the frozen norm-v1 smoke
tolerances; actual differences are recorded.

Training saves durable optimizer checkpoints every32 successful updates, on
clean termination and at final1024. A per-attempt journal is flushed to disk.
SIGTERM/SIGINT request a stop after the current attempt, allowing an exact
checkpoint. A hard kill can leave uncheckpointed activity: a resumed invocation
rejects any journal start/completion absent from the selected checkpoint. It
does not replay such work or silently expand the exposure budget. Numerical
failures create a nonresumable diagnostic snapshot and cannot resume. Prior
reports and process records are retained.

The final export is verified in a private staging directory before being moved
to the candidate directory. A complete candidate requires checkpoint1024's
393 FP32 masters to round bitwise to its393 FP16 tensors, exact inherited file
hashes and unchanged507-parameter architecture. Inference resolves the original
E8/W5 files plus one new small-tensor payload; the initialization model and
training files are provenance inputs rather than hidden inference dependencies.

GPU jobs are coordinated separately. The trainer never evaluates validation,
test or MK; the final reloaded candidate proceeds to independent full-validation
PPL evaluation under the declared protocol. No publication is authorized.
