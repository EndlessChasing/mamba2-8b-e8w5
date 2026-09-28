# Paired native projection statistics

Implemented in `mamba_e8w5/projection_statistics.py`. This module collects
statistics and scores local replacements. It does not fit, quantize, install,
train, select, publish or evaluate the language quality of a replacement model.

## API and normalization

```python
fit = collect_projection_statistics(
    teacher, candidate, fit_windows, selected_names=DEFAULT_SIX,
    split_label="fresh_train_fit",
)
proxy = score_dense_from_statistics(weight, fit["matrices"][label])
native = score_native_fp16(
    teacher, candidate, heldout_windows,
    {label: already_rounded_fp16_weight}, split_label="fresh_train_heldout",
)
```

Defaults are layers 0, 18 and 55, each `in_proj` and `out_proj`. Call separately
for fit and heldout positions. A tensor shaped `[windows, length]` is processed
one window at a time; a list of `[batch, length]` tensors permits batching.
Each batch starts a fresh native backbone forward without an inference cache.
The caller owns token selection, disjointness, model identity and provenance.

For matching token rows `Xq[N,n]` and actual teacher output `Yt[N,m]`, each
`matrices[label]` contains normalized uncentered moments:

- `H[n,n] = Xq.T @ Xq / N`, FP32 on the candidate device.
- `K[m,n] = Yt.T @ Xq / N`, FP32 on the candidate device.
- `teacher_energy = sum(Yt**2) / N`, sum over output coordinates.
- `tokens=N`, weight shape, native dtypes, input/output energies and H/K norms.

Teacher targets are the outputs of native `nn.Linear` calls. Replacing them
with `Wt @ (Xt.T @ Xq / N)` would omit native FP16 output rounding. The CPU
rounding test demonstrates that difference. No centering or intercept is added.

`score_dense_from_statistics` evaluates
`sum((W @ H) * W) - 2 * sum(W * K) + teacher_energy`, using FP64 arithmetic by
default on FP32 moments. It is the real-linear prediction error, without native
FP16 GEMM output rounding. FP64 scoring cannot remove FP32 accumulation error;
small negative cancellation is explicitly reported, with a bounded display
clamp, and materially negative/nonfinite results fail.

Native scoring instead invokes FP16 `F.linear(Xq, replacement)` while the
candidate continues using its original output. Every replacement therefore
sees the unchanged candidate prefix, even when several are scored in one call.
This does not measure a model with all replacements installed together.

Native receipts are JSON serializable. Metric conventions are explicit:

- `mse_per_token`: mean over tokens, sum over output coordinates.
- `mean_output_squared_error` / `mse_per_output_element`: mean over both axes.
- `relative_output_mse` / `relative_mse`: total error energy / teacher energy.
- `relative_reduction_vs_baseline`: reduction from the unchanged candidate's
  native error on the same matched inputs and teacher targets.

For the 18560-row `in_proj`, the native receipt also reports z `[0,8192)`,
x `[8192,16384)`, B `[16384,17408)`, C `[17408,18432)` and dt `[18432,18560)`.
The partition means use each partition's own coordinate count and energy.

## Runtime guards and memory

Both models must be in evaluation mode on one device, with matching bias-free
Linear shapes. Default operation requires actual FP16 inputs, weights and
outputs. The selected mixers must disable `use_mem_eff_path`; otherwise native
Linear hooks may be bypassed. CUDA callers must disable TF32, select `highest`
FP32 matmul precision, and run without autocast. Model parameters must have
normal version counters; create/load models outside `torch.inference_mode()`.
Ordinary `torch.no_grad()` is supported.

The collector requires one hook call per selected projection and batch, finite
inputs/outputs, equal token axes, complete pairing, and unchanged selected
parameter identities/versions. Hooks and cached activations are removed in a
`finally` block. Identity/version checks are not a substitute for the pilot's
independent complete tensor-content hash audit.

Only the current batch's teacher outputs are retained; student inputs are
consumed inside their own hooks. For the six default projections:

| FP32 moment storage | Per matrix | Six-matrix total |
|---|---:|---:|
| in_proj H + K | 354 MiB | 1062 MiB for three |
| out_proj H + K | 384 MiB | 1152 MiB for three |
| Combined | | 2214 MiB / 2.162 GiB |

A separate fit and heldout collection doubles moment storage to 4.324 GiB.
At batch 1 and 2048 tokens, the six native teacher outputs occupy at most
265.5 MiB. Temporary activations, matmul/cast workspaces, FP64 scoring and model
runtime memory are additional. Two full FP16 weight sets use about 32.95 GB
(30.69 GiB). This bounded pilot fits the 96 GB device; actual peak memory still
must be measured. Do not extend simultaneous storage to all 112 projections:
their H/K would occupy about 40.36 GiB before model/workspace costs and exceed
the currently available disk if persisted.

## CPU verification

Command, run on the GPU host with CUDA explicitly hidden:

```sh
CUDA_VISIBLE_DEVICES="" /home/horde/.venvs/lodram/bin/python -m unittest \
  tests.test_projection_statistics tests.test_projection_repair -v
```

Result: 12/12 passed, exit 0. Seven collector tests cover direct native-target
moments/orientation/counts, independent fit/heldout calls, FP16 target rounding,
the real-linear quadratic, independent native replacements, model identity,
role partitions, JSON serialization and hook cleanup on rejected inputs.
The other five tests cover the parent's anchored ridge/window-selection helper.
This establishes CPU algebra and interface behavior; it does not establish
real-model GPU correctness or any quality improvement. The pilot driver has a
separate discarded 32-token native pairing check before collecting fit data.

Frozen collector SHA256:
`3247ef752491a0542210e9b3058ffa2149d3485c56a4aca851d67edd971f595f`.
Machine-readable test receipt: `reports/projection_statistics_cpu.json`.
