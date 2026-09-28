# Existing-metadata training: read-only design audit

**Status: proposal, unimplemented and unmeasured.** This note authorizes no training,
GPU experiment, new candidate, or publication. The eight-sweep refinement remains
the current experiment. No frozen runtime, codec, evaluator, or weights were changed.

## Proposed ordering and representational limit

**The simpler first training baseline is norm-only fitting of the existing FP16
norm tensors**, described below. It needs no custom decoder backward or FP32
numerator cache. A second candidate specific to E8 metadata is to **freeze indices,
signs and the existing FP32 scale, and train the 688,128 existing FP16 input-balance
values**. Cache the exact FP32
decoder numerator for training, and round the balance and decoded weights exactly
as the deployed decoder does. The cache and optimizer are training workspaces;
only replacement balance bytes would be exported. This adds no weight payload
fields or inference adapter. Changed metadata may change an entropy-coded archive's
size, and new provenance metadata is not free.

This changes a column gain, shared across every output row of each projection.
It cannot independently repair individual E8 codeword errors or rotate a column's
output direction. The available function family is much smaller than retraining
the 6.136B projection weights. Language quality improvement is unknown for either
proposal. The wider all-small-tensor variant should remain a separate comparison.

## Simpler alternative: retained FP16 norms or small tensors

Exact counts below come from the current 507-tensor
[source inventory](../reports/source_tensor_inventory.json), excluding the 112
E8 projections and the two W5 vocabularies.

| Existing tensor family | Tensors | Parameters |
|---|---:|---:|
| Block `norm.weight`, 4096 per layer | 56 | 229,376 |
| Grouped gated `mixer.norm.weight`, 8192 per layer | 56 | 458,752 |
| Final `backbone.norm_f.weight` | 1 | 4,096 |
| **Norm-only total** | **113** | **692,224** |
| `conv1d.weight` | 56 | 2,293,760 |
| `conv1d.bias` | 56 | 573,440 |
| `dt_bias`, `A_log`, `D`, 128 each per layer | 168 | 21,504 |
| **All retained small tensors** | **393** | **3,580,928** |

The norm-only tensor payload is 1,384,448 bytes; the complete small-tensor payload
is 7,161,856 bytes, already present in `other_fp16.pt`. FP32 masters, FP32 accumulated
leaf gradients and two FP32 Adam moments require approximately 11,075,584 bytes
(norms) or 57,294,848 bytes (all small tensors), plus cast copies and activations.
Teacher plus decoded student still occupy 32,947,998,720 bytes (30.685 GiB),
without the balance proposal's additional 24.545GB numerator cache.

Keep E8/W5 and all unselected tensors frozen. Create FP32 master `Parameter`s
only for the selected existing tensors, initialized from their decoded FP16
values. For each forward, provide `master.to(torch.float16)` through
`torch.func.functional_call`; use a complete parameter mapping with `strict=True`
and preserve the untied vocabulary configuration. Calling the backbone separately
allows the frozen head and CE/KD loss to be processed in token chunks. This is an
ordinary functional call plus ordinary autograd, not a `vmap`/`torch.func.grad`
transform over the Mamba custom autograd functions.

The forward then consumes the same FP16 parameter values that would be serialized.
Gradients return through the cast to FP32 masters and FP32 optimizer accumulation.
This does **not** make every local derivative FP32: the installed gated norm
backward reduces internally in FP32 but casts `dw` to its FP16 input weight dtype
before returning it. Loss scaling and finite/underflow checks would still need
validation. Do not put this student forward under inference/no-grad mode.

With activation checkpointing, each recompute closure must re-enter
`functional_call` with the same rounded master values. Merely wrapping the outer
forward in a temporary functional parameter swap can leave checkpointed child
modules reading restored, stale parameters during backward recomputation. Freeze
master values until all accumulated backward work has completed.

After a future authorized run, serialize `master.detach().half()` into a new,
parent-bound `other_fp16.pt`, retaining all 393 expected keys, shapes and FP16
dtypes. Verify changed/unchanged tensor coverage and reload/export forward parity.
E8 and W5 file hashes must remain exact. The *tensor payload capacity* is unchanged;
measure actual `torch.save` archive bytes and metadata rather than assume identical
ZIP overhead. This needs a different experimental overlay schema from the current
112-E8-file refinement helper; do not relax that helper to accept it silently.

Norm-only is the narrower initial comparison and the simpler implementation.
It can still change gates and recall. Training all small tensors also directly
changes recurrent time constants, skip paths and convolution, increasing the
potential to change MK behavior even when prose CE improves. Neither scope is
evidence of preserved recall; both would require a predeclared train-only CE/KD
objective and separate exported PPL/MK checks.

## Exact decoder algebra and the scale ambiguity

For a projection with `m` outputs and `n` inputs, let `Q` be the fixed E8 codebook
matrix, `s` its FP32 scalar, `b` its positive FP16 balance vector, and `S_u/S_v`
the fixed sign diagonals. Define `R_d` by `rotate_last(x) = x R_d` in exact
arithmetic: `R_d = DCT.T ⊗ normalized_Hadamard`. The ideal decoded weight is

`W = s S_v R_m Q R_n.T S_u diag(b)^(-1)`.

The actual implementation is more specific. [codec.py](../mamba_e8w5/codec.py)
lines 175–183 multiply the codebook values by `s` **before** the two inverse
rotations, execute those operations and signs in FP32, divide by FP32 values
converted from stored FP16 `b`, and finally cast the entire result to FP16.
Lines 199–202 store precisely two bytes per balance and four bytes per scale.

Denote the exact FP32 tensor immediately before the division by `C(s)`. With
`s=s0` frozen, cache `C0=C(s0)` using those same operations, dtype and arithmetic
settings. The proposed deployed forward is exactly

`b_q = FP16(b_shadow)`; `U = FP32(C0 / FP32(b_q))`; `W_q = FP16(U)`;
`Y = F.linear(X, W_q)`.

Use the existing full FP16 linear operation, shapes and layouts. Do not replace
it with runtime input scaling `linear(X/b, C0)` or with multiplication of the old
FP16 decoded weights. These are real-arithmetic identities with different FP16
rounding points, so their training forward can disagree with exported weights.
Caching `C0` in FP16 would also lose the exact numerator.

Learning `s` is possible only if each changed scale reconstructs `C(s)` in the
original operation order for the exact forward. `C(s0)*(s/s0)` is not guaranteed
bitwise equal to recomputing the FP32 inverse rotations on `Q*s`. An ideal-linear
STE may be used in backward, but cannot justify changing the forward arithmetic.

There are 112 continuous gauge directions: in real arithmetic `(s,b) → (a*s,a*b)`
leaves `W` unchanged for each projection. Rounding weakly breaks this symmetry;
it does not make simultaneous scale/balance fitting well-conditioned. This is
not a fatal obstacle to training the function, but the redundant parameterization
should be removed. Freezing `s` removes it. Alternatively, constrain each vector's
mean log-balance and train its scalar; that version needs a new exact-forward
implementation. A global gain remains available with fixed `s` by changing all
of its balances together, at FP16 balance resolution.

## Custom exact forward, surrogate backward without weight gradients

Use a custom autograd linear operation taking `X`, an FP32 balance shadow, and
frozen `C0`. Its forward casts the shadow to FP16 and materializes `W_q` as above
under no weight autograd graph. Save `X` and the rounded balance; frozen numerator
buffers can be referenced. The optimizer must not change metadata between forward
and its checkpoint/backward recomputation.

For flattened tokens `X[T,n]` and upstream derivative `G[T,m]`, use:

`dX = G @ W_q`;
`db_shadow[j] ≈ -(1/b_q[j]) * sum_t X[t,j] * (G @ U)[t,j]`.

The second expression is the chain rule for `U=C0/b_q`, using identity STEs for
both FP16 rounding operations; it is **not** the mathematical derivative of a
piecewise-constant quantizer. Compute its contractions/reductions in FP32 and
optionally accumulate reductions in FP64. Process input-column tiles to bound
the temporary `U` and `G @ U`; never form `G.T @ X` as a dense weight gradient.
Return only activation and metadata gradients, with no gradient for `C0`.
`dX` uses the actual rounded forward weight, not `U`.

An FP32 shadow allows sub-ULP updates to accumulate before they change FP16
metadata. Enforce positive, finite, representable balances and reject zero or
overflow after casting. Direct positive shadow parameters with a predeclared
projection/clamp are simpler to audit than a log reparameterization. If using
`b_shadow=exp(theta)`, multiply the displayed `db_shadow` by `b_shadow`, not by
`b_q`; they can differ after rounding. Clamp derivatives need explicit treatment.

If scales were also trained, their ideal-linear STE is
`ds ≈ sum_tj X[t,j]*(G@U)[t,j]/s`, while the forward must still reconstruct
`C(s)` exactly. This is a separate, less attractive first experiment.

## Parameters and memory budget

The 56 `in_proj` matrices are `[18560,4096]`; their balance length is 4096.
The 56 `out_proj` matrices are `[4096,8192]`; their balance length is 8192.
Counts are 229,376 + 458,752 = **688,128 balances** (0.008354% of all parameters).
Adding 112 scales gives 688,240 scalar coordinates before removing the 112 gauge
directions. Existing balances occupy 1,376,256 bytes and scales 448 bytes.

| Resident data / workspace | Bytes | GiB |
|---|---:|---:|
| Original teacher, all 8,236,999,680 weights in FP16 | 16,473,999,360 | 15.343 |
| Candidate decoded weights in FP16 | 16,473,999,360 | 15.343 |
| All frozen projection numerators in FP32 | 24,545,067,008 | 22.859 |
| Those three together | 57,493,065,728 | 53.545 |
| Balance FP32 masters + gradients + two Adam moments | 11,010,048 | 0.0103 |
| Largest one-projection FP32 temporary | 304,087,040 | 0.2832 |
| One hidden tensor per boundary, 57×2048×4096 in FP16 | 956,301,312 | 0.8906 |

These are tensor arithmetic budgets, **not measured training peaks**. Block
checkpointing must also preserve residual inputs; the boundary row is only a
lower-bound illustration. Scan/norm/conv intermediates, activation gradients,
allocator reserve, CUDA workspaces and cached rotation factors are additional.
For batch 1/context 2048/chunk 128, one FP32 chunk-state tensor is about 64 MiB;
the scan backward creates multiple such workspaces. The installed backward
recomputes chunk states rather than saving every recurrent time-step state.

A 96GB GPU appears feasible with microbatch 1, block checkpointing, no weight
gradients, and token-chunked loss. It has not been demonstrated. Avoid caching all
FP32 unrounded weights in addition to the numerator: that would add another
22.859 GiB unnecessarily. A memory-saving version may discard persistent candidate
projection weights and rematerialize one projection at a time, but would require
separate forward-parity checks.

The 256K vocabulary produces 1.953 GiB of FP32 logits for 2048 tokens per model.
Use teacher no-grad hidden states and 64-token head/loss chunks (62.5 MiB of FP32
logits per chunk). For distillation, accumulate the student final-hidden gradient
per chunk and backpropagate through the student backbone once, rather than retain
all full-vocabulary chunk graphs. Teacher and student vocabularies remain separate
and frozen. A combination of next-token CE and original-teacher distillation is a
proposal; its mixture, temperature, train-token budget and stop rule are undeclared.

The host reports 65,840,424 KiB total RAM (~62.79 GiB). Keep numerator buffers on
GPU, mmap the original checkpoint, and decode one projection at a time. Do not
retain full CPU teacher, student, numerator and FP32 weight copies together.
Teacher weights can be resident on GPU with no saved teacher autograd activations.

## Installed backward support and required limits

Read-only inspection of `/home/horde/.venvs/lodram` found PyTorch 2.11.0+cu128 and
`mamba-ssm` 2.3.2.post1. `MambaChunkScanCombinedFn.backward` exists in
`ops/triton/ssd_combined.py:620` and returns activation/SSM-input gradients;
`return_varlen_states=True` explicitly lacks backward. Fixed-length prefill with
that flag false is the applicable path. Grouped gated RMSNorm has a backward at
`ops/triton/layernorm_gated.py:369`. `causal_conv1d` is not installed; the current
Mamba2 source falls back to ordinary PyTorch `Conv1d` + SiLU, which has autograd.

Keep `use_mem_eff_path=False`: the frozen model calls the projection modules
normally on that path, allowing custom metadata linear wrappers. The fused path
reads projection weights directly and could bypass such wrappers. Do not use
the inference-cache token-step path for this training design. Keep all original
weights frozen and enable gradients only for metadata shadows; the frozen
evaluation helpers run under inference mode and cannot serve as training loops.
Training uses SSD prefill and would not establish tokenwise FP16 cache quality.

Source presence establishes an implementation route, **not** successful backward
on this GPU/model. Before any future training, a separately authorized prototype
would need exact serialized-forward parity, a small analytic/STE-gradient oracle,
a native-block finite backward check, and measured peak memory. Any trained
candidate would require fresh payload/hash receipts and independent exported
PPL/MK evaluation, including the tokenwise check. All such work remains unstarted.

### Audited source identities

- Frozen codec: `ece47122f23dd0a726b45fba0b79466134534f011d560f8473c74ae9494aa79f`.
- Frozen runtime: `bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369`.
- Source tensor inventory: `57fe43d21c5dd5fb781815a7bca7bbcc1b795f6de723ce449fddd4bbdbd721f6`.
- Installed `mamba2.py`: `605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707`.
- Installed `ssd_combined.py`: `0b7c4cfa9e994278860e7254c27baa31ca32019a808a423ed24b80eabcbe919d`.
- Installed `layernorm_gated.py`: `eb6252e247b90f1c8a75946efbc1a221e0c4da701b6757ddae49f3495cf7a42f`.
