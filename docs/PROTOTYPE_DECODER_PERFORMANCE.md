# Prototype decoder performance probe

**Decision: keep the frozen decoder for the fixed128-update experiment.**
The explicit2D GEMM alternative was faster but failed exact FP16 weight parity.
It was not adopted. No custom adjoint, transform change or decoder caching was
introduced into the bound training code.

## Scope and reproducibility

Probe: `scripts/probe_prototype_decoder_performance.py`.
Receipt: `reports/prototype_decoder_performance_v1.json`.

- Script SHA256: `cf46809310688f6fd71cf2e4aae754d86efca8c5b5a4973e8aab8c5d8856bdc8`.
- Receipt SHA256: `93e0d7c6e11654145fc7699977f6ce8f75a9f40a667ac9e17e6044a534d48436`.
- Device: RTX PRO6000 Blackwell Server Edition; Torch2.11.0+cu128; TF32 disabled.
- Exactly24 trials: layer0 in/out, zero/nonzero tables, frozen/explicit-MM,
  first invocation plus two warm repetitions. Payloads remained on CPU.
- Cotangent was a fixed dense FP16 pattern. There was no language model,
  optimizer, training update, model-quality evaluation or candidate write.
- Total probe wall time:28.106seconds. Frozen files and codebook hashes were
  unchanged; the process-local rotation monkeypatch was restored.

The optional CUDA-payload branch was not run. First invocation is not globally
cold: factor caches persist across cases. Timers exclude output hashing and
parity checks. Wall timings include decoder validation and payload copies;
CUDA-event spans can include GPU idle time awaiting host dispatch.

## Measured warm times

Means of the two warm repetitions, zero-table case:

| Matrix and implementation | Forward seconds | Backward seconds | Total seconds |
|---|---:|---:|---:|
| in_proj, frozen |0.208391|0.818580|1.026971|
| in_proj, explicit2D GEMM |0.048726|0.019671|0.068397|
| out_proj, frozen |0.243962|1.075386|1.319348|
| out_proj, explicit2D GEMM |0.027830|0.017981|0.045811|

The nonzero-table cases had similar times. Frozen repetitions reproduced their
FP16 output and FP32 table-gradient hashes exactly on these inputs.

Metadata-only tracing of the frozen decoder showed large Hadamard products
using batched single-row matrix multiplication because of transposed strides.
The relevant shapes include `[18560,1,4096]`, `[8192,1,4096]` and
`[4096,1,8192]`, each multiplying a broadcast Hadamard matrix. Backward repeats
these products. The explicit-MM probe flattens these into large matrix GEMMs
and also packs the DCT multiply into2D form. Although algebraically equivalent,
this changes FP32 accumulation kernels.

The frozen warm measurements predict approximately
`56 * (2 * (in_forward + out_forward) + in_backward + out_backward)`
= **156.726seconds** for two decoder forwards plus one decoder backward per
block. The full2047-target smoke update measured **154.609seconds**. This close
agreement strongly explains the slowdown as repeated decoder work; it is an
extrapolation from two matrices, not a full-model profiler attribution. It does
not support treating the154-second update as only first-use compilation.

At that measured update time,128 updates alone take approximately5.50hours;
loading, overflow retries, checkpointing and export checks are additional.

## Exactness result: alternative rejected

The table shows one warm repetition; all explicit-MM repetitions failed FP16
bitwise parity. Signed-zero differences also count as different FP16 bits.

| Matrix | Table | FP16 elements with different bits | Total elements | Maximum absolute weight difference |
|---|---|---:|---:|---:|
| in_proj | zero |364049|76021760|0.0001220703125|
| in_proj | nonzero |364534|76021760|0.00006103515625|
| out_proj | zero |361087|33554432|0.00006103515625|
| out_proj | nonzero |360446|33554432|0.00006103515625|

Relative L2 table-gradient differences were about2.35e-5 for in_proj and2.62e-5
for out_proj. They passed the probe's explicitly diagnostic allclose check,
but that does not override the failed weight-bit criterion. There is no PPL or
recall result for the explicit-MM alternative, and no same-forward claim.

The original frozen forward/backward implementation remains the basis of the
predeclared training run and independent serialized-model evaluator.
