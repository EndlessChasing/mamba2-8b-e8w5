# E8 balance fitting versus existing norm gains

This is a read-only mathematical audit while the fixed all-small-tensor run is
in progress. It changes no experiment, tensor, protocol or decoder. It explains
why balance-only fitting should not be counted as a substantially new source of
continuous compensation after training the existing norms.

## Exact-arithmetic redundancy

For each E8 projection with fixed indices, signs, rotation and scalar scale,
the decoder has a fixed numerator C and weight

`W(b) = C diag(1 / b)`.

Its immediate normalization produces `diag(g) r(x)`, where r includes the
normalization denominator and any gating but excludes the learned gain g.
The projection output is therefore

`C diag(g / b) r(x)`.

For any positive new balance b', the old balance b produces the same output
with **g' = g * b / b'**, channel by channel. Thus only the ratio g/b is a
continuous degree of freedom when the numerator and all other inputs are fixed.

This applies to both projection families in the actual pure Mamba-2 runtime:

- `in_proj`: the preceding block RMSNorm supplies the 4096 channel gains.
- `out_proj`: grouped gated RMSNorm supplies the 8192 channel gains. The current
  `norm_before_gate=False` applies the gate before group normalization and the
  affine gain after the denominator, so folding into the gain does not cross
  normalization, its epsilon, or the nonlinear gate.

There are 56*(4096+8192) = **688,128** balances, matching exactly those existing
block and gated norm gains. The final 4096 norm gains are additional existing
parameters in the norm experiment. Positive balance constraints also cannot
change gain signs, whereas unconstrained norm fitting can do so.

Keeping a projection output identical keeps its subsequent gate/convolution/SSM
inputs identical. The pre-normalization residual bypass is unchanged; it does
not invalidate this local equivalence. The out-projection balance/gain change
acts after recurrent-state formation. With the above exact matching, recurrence
and residual paths therefore do not create an additional function class.

## Why this is not bitwise equivalence in FP16

The actual decoder computes C/b in FP32 and rounds the weight to FP16. Norm
masters round to FP16 and native normalization also rounds its output. Generally,

`R16(C/b') @ R16(g*r(x)) != R16(C/b) @ R16(g'*r(x))`.

Balance fitting can change individual decoded-weight rounding thresholds and
effective activation scaling/conditioning. These finite-precision and optimizer
behavior differences may affect PPL, but do not imply an arbitrary new correction
to each E8 quantization error. Fitting balance and norm together also introduces
a continuous ratio ambiguity in the idealized arithmetic.

Reassigning E8 indices or modifying a rotation changes C and is outside this
redundancy argument. Those would be separate experiments with new correctness
checks and measured quality. No such experiment has been run by this audit.

## Practical decision

Prefer finishing the current training of all existing small tensors, which
adds convolution and recurrent-parameter adjustments beyond norm gains. Do not
spend the proposed additional 24.545 GB FP32 numerator cache on balance-only
training on the assumption that it adds 688,128 independent compensation modes.
The earlier [metadata design](METADATA_TRAINING_DESIGN.md) remains historical
implementation analysis; this note narrows its expected representational benefit.

Evidence: fixed decoder in [codec.py](../mamba_e8w5/codec.py), the exact current
runtime [MODEL_CONFIG and source mapping](../mamba_e8w5/runtime.py), and the
[source tensor inventory](../reports/source_tensor_inventory.json). This is algebra
and source inspection, not a new PPL or hardware measurement.
