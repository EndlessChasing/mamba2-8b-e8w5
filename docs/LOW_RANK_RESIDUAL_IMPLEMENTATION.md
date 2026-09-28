# Low-rank factor representation: CPU preparation

This isolated infrastructure does not select or run a low-rank model experiment.
The prototype training/evaluation remains authoritative. No frozen decoder,
runtime, trainer or protocol is changed.

## API and format

`mamba_e8w5.low_rank_residual` exports:

- `factor_layout(m, n, rank)`: validate geometry and compute exact capacity.
- `validate_factors(B, A)`: require finite, strided CPU FP16 matrices with
  shapes `[m,r]` and `[r,n]`; no implicit dtype/device conversion.
- `write_factors(path, B, A)`: deterministic exclusive publication, verified
  disk readback, and SHA256/byte receipts. Existing files and failed `.partial`
  files are preserved.
- `read_factors(path, expected_shape=None, expected_rank=None)`: independent
  CPU FP16 factor tensors. Future model loading should provide expected shape
  and rank from an independently verified manifest.
- `merge_weight(base, B, A)`: the exact native merge described below.

Version 1 has a **32-byte** little-endian `<8sHHIIIQ` header:

| Offset | Field | Encoding / valid value |
|---:|---|---|
| 0 | Magic | 8 bytes, `ME8LR001` |
| 8 | Version | uint16, 1 |
| 10 | Dtype | uint16, 1 = IEEE FP16 |
| 12 | Output dimension m | uint32 |
| 16 | Input dimension n | uint32 |
| 20 | Rank r | uint32 |
| 24 | Payload bytes | uint64, exactly `2r(m+n)` |

The payload is row-major little-endian `B[m,r]`, then `A[r,n]`. There is no
padding, bias, scaling field or extra data. Signed zeros and all other finite
FP16 bits are retained by serialization. NaNs and infinities are rejected.
Hash receipts do not replace a future authenticated or independently pinned
model manifest; the standalone file has no embedded checksum.

Header geometry and actual file length are checked before reading the payload
or allocating arrays. Limits are 65,536 per dimension, rank at most 256 and
`min(m,n)`, 100,000,000 merged matrix elements, and 32 MiB of factor payload.
These cover the current projection shapes and ranks 4/8/16. Oversize, truncated,
inconsistent or trailing-byte files fail. Nonregular files fail; Unix platforms
with `O_NOFOLLOW` reject symlinks. These limits bound resource requests, without
claiming full-size merge feasibility on every CPU host.

## Native arithmetic and gradient boundary

The operation is exactly:

```python
merged = (base.detach().float() + B16.float() @ A16.float()).half()
```

Factors supplied to this function must already be FP16. A future FP32 master
caller must execute `master.half()` **inside every checkpoint replay**, then
pass these rounded tensors. The factor product and addition are FP32; there is
one final weight cast to FP16. CPU autocast is rejected because it can change
the stated FP32 matrix product. Ordinary cast/matmul autograd is used, without
a custom derivative or gradient through the frozen base/decoder.

The base is never written or replaced. The returned weight has separate
storage. There is no zero-factor shortcut: that would suppress useful initial
factor gradients. A zero correction preserves finite base values, but native
addition can change a **negative-zero bit to positive zero**. The helper follows
that arithmetic instead of promising arbitrary bitwise preservation. Both
factor matrices initialized to zero produce zero gradients; a zero B with a
nonzero A retains the expected initial B gradient.

This materialized weight is not interchangeable with a two-linear residual
branch: its intermediate rounding differs. Rounding can also make the effective
weight difference higher rank than the factor product. This helper only accepts
CPU tensors and changes no global precision flags. A future CUDA implementation
must explicitly enforce TF32/autocast requirements and test its own parity;
the CPU tests establish no CUDA bitwise equivalence or performance result.

## CPU verification

`tests/test_low_rank_residual.py` checks a manually packed file, deterministic
strided-input serialization, signed-zero bytes, malformed geometry/dtype/length,
preallocation rejection, nonfinite payloads, exclusive writes and resource
bounds. Hand-computed dyadic products establish the merge orientation. A
separate analytic zero-B derivative verifies the gradient path. A tiny CPU
linear model checks checkpoint/direct output and gradient equality with FP32
masters rounded inside replay, while frozen-base bytes/version/grad remain
unchanged. FP16 overflow and CPU autocast are rejected.

Run from the repository with CUDA devices hidden:

```sh
CUDA_VISIBLE_DEVICES='' /home/horde/.venvs/lodram/bin/python -m unittest tests.test_low_rank_residual -v
```

The final 11-test CPU run passed in **0.908 seconds**, exit code 0, including
the explicit 32-MiB allocation boundary. There was no GPU work, model scoring,
optimizer experiment, trainer CLI or new training recipe.
The [CPU receipt](../reports/low_rank_residual_cpu.json) preserves the actual
command, combined test output, final file hashes and a separate environment
probe using the same interpreter with CUDA devices hidden.
