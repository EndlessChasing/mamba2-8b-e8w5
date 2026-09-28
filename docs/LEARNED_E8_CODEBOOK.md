# Learned E8-derived codebook foundation

`mamba_e8w5/learned_e8_codebook.py` adds a representation primitive without
changing any frozen codec/runtime file or existing model artifact.

Each projection may add one256x8 FP16 correction table. For uint16 code `c`,
the prototype is `c >> 8`; the original low-eight-bit sign/parity mapping and
fixed coordinate shuffle are retained. The expanded grid equals
`base.grid[c,j] + sigma(c,j) * delta[c >> 8,j]`. The effective sign includes
the original packed prototype's signed last coordinate. The original quarter
offset is unchanged; parity is never recalculated from learned magnitudes.
Arbitrary corrections form an E8-derived codebook, not a strict E8 lattice.

## API and bytes

- `prototype_mapping(cb, device)` returns prototype IDs and signed directions.
- `corrected_grid(cb, delta, device)` requires finite FP16[256,8].
- `decode_e8_with_prototypes(payload, cb, delta, device)` calls the frozen
  inverse decoder with the corrected grid, preserving order and FP16 rounding.
- `write_prototypes(path, delta)` and `read_prototypes(path, device)` use a
  strict32-byte little-endian header plus4096 table bytes: **4128 bytes/file**.
- `corrected_grid_from_rounded_fp32` is an optional pure-tensor adjoint
  primitive. Its FP32 values must already be exactly FP16 representable; it
  never silently relaxes the file/inference representation.

112 separate files add462,336 bytes, including458,752 table bytes. Manifests,
software and any future entropy-container overhead are separate. No quality,
training success, compressed residency or compression improvement follows from
the foundation alone. The declared prototype training protocol uses ordinary
FP16 casts, not the optional FP32 adjoint primitive.

## Completed CPU checks

The original seven-test run passed on `horde-gpu` with CUDA disabled, in10.150s:

1. Every65,536 zero-correction codeword matches the pinned upstream grid bitwise.
2. Every65,536 nonzero codeword matches an independent scalar packed decoder.
3. Frozen inverse-transform ordering and finalFP16 results match at zero/nonzero.
4. FP32 adjoint primitive keeps exact forward values and prototype/sign gradients.
5. Invalid table geometry/dtype/nonfinite values and mutated base books fail.
6. Invalid payload indices/signs/balance/scales/residual formats fail.
7. Exact4128-byte serialization preserves allFP16 bits, including signed zero;
   malformed headers, wrong lengths and nonfinite stored values fail.

This documents the completed run; it does not represent a rerun. Exact source
hashes and command are preserved in `reports/learned_e8_codebook_cpu_tests.json`.

A separate one-case CPU check subsequently passed in9.633s without rerunning
the original seven. It exercises repeated codes and different low bits sharing
a prototype through the actual `corrected_grid(cb, master.half())` path. At zero
and nonzero tables, all256x8 master gradients exactly equal a manually computed
signed sum with the actualFP16 backward rounding. Its distinct receipt is
`reports/prototype_shared_gradient_cpu_test.json`.
