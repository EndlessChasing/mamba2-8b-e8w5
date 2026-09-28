# E8 coordinate refinement: controlled six-matrix screen

The experiment changes only the number of additional LDLQ coordinate sweeps,
from two to eight. It uses the original training Hessians and decoded FP16
weights. The predeclared labels are layer 0, 18, and 55, each `in_proj` and
`out_proj`; no layers are substituted after observing results.

## Method

The unchanged codec applies the same diagonal balance, seeded sign flips,
DCT/Hadamard rotations, E8 codebook, initial LDLQ pass, scale 0.9, and damping
0.01. Seeds remain `1000 + 2*layer + part`. Vocabulary weights remain W5 and
the remainder stays FP16. Eight sweeps add offline computation without adding
stored E8 fields.

For each matrix, the separate driver first reproduces two sweeps and requires
both the encoded-file SHA and decoded FP16 SHA to match the completed parent
package. It then recomputes eight sweeps from the same original source and
Hessian, serializes the result, and requires exact FP16 disk readback.

The independent measurement uses the original undamped Hessian
`H = XᵀX / T` and final decoded FP16 weights:

`J = tr((W_hat - W) H (W_hat - W)ᵀ)`.

`J / number_of_output_channels` is the mean squared projection-output error
over the calibration tokens. `J / tr(W H Wᵀ)` is relative output MSE, the
square of the codec's reported relative RMS error. The driver uses FP32 GEMM
with FP64 accumulation, and its CPU test compares this result against direct
calibration outputs. This metric contains no damping penalty and no PPL data.

The gate was declared before the run: median reduction in `J` at least 1%,
at least four of six matrices improve by 0.5%, and none regresses by more
than 0.1%. See [the complete protocol](REFINEMENT_PROTOCOL.md).

## Results

**The training-proxy screen passes.** All six matrices improve; median
squared-output-error reduction is **1.6574%**, minimum **1.0351%**, maximum
**2.1863%**. All six baseline raw-file and decoded-FP16 hashes reproduce the
parent, and all twelve encoded results decode exactly to their quantized
FP16 tensors. These are small projection-MSE improvements, not measured
language-quality improvements.

| Matrix | Output MSE, 2 sweeps | Output MSE, 8 sweeps | MSE reduction | Quantization time, 2 / 8 sweeps |
|---|---:|---:|---:|---:|
| layer0.in_proj | 0.002589512 | 0.002558561 | 1.1953% | 3.326 / 7.603 s |
| layer0.out_proj | 0.000003427413 | 0.000003352481 | 2.1863% | 3.918 / 11.755 s |
| layer18.in_proj | 0.012011419 | 0.011780227 | 1.9248% | 2.457 / 7.128 s |
| layer18.out_proj | 0.0006258662 | 0.0006125835 | 2.1223% | 3.943 / 11.041 s |
| layer55.in_proj | 0.019558992 | 0.019356539 | 1.0351% | 2.543 / 7.141 s |
| layer55.out_proj | 0.067171830 | 0.066238180 | 1.3899% | 3.820 / 10.579 s |

Summing `J` over these six matrices gives a 1.3712% decrease. That aggregate
depends on each matrix's output scale; the predeclared median gate gives each
matrix equal weight. Both figures are reported without changing the gate.

Raw E8 bytes remain **19,016,655** for each input projection and **8,406,718**
for each output projection. Either six-file arm totals **82,270,119 bytes**;
both preserved arms together use **164,540,238 bytes**, plus small reports.
No entropy-compression comparison was performed.

The six quantizations take 20.01 seconds at two sweeps and 55.25 seconds at
eight, a 2.76× increase. Including readback and independent output metrics,
the respective totals are 23.20 and 58.37 seconds. Entire screen time,
including source verification/setup, is 107.0 seconds. Maximum recorded
allocated GPU memory is 3,733,252,608 bytes. Extrapolating the three input
and three output measurements gives roughly **17.2 minutes** for 112
eight-sweep quantizations, before setup/readback/metrics; plan **18–21 minutes**
under similar GPU load. This is an estimate, not a full-run measurement.

Machine-readable evidence: [screen report](../reports/e8_refinement_screen_v1.json),
SHA-256 `a97afa3cb3252164dc90251e3733cc9bc26d52f829933c906a7758eb0132a22f`.
The protocol SHA is
`70fcc73a51bea70d0b308881af31482dce0f3e83175c03cbca2484ca230e6b89`.
No sixteen-sweep arm or PPL/MK evaluation was run in this screen. The next
eligible step is the full eight-sweep candidate under the existing protocol;
the screen does not authorize publication or establish source-relative quality.

## Interpretation limits

The previous 2.7B work changed calibration composition, context length,
readout configuration, and the number of sweeps together. Its better score
is not isolated evidence for a sweep-count effect. The local historical
record is `../LowDram ASIC Design/compute_for_compression.md`, lines 116–120,
and its linked `artifacts/e8_mixed2048_cd2_embed5_manifest_v1.json`.

The pinned coordinate update quantizes a conditional center using Euclidean
E8 nearest-neighbor search. The exact conditional quadratic objective has an
eight-dimensional Hessian block, so the Euclidean step need not minimize
that objective unless the block is proportional to identity. Neither more
sweeps nor a training-proxy reduction guarantees lower language-model PPL.

The most directly controlled compute-only first step is the measured
two-versus-eight comparison. If that route is insufficient, a separately
declared experiment could score nearby E8 candidates with the actual local
Hessian metric and retain the existing codeword whenever a proposal increases
the objective. That is a new algorithm requiring its own oracle and train-only
screen; it has not been implemented or measured here. Changing seeds, damping,
or scale simultaneously would obscure the present sweep-count effect.

## Reproduction

```bash
python scripts/diagnose_e8_refinement.py --self-test
python scripts/diagnose_e8_refinement.py \
  --source-dir models/source \
  --hessian-dir calibration/v1 \
  --parent-dir artifacts/e8w5_v1 \
  --out-dir artifacts/e8_refinement_screen_v1
```

The output directory must be new. The original package is never modified.
Raw candidates and baseline reproductions remain on the GPU host; only
the small report is copied to this repository. The partial overlay has
`complete=false` and cannot be loaded as a complete 112-projection model.
