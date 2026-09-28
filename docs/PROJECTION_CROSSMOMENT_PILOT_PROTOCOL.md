# Candidate-input cross-moment repair: six-projection pilot

Declared before collecting candidate statistics or observing pilot outcomes.
This is a separate, PPL-priority research experiment. Publication remains held;
recall is deferred and Resurface recovery is not assumed.

## Fixed model and intervention

Start with the original two-sweep E8/W5 package and the final1024 all-small
overlay (manifest `edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006`).
Teacher: verified original NVIDIA source evaluated in the existing FP16 runtime.
Pilot projections: layers0,18,55, each in_proj and out_proj, exactly six matrices.
Keep all393 trained small tensors, both W5 vocabularies, and other106 E8
projections frozen. No small-parameter retraining, new bias, state quantization,
additional residual weights, codebook learning or architecture change.

Compare three fixed arms:

1. Current decoded E8 projections, unchanged.
2. H-refresh control: original FP16 source weights quantized using current
   candidate projection-input covariance H.
3. Cross-moment repair: quantize the anchored least-squares target below under H.

The unquantized target is only a local fitting diagnostic, not a deployable arm.
The separate original-projection restoration factorial measures coadaptation;
it does not select pilot hyperparameters or make an attainable-quality claim.

## Fresh training positions

Use the same pinned WikiText-2 TRAIN token stream and native tokenizer. Select
2048-token grid intervals excluding any token overlap with the original32
calibration windows or the256 all-small training windows. From the remaining
grid choose48 spread ranks `floor(i*(eligible_count-1)/47)`, i=0..47. Every third
selected interval (indices2,5,...47) is held out from projection fitting;
the other32 fit the statistics. Bind starts, token digests and source hashes
in the report before collecting statistics.

Before the full collection, use the first32 input positions of the first fit
window for a discarded pairing/serialization smoke check. Independently scored
native baseline error must match collected native error within1e-6 relative.
Those temporary statistics are discarded, and no weight is fitted by the smoke.

Fit:65536 input positions. Heldout:32768 projection-input positions and32752
language targets. Positions are disjoint within the same TRAIN corpus;
this is not an independent corpus or final validation result. No validation
or test tokens enter fitting, local selection or this pilot's language screen.

## Math and fixed quantizer

Run teacher and unchanged candidate on identical windows from zero recurrent
state. For each projection, collect actual candidate input Xq and actual native
FP16 teacher linear output Yt, then accumulate in FP32 with TF32 disabled:

H = Xq.T Xq / N; K = Yt.T Xq / N.

Do not center inputs or approximate Yt by a weight-times-cross-covariance product.
With Wa the current decoded FP16 E8 matrix, use fixed
lambda = 0.01 mean(diag H):

Wstar (H + lambda I) = K + lambda Wa.

Solve using FP32 Cholesky without an explicit inverse or adaptive damping.
Require relative normal-equation residual <=1e-4. Pass raw, undamped H to the
unchanged codec with damping0.01, scale_override0.9, tune_iters2 and original
seed1000+matrix_index. This gives the same ridge metric without double damping.
Recomputed balance/scale and codeword assignments use their original formats.
Raw E8 file bytes per matrix must equal the parent, with exact export/readback
FP16 equality. Equal raw bytes do not imply equal Huffman size.

All six statistics come from the unchanged candidate prefix. This pilot is
not sequential layer repair: in_proj replacements do not refresh out_proj
statistics, and earlier replacements do not refresh later-layer statistics.

## Measurements and advancement rule

For every arm/matrix, score fit and heldout real-linear quadratic error from H,
K and teacher energy, explicitly separating this proxy from native FP16 GEMM.
Also rerun all16 heldout windows to measure native FP16 projection-output MSE
against actual teacher output while the prefix remains the unchanged candidate.
Record z/x/B/C/dt partitions of in_proj where available. New matrices must have
finite values and pass raw and decoded identity checks.

Then install all six reloaded replacements together and score each arm on the
same16 heldout TRAIN windows using frozen evaluation.py, native FP16 head GEMM
and FP32 loss, chunk64. Restore every original parameter reference after scoring.
This joint PPL measurement includes changed prefix interactions.

Advance cross-moment repair only when ALL hold:

- Native heldout teacher-output MSE: median reduction >=1% versus current E8;
  at least4/6 matrices improve >=0.5%; no matrix worsens >0.1%.
- Median native heldout MSE reduction >=0.5% versus H-refresh control.
- Joint six-replacement heldout TRAIN PPL does not exceed current candidate PPL.
- All finite-value, unchanged-tensor, coverage and exact decode checks pass.

Report all three arms regardless of outcome; no per-matrix cherry-picking,
learning-rate search or additional fitting sweep. A passing pilot permits design
of a separately bounded larger repair; it does not establish full-validation
PPL improvement or the source+5% target. A failing pilot is retained and stopped.

## Storage and execution

Two FP16 models, six H/K pairs and decoder/solver workspace fit the96GB GPU.
Collect fit and heldout statistics in GPU memory, without saving multi-GB
statistics to disk. Save two sets of six E8 files (~157MiB total), small JSON
receipts and code only; keep all model binaries on the GPU host. The complete
candidate logically replaces the same-size parent files and adds no inference
payload fields. No compressed-container size is claimed without repackaging.
