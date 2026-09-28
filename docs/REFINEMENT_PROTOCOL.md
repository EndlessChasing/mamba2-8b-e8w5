# LDLQ refinement protocol: two versus eight sweeps

Status: training-proxy screen declared before its results. Publication, pushes
and model uploads remain paused. This protocol changes offline codeword search
effort without increasing the nominal projection bitrate. It does not promise
that more sweeps improve either reconstruction or language quality.

## Objective and fixed inputs

The current component diagnosis attributes most measured PPL degradation to E8
projections. The experiment therefore changes only `tune_iters=2` to
`tune_iters=8` in the pinned `LDLQ_buffered_lowmem` primitive. Both counts are
additional coordinate sweeps after the same initial LDLQ pass; eight is not
eight passes including initialization. Each arm is recomputed from identical
original weights, Hessians, preprocessing and seeds.

Keep all of the following fixed:

- Official source checkpoint, tokenizer, model configuration and parent package.
- The 65536 training-token calibration and all 112 original Hessian files.
- E8P12 codebook, 16-bit index per eight weights, no axis residual or new adapter.
- Scale override `0.9`, damping `0.01`, buffer width `128`, diagonal balancing,
  DCT/Hadamard rotation, and stored metadata precisions.
- Seed `1000 + 2*layer + part`, with part 0 for `in_proj` and 1 for `out_proj`.
- Both separate W5 vocabulary files, the other FP16 tensors and the decoder.
- Frozen runtime, codec, quantizer and vendored QuIP# sources; use a separate
  driver and output directory for this experiment.

Parent raw manifest SHA-256:
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
Calibration manifest SHA-256:
`70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab`.
Bind the experiment report to those hashes, every source/Hessian file used,
driver/code hashes, environment and exact per-matrix settings.

## Why monotonic improvement is not guaranteed

For original input second moment `H = E[x xᵀ]`, score reconstruction with

`J(W_hat) = tr((W_hat - W) H (W_hat - W)ᵀ)`.

Use the final decoded FP16 `W_hat`, as the current codec does. Its reported
relative calibration-output RMS error is
`e = sqrt(J / tr(W H Wᵀ))`; hence the relative squared-error improvement is
`r = 1 - J8/J2 = 1 - (e8/e2)^2`. The acceptance gate below uses `r`, not the
percentage change in RMS error or unweighted weight error.

The primitive forms an eight-coordinate conditional center using the inverse
diagonal block of transformed `H`, then selects a Euclidean-nearest E8 vector.
Exact conditional minimization would instead use that block's quadratic metric,
unless the block is proportional to identity. There is no explicit reject-if-loss-
increases check or best-so-far retention in the pinned primitive. Moreover, its
internal Hessian is normalized, damped, balanced and rotated; the reporting
objective above uses original `H` after inverse transforms and FP16 rounding.
Even monotonic internal progress would not guarantee monotonic reported `J` or
end-to-end PPL. All stages therefore require measured comparisons.

References: [codec](../mamba_e8w5/codec.py),
[pinned LDLQ primitive](../third_party/quip-sharp/lib/algo/quip.py),
[pinned E8 quantizer](../third_party/quip-sharp/lib/codebook/latticee8_padded12.py).
The project's earlier smaller-model runs did not isolate a two-versus-eight
sweep effect; they provide motivation, not evidence that this repair will work.

## Stage 1: fixed six-matrix training screen

Evaluate exactly `layer0`, `layer18`, and `layer55`, each with `in_proj` and
`out_proj`: six matrices, two sweep settings per matrix. The chosen early,
interior and final layers are fixed before results, not replaced if inconvenient.

Record per arm: original-H `J` or equivalent RMS proxy, relative weight error,
decoded FP16 SHA, actual file bytes/SHA, finite-value checks, disk readback equality
and elapsed time. Record whether the recomputed two-sweep arm reproduces the
parent's decoded identity. Resolve any material reproduction discrepancy before
expanding the experiment; retain both receipts rather than replacing the parent.

**All three predeclared conditions must hold:**

1. Median relative squared-error reduction across the six matrices is **at least 1%**.
2. **At least four of six** matrices improve by **at least 0.5%**.
3. **No matrix worsens by more than 0.1%**.

For example, an RMS error change from 0.200 to 0.199 is a 0.9975% reduction in
squared error, slightly below 1%. Nonfinite values, missing matrices, mismatched
inputs or corrupt reconstruction fail the screen. A zero-error two-sweep arm
cannot claim a relative improvement; report it explicitly and reject a nonzero
eight-sweep error for that matrix.

If the screen fails, stop this fixed eight-sweep route. Do not lower thresholds,
select only successful matrices or add sweep counts after inspecting results.
Any different search requires a new protocol and must retain this outcome.

## Stage 2: complete candidate only after a screen pass

Generate all **112 eight-sweep E8 projection files** under a new candidate
directory. Keep the six pilot matrices in the full inventory, using the same
settings and hashes; do not construct a post hoc mixture of two/eight sweeps.
Record the original-H proxy for every matrix, including any regressions that
were absent from the pilot. Verify all encoded/decoded values and all 507 model
tensors before interpreting quality.

Represent the candidate with a separate overlay manifest, never by editing the
parent raw manifest or replacing its files. The overlay must identify:

- Its own format/version, completion flag and independent manifest SHA.
- Parent manifest SHA, source/config/tokenizer identity and code/calibration pins.
- Exactly 112 replacement projection paths, shapes, bytes, file SHA and decoded
  FP16 SHA, mapped to unique model parameter names.
- An explicit list of inherited files with their parent bytes/SHA: both W5
  vocabulary files, `other_fp16.pt`, `e8_codebook.bin`, and `config.json`.
- The resolved model parameter coverage, file inventory and total required bytes.

Validate parent and replacement files before loading. Missing, overlapping or
unbound replacements are errors. An overlay is not a complete standalone raw
package and must not be silently passed to a loader that ignores its replacements.
An evaluation receipt must carry both parent and overlay identities, or an
equivalent hash of the complete resolved file ledger. A parent raw manifest SHA
alone does not identify the refined candidate.

## Stage 3: paired development quality

Run parent and candidate in the same process and environment on the frozen four
validation windows and all 12 normal plus 12 target-removed public MK prompts.
Retain a source baseline when practical. Verify parameter identities, execution
mode, prompt/token hashes and each candidate endpoint. Record the known small
historical parent-Q endpoint discrepancy separately; do not disguise it by
rewriting old reports or comparing across different numerical contexts.

The following downstream advancement rule is **accepted before any refined-
candidate language evaluation**, separate from the Stage 1 training-proxy gate.
Bind this protocol's SHA in the experiment report before recording dev scores:

- Candidate PPL is at least **1% below** the same-process parent, equivalently
  candidate-minus-parent mean NLL is at most `log(0.99)`.
- Normal MK successes are no lower than the paired parent, and target-removed
  matches are no higher.
- All integrity/finite-output checks pass; report every paired case and window.

This tests material improvement over the degraded parent. It does not replace
the original source-relative +5% PPL / recall quality target and does not establish
statistical equivalence from 12 normal prompts. If the training screen passes but
this development gate fails, retain the result and stop this route: do not advance
to full validation or promote it. No retrospective threshold changes are permitted.

Optional confirmation after a development pass can use the complete validation
split with the same source/parent/candidate pairing. It remains validation data,
which has already been consulted in diagnosis. The original full test result has
also been inspected. Keep it visible and do not call a later test-driven choice
an untouched held-out result. This protocol does not authorize publication.

## Storage and compute accounting

Eight sweeps change code values without adding code fields: the E8 indices and
fixed-shape balancing/sign/scale payload have the same raw byte capacity as two
sweeps. Metadata and actual file counts must still be measured. Changed code
frequencies can change Huffman size in either direction; **same nominal bitrate
does not imply identical entropy-coded bytes**.

Report separately:

1. New overlay bytes: 112 replacement files plus its manifest.
2. Complete logical model bytes: replacements plus every inherited file needed
   for loading, codebook, configuration and manifest; add tokenizer, software,
   licenses and reports when quoting a full distribution size.
3. Actual experiment disk use: if the full parent directory remains necessary,
   count it too. Do not advertise overlay-only size as a standalone model size or
   subtract unused parent files that are still physically required by the loader.
4. Offline refinement time and peak memory. These are separate from inference
   cost; the current accuracy runtime still expands weights to FP16.

No original floating-point checkpoint may be an undeclared inference dependency.
Source weights and Hessians are reproduction inputs, not part of a successfully
resolved quantized model. No residual adapter or FP16 rescue is introduced here.
