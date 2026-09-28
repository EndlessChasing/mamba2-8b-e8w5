# W4 vocabulary penalty diagnostic

Declared before execution. Following the negative scalar-temperature screen,
measure the PPL cost of reducing vocabulary precision before investing the
saved bytes in E8 projection precision. The accepted all-small model and its
distribution remain the baseline. PPL is the priority; MK and publication
remain deferred.

## Fixed artifacts and four arms

Create two actual group-128 W4 vocabulary files from the pinned original
checkpoint, using the existing `write_uniform(bits=4, rows_per_chunk=256)`.
Pass original checkpoint tensors exactly as the original W5 quantizer did:
the writer converts each source row batch to FP32. Do not requantize decoded
W5 or first cast source weights to FP16. The source checkpoint SHA256 is
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.

Evaluate these four arms, in this order, without fitting or temperature:

1. Accepted all-small with W5 embedding and W5 head.
2. W4 embedding and W5 head.
3. W5 embedding and W4 head.
4. W4 embedding and W4 head.

All 112 E8 projections and 393 small tensors remain the accepted values.
Only the named vocabulary tensors may change. Load replacements from their
actual serialized W4 files. Preserve all 507 expected tensor content hashes
for every arm and verify the full model before and after each scoring pass.
Restore both W5 tensors and repeat the baseline after the fourth arm.

## Fixed observed development data

Use the existing 64 reserved TRAIN windows in their stored order from
`training_data/teacher_kl_compensation_v1`, whose data manifest SHA256 is
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
These windows have already informed diagnostics. They are an observed
development set, not fresh or independent generalization evidence.

Each window has 2048 stored tokens and 2047 next-token targets, totaling
131,008 targets per arm. Use fresh state per window, native FP16 backbone
and head, then FP32 full-256000-vocabulary cross entropy in 64-token chunks
with a 63-token tail. Aggregate token-weighted NLL with FP64 host summation.
Do not load validation/test data or compute MK. No teacher model or KL
calculation is required for this paired vocabulary comparison.

Require the final repeated baseline CE chunks to equal the initial baseline
within this process. Report historical baseline drift separately; do not use
cross-process bitwise equality as an execution gate. No parameter, layer or
window selection follows an intermediate arm result.

## Controls, accounting and interpretation

Before the GPU run, perform bounded CPU checks for W4 packing/readback,
group scales, non-multiple row chunks and weighted loss tails. Verify actual
data and bound input/code hashes without initializing CUDA. Record actual
source dtype, source identity, file hashes, readback identities and execution
settings. Keep previous codec/runtime files unchanged. Save partial/failure
evidence, and refuse to overwrite pre-existing experiment artifacts.

Each vocabulary has 1,048,576,000 parameters. Under the existing format,
W5 uses 671,744,079 bytes and W4 is expected to use 540,672,079 bytes,
including 16,384,000 scale bytes and a 79-byte header. Confirm actual sizes;
both W4 files are expected to save 262,144,000 raw bytes. Metadata, tokenizer
and software accounting remain separate. No new Huffman distribution or
compressed runtime residency is implied by this raw byte calculation.

Report each arm's aggregate PPL, NLL difference from baseline, relative PPL
change and per-window changes. Also report the interaction between the two
replacements in NLL, without interpreting it as causal layer attribution.
This is a diagnostic with no quality promotion gate: W4 alone is expected to
trade quality for space. Complete all four arms unless integrity/numerical
controls fail. A subsequent projection-bit reallocation is a separate
experiment, requiring its own actual package and PPL evidence. Do not claim
that vocabulary savings guarantee a useful combined model.
