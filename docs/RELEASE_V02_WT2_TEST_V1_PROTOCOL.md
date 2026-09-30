# Frozen E8/W5 v0.2.0 paired WikiText-2 test

Protocol identifier: `E8W5_RELEASE_V02_WT2_TEST_V1`.

## Model and sources

Evaluate the existing public `EndlessChasing/mamba2-8b-e8w5` release
`v0.2.0-resurface`, source tag commit
`e89c764e33a7fcd72095e1b9b9ea47f22d3ff7bb`. The restored raw manifest must have
SHA-256 `7dd96a44d3de6e634b949e2fb94bbc004c0aecc8404d3848d13f83b396616a1f`.
The archived software and all 119 payload files are checked by the unchanged
published loader. The actual decoded 507 FP16 parameter tensors must equal the
manifest ledger. The separate 224-tensor soft adapter must equal its serialized
published ledger and retain its original binding. No fitting, calibration,
candidate selection, weight edits or state codec edits are permitted.

The external runner imports only `mamba_e8w5` from the supplied archived software
directory. It disables Python bytecode creation before these imports. All source
hashes, dataset hashes, loaded tensor hashes, backend versions, numerical settings
and per-arm CUDA memory peaks are recorded. This test does not mutate the sealed
raw package or archived source tree.

## Corpus and population

Use `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Join all dataset text rows with two
newlines; use the unchanged NVIDIA SentencePiece tokenizer, no automatic BOS/EOS.
Tokenizer SHA-256:
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.

Expected UTF-8 text SHA-256:
`696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`.
Expected raw little-endian int64 token SHA-256:
`5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608`.
There are 300,964 tokens and 300,963 next-token targets. Starts are `i*2048` for
147 windows. Each slice contains up to 2,049 tokens, with one boundary token
shared between adjacent windows. The last window contains 1,955 targets.
Every target is scored once; the final partial window is included.

## Frozen execution

Use the published `evaluation.evaluate_ppl(execution="prefill", logits_chunk=64)`:
native parallel SSD prefill, zero initial state for every window, FP16 decoded
weights and activations, FP32 logits/cross entropy and Python float aggregation.
No external recurrent cache is used by this PPL path. It is not a measurement of
FP16 cache rounding after every token, or compressed-resident inference.

Preserve the published environment: Python 3.10.12; torch 2.11.0+cu128;
mamba-ssm 2.3.2.post1; triton 3.6.0; numpy 1.26.4; datasets 4.8.5;
sentencepiece 0.2.1. Set TF32 off for CUDA/cuDNN, matmul precision `highest`,
FP16 reduced-precision reduction enabled, deterministic algorithms disabled,
cuDNN benchmark/deterministic disabled. The published numerical environment
allowlist is absent. Do not pin an alternate singleton Triton configuration.
Record actual loaded external module source hashes; their identity is a receipt,
not an unmeasured assertion of equality with historical source hashes.

## Paired arms and audits

Load once through `release_runtime.load_model`, which installs the mandatory
published bank. Verify its actual 507/224 hashes. Remove this bank with
`release_runtime.close_model` (the native bank exposes `close`, not `uninstall`).
Score all windows with no adapter. Install a fresh copy with the exact manifest
adapter binding and base ledger; assign the bank to the release model attribute.
Score the same windows with the bank active. Validate all 507 base and 224 adapter
tensor hashes, base identity/version/gradient invariance, and open soft bank state.
Remove the bank, repeat the first test window, and require exactly restored NLL,
PPL and row hash. The restoration probe is excluded from aggregate populations.

The completed report contains two complete PPL rows, per-window rows, coverage,
paired NLL changes, independent memory accounting, and all binding/reset audits.
No PPL threshold is applied, no candidate is selected, and no release is triggered.
Failure of a binding or population audit produces an incomplete receipt.

## Test history and interpretation

The published latest v0.2.0 paired scores were WikiText-2 validation:
7.622396587826496 without Resurface and 7.593163113563457 with Resurface, on
264,764 targets. The original `e8w5_v1` previously used the WikiText-2 test split
(source 7.2445, original E8/W5 8.7085, rounded in the original card). Those are
different model tensors and are not the latest v0.2.0 test scores. The project has
prior test exposure; this measurement is a frozen retrospective test of the
already published v0.2.0 artifact, not a claim of an untouched project test set.

Use fresh output paths and preserve all original research reports and releases.
