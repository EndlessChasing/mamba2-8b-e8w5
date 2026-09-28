# Input-projection axis-residual repair protocol

Declared after the completed W4 vocabulary diagnostic and before this
experiment's implementation or quantization. The W4 result is bound by
`reports/vocab_w4_v1.json`, SHA256
`3ae1bb986ddfa7f2e514fb2451885a83feee5b6c5cc3a1924f248b4a259e353b`.
On its observed development set, W4 embedding costs 0.158945% PPL and W4
head costs 5.307371%. This motivates retaining the W5 head for the primary
PPL repair candidate. No size ceiling was specified by the user; the primary
candidate deliberately spends additional bytes. MK and publication remain
deferred. The accepted all-small model is unchanged.

## Fixed quantization, no language-loss training

Requantize exactly all 56 `in_proj` matrices, shape `[18560, 4096]`, from
the original BF16 checkpoint and the original `calibration/v1` Hessians.
Use the frozen codec's `vector_quantize` with `residual_bits=4`,
`seed=1000+2*layer`, `damping=.01`, `scale_override=.9`, `tune_iters=2`,
and feedback enabled. Preserve all 56 `out_proj` matrices and all 393
accepted small tensors. No layer selection, scale/damping search, extra
sweeps, new calibration data, scalar temperature or compensation training.

The existing code fits one axis-correction amplitude per matrix from its
original rotated weights. Every eight-value vector stores 16 E8 bits plus
four axis/sign bits, hence 2.5 bits/weight before metadata. This is a new
LDLQ quantization with a larger codebook; it is not a guaranteed-improving
correction to the old codes. There is no zero-correction option.

Write a separate 56-file overlay, with source, parent, accepted-small,
calibration, every Hessian and code hashes in its manifest. Use the pinned
source checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`
and calibration manifest SHA256
`70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab`.
Keep the original codec and runtime files unchanged. Verify each stored
20-bit representation, finite positive amplitude, exact file length and
decoded FP16 readback against the quantizer output. Save incremental
receipts and preserve failures; no automatic retry or overwritten outputs.

## Four fixed development arms

Evaluate in this order using an independent loader of the actual new files:

1. Accepted all-small, original E8 inputs, W5 embedding / W5 head.
2. Enhanced input projections, W5 embedding / W5 head: precision control.
3. Enhanced inputs, W4 embedding / W5 head: **primary candidate**.
4. Enhanced inputs, W4 embedding / W4 head: near-capacity control.

Reuse the actual W4 files whose hashes are recorded in the completed W4
diagnostic. Audit all 507 parameter hashes before and after each arm;
expected changed counts relative to accepted baseline are 0, 56, 57 and
58 respectively. Restore the baseline and repeat all development CE chunks
exactly in the same process. Stream matrix replacement instead of retaining
a second dense copy of all 56 input matrices. Model states reset per window.

Use the same 64 previously observed reserved TRAIN windows, in stored order,
from `teacher_kl_compensation_v1`, manifest SHA256
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`:
2048 stored tokens / 2047 targets per window, 131,008 targets per arm.
Native FP16 backbone and head, then FP32 full-vocabulary CE in 64-token
chunks and FP64 host summation. Reuse the established tail and native CE
checks; add only checks needed for 20-bit file/decode correctness. Record
historical drift separately from exact within-process controls.

These are development comparisons, not fresh generalization evidence.
All four arms must finish before judging the primary candidate. Do not
choose a different primary arm after observing intermediate results.

## Conditional complete validation

Require the primary candidate's development PPL to be at most 99% of the
paired baseline. If it fails, stop this fixed recipe without full validation,
extra quantization sweeps or layer/precision selection. Retain all controls.

Only if it passes, independently evaluate source FP16, accepted baseline
and the reloaded primary candidate on the established full validation:
130 windows / 264,764 next-token targets, including the final partial window.
Report both at least 1% improvement over paired baseline and at most 5%
above paired source PPL. This validation set has informed development and
is not an untouched test set. The other two controls have development-only
results under this protocol. No automatic promotion or publication follows
process completion; a usable candidate also requires actual package checks.

## Capacity accounting and scope

The four additional bits per eight-value vector add 266,076,160 code bytes
across all inputs. Before changed per-file JSON headers and new manifests:

| Arm | Resolved raw data bytes | Difference from accepted |
| --- | ---: | ---: |
| Accepted baseline | 2,886,482,462 | 0 |
| Enhanced inputs, W5 / W5 | 3,152,558,622 | +266,076,160 |
| Primary: enhanced inputs, W4 / W5 | 3,021,486,622 | +135,004,160 |
| Enhanced inputs, W4 / W4 | 2,890,414,622 | +3,932,160 |

The primary candidate is approximately 4.68% larger in raw data, not the
same capacity. Measure actual headers, manifests and decoded identities.
Physical overlay storage is about 1.33 GB and reuses the existing W4 files;
do not add overlay bytes to the resolved-model totals a second time.
Current Huffman packaging rejects axis-residual files. Actual complete
distribution size requires a separately verified packing extension and
readback; no approximately-2.7-GB bundle or compressed inference-residency
claim is supported by these planning numbers. Dense FP16 PPL execution
memory remains a separate measurement.
