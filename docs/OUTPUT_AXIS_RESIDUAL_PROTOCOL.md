# Output-projection axis-residual repair protocol

Declared after the completed input-axis experiment and before implementing
or measuring this new candidate. The completed input evaluation is
`reports/input_axis_residual_v1_eval.json`, SHA256
`1dd6be0ddb4481e765da2658926ecea92183521ac9862b914fc3d3ba2a80a7eb`.
It measured full-validation PPL 7.334175947318572 for source,
8.359867548009992 for accepted all-small, and 8.109428665707021 for
enhanced inputs plus W4 embedding / W5 head. The latter improves PPL,
but remains 10.5704% above source and fails the source-plus-5% target.
This experiment prioritizes further PPL repair. MK, Resurface and publication
remain deferred; existing distributions and completed experiments stay intact.

## Fixed candidate

Start from that measured input-axis / W4-embedding / W5-head candidate.
Requantize exactly all 56 `out_proj` matrices, each `[4096, 8192]`, from
the original BF16 checkpoint and original TRAIN calibration Hessians,
each `[8192, 8192]` FP32. Use the unchanged codec's `vector_quantize`:
`residual_bits=4`, seed `1001 + 2*layer`, damping `.01`, scale `.9`,
two LDLQ sweeps, feedback enabled. The same axis-amplitude fitting rule
uses original rotated weights only. No language-loss training, new
calibration, layer selection, amplitude/scale search or extra sweeps.

Keep all 56 already verified input-axis files, W4 embedding, W5 output
head, and all 393 accepted small tensors fixed. Bind the input manifest
SHA256 `0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85`,
the completed input evaluation above, original source checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`,
and original calibration manifest SHA256
`70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab`.
Verify every original output Hessian against both calibration and parent
ledgers. Record source/code/input-file hashes and exact inheritance.

Write a separate output-only overlay, with exclusive outputs, incremental
receipts, exact 20-bit packed-index checks and finite decoded FP16 hashes
matching quantizer output. Preserve the frozen codec/runtime/input-stage
files. Reuse the generic strict axis reader; a separate output-stage
verifier must enforce its own geometry, recipe and complete inventory.
Fit errors are diagnostics only, not a quality gate.

## Independent paired development evaluation

Use these three arms in fixed order:

1. Measured input-axis candidate with original E8 outputs, W4 embedding
   and W5 head: the paired baseline.
2. The same model with all 56 output projections replaced by axis-residual
   files: the sole primary candidate.
3. Restore the paired baseline and repeat all development CE chunks exactly.

Use the same 64 previously observed reserved TRAIN windows from
`teacher_kl_compensation_v1`, manifest SHA256
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
Keep stored order, 2048 stored tokens / 2047 targets per window and
131,008 targets per arm. Use native FP16 backbone/head, FP32 full-vocabulary
CE in 64-token chunks with the final 63-token chunk, FP64 host sums and
fresh state per window. These are development comparisons, not fresh tests.

Reuse the frozen input evaluator's scoring and identity helpers without
calling quantization/training during evaluation. Independently decode
actual files; hash all 507 tensors before and after each arm. Exactly
56 tensor identities change between the paired baseline and candidate;
relative to accepted all-small the changed counts are 57 and 113.
Do not hold a second dense copy of all replaced matrices: stream replacement
and re-decode originals for restoration. Restore and audit the final model.
Record historical drift separately from exact same-process controls.

## Gate and conditional complete validation

Require primary development PPL to be at most 99% of the paired input-only
baseline. If it fails, stop this fixed recipe and skip full validation.
Do not use reconstruction error or an intermediate window to select a
different candidate, precision, layer subset or checkpoint.

Only after a pass, evaluate three arms on the established complete
validation: source FP16, paired input-only baseline, and freshly reloaded
primary. Include all 130 windows / 264,764 targets, including the final
572-target window and its 60-token CE tail. Bind the exact established
window/token plan. Report at least 1% improvement over paired baseline
and at most 5% above source as separate results. Comparisons with the older
all-small baseline are historical, not an additional same-process arm.
This validation set has informed development and is not an untouched test.

No automatic promotion, packaging or publication follows completion.
An improvement short of the source-plus-5% target remains a partial repair.
Preserve both positive and negative outcomes and all failed receipts.

## Capacity and environment

The additional four bits per eight output weights add exactly
117,440,512 code bytes (112 MiB), plus changed JSON headers. Original
56 output files occupy 470,776,208 bytes. The expected new output-only
overlay is 588,216,720 bytes plus header differences and its manifest.
Resolved primary raw data are expected to be 3,138,927,963 bytes plus
the new header differences, versus 3,021,487,451 bytes for the paired
input-only baseline. Measure actual files; exclude manifests, software
and tokenizer from these raw-data figures and label that scope.

This deliberately adds capacity. It is not a same-size repair or a measured
Huffman distribution. Original Hessians describe original-model activations;
their suitability for this compressed candidate is not guaranteed. Input
and output improvements cannot be assumed additive. GPU evaluation uses
dense FP16 weights; compact runtime residency remains unmeasured.
