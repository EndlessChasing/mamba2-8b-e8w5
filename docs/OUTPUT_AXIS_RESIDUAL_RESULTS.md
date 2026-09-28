# Output-axis residual: complete results

Quantization and independent native evaluation completed with exit code 0.
Adding output-axis residuals improves complete-validation PPL from
**8.109428666 to 7.977512009 (1.62671%)** over the input-axis candidate.
The candidate remains **8.77176% above the FP16 source**, so the source +5%
quality target still fails. This is the lowest measured compressed-candidate
full-validation PPL in this project; the historical accepted all-small model
and its offline distribution remain unchanged. No promotion or publication.

## Fixed experiment

Follow the [frozen protocol](OUTPUT_AXIS_RESIDUAL_PROTOCOL.md), SHA256
`3ac0a78c4b93dcc84f2402b39847fd93c5daf1705cbca360c2cdfa84e7e545e4`.
The paired baseline is the [measured input-axis candidate](INPUT_AXIS_RESIDUAL_RESULTS.md):
56 enhanced inputs, original E8 outputs, W4 embedding, W5 head and 393 fixed
small tensors. Its full-validation PPL was remeasured in this process and
matches the previous 8.1094286657 result exactly.

Requantize all 56 output matrices `[4096,8192]` from original BF16 weights and
original `[8192,8192]` TRAIN Hessians. Keep the fixed 20-bit E8/axis format,
seed 1001+2×layer, two sweeps, scale 0.9 and damping 0.01. Only the fixed
axis amplitude is fitted from original rotated weights; no language-loss
training, new calibration data or recipe search is used. The candidate adds 56 changes to
the paired baseline, for 113 changes relative to historical accepted all-small.

## Paired quality results

Development used the same 64 previously observed reserved TRAIN windows /
131,008 targets per arm. The fixed primary passed the predeclared 1% gate,
enabling complete validation on 130 windows /264,764 targets.

| Arm | Observed TRAIN PPL | Complete-validation PPL |
| --- | ---: | ---: |
| Original source cast to FP16 |Not part of this development comparison |7.334175947318572 |
| Paired input-axis baseline, W4 embedding / W5 head |8.168229288617539 |8.109428665707021 |
| Input + output axis primary, W4 embedding / W5 head |8.013059657099294 |7.977512009035770 |
| Restored paired baseline repeat |8.168229288617539 |Not an additional full-validation arm |

- Development improves **1.89967%**, with 61/64 windows improving and 3 worsening.
  All 64 baseline-repeat windows and 2,048 CE chunks match exactly.
- Complete validation improves **1.62671%**, with **116/130** windows improving
  and 14 worsening. Weighted mean NLL decreases **0.016400833 nats/target**;
  the 1% improvement condition passes.
- The source +5% limit is **7.700884744684501**. Candidate PPL 7.977512009
  exceeds it by an overall source gap of **8.77176%**: quality retention fails.
- Against the *historical* accepted all-small PPL 8.359867548, the cumulative
  reduction is **4.57370%**. All-small was not a fourth same-process arm here.

Both development and validation have informed development; this is not
untouched test evidence. Recall/MK is unmeasured for this candidate and deferred.

## Completed quantization and CPU checks

The quantization receipt records exact packed-index readback and decoded FP16
hash equality for every output file, with finite positive residual amplitudes.
Its manifest lists 56 new files and 61 inherited files. The terminal process
receipt binds the final quantization report and records 472.24524 s wall time
(470.53907 s inside the quantizer).

CPU preflight passed with CUDA uninitialized: original source/calibration
bindings, 56 output Hessians, 61 inherited files, the 20-bit format roundtrip
and malformed-file rejection. The separate evaluator self-test passed inventory,
paired-gate and complete-validation tail checks. Actual-overlay verification
subsequently completed in 73.41604 s with CUDA uninitialized, checking 392 bound
files and the expected model inventories. These CPU checks do not establish
model quality or replace native loaded-tensor audits.

The native evaluator independently reloads the 56 stored output files, verifies
their decoded hashes, and checks all 507 actual tensor hashes before/after each
arm. The candidate changes exactly 56 outputs relative to the paired baseline;
the other 451 tensors remain fixed. Relative to accepted all-small it changes
113 tensors. Final restoration preserves the enhanced inputs, W4 embedding,
W5 head and 393 small tensors, restores original outputs, and passes all 507
paired-baseline hashes plus bound-file rechecks.

All arms use identical token hashes and window boundaries. The full-validation
tail includes all 572 targets with a final 60-token CE chunk. Source and paired
full-validation PPL match their prior results exactly. Independent arithmetic
checks of local receipts reproduce the weighted PPL, window counts, gates,
decoded-file bindings and restoration hashes; no extra GPU run was performed.

## Actual raw capacity

| Scope | Measured bytes |
| --- | ---: |
| Existing paired-baseline raw data |3,021,487,451 |
| Original 56 output files to replace |470,776,208 |
| Extra output-axis code plane |117,440,512 |
| Net file-header growth |829 |
| New 56 output files |588,217,549 |
| Raw replacement increase |117,441,341 |
| Resolved primary raw data |3,138,928,792 |
| Separate overlay manifest |193,007 |
| Physical new overlay including its manifest |588,410,556 |

Resolved raw data increase 3.88687% over the paired input-axis baseline.
They increase 8.74581% over the historical accepted raw size of 2,886,482,462 B.
Resolved data replace the original outputs; adding the physical overlay again
would double-count weights. These raw figures exclude manifests, tokenizer,
software and reports. No new Huffman distribution or compact GPU-residency
result is claimed. The candidate deliberately spends more bytes.

## Completed receipts

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
| [Quantization](../reports/output_axis_residual_v1_quantization.json) |193,162 |`678b5a215a9f8404a6c9baa510e62a4378e0e2e8c57ff5fd477bbbe8d3673c2f` |
| [Overlay manifest](../reports/output_axis_residual_v1_manifest.json) |193,007 |`2b6f56e4f1705ddfad18fc58662a28b1f36b7fd63dd4ff540e733e1f0747812f` |
| [Quantization process](../reports/output_axis_residual_v1_quantization_process.json) |1,182 |`aca1e68189d6e96ca22d47a4a212de39b46f9b51560c10c9d6acd55126484a7e` |
| [CPU preflight](../reports/output_axis_residual_v1_cpu.json) |47,362 |`c57e8cbdcec09db6af9b63458b85808993f66480e17fcda334fb7f0fb7ad0107` |
| [Evaluator CPU self-test](../reports/output_axis_residual_evaluator_cpu.json) |1,956 |`d137d8f5a48e54fdbf4babdcac72d2395ea9aecbe1099ee757dd53cd6faf4ef6` |
| [Actual-overlay CPU verification](../reports/output_axis_residual_v1_eval_cpu.json) |465,885 |`baa8b3805b053467c3fc70d4e1fe54e4322c0d37efa942f8b54e35944cf00167` |
| [Final native evaluation](../reports/output_axis_residual_v1_eval.json) |2,543,247 |`23f48ae741195ca646f68dd5008493d003e15335d7e69c034e6d424b69871eb4` |
| [Evaluation process](../reports/output_axis_residual_v1_eval_process.json) |1,157 |`c4c1302fd287bd27ba9545e6ae28982c8327a54e25e26bf2696d2857f930112c` |

The evaluation process receipt records `complete:true`, `experiment_complete:true`,
exit code 0 and the final report SHA. Wall time is **618.19173 s**, with
**615.60519 s** inside the evaluator. Peak GPU allocation is **33,626,613,760 B**
(reserved 34,259,075,072 B) for the decoded FP16 quality reference, not compressed
inference residency or isolated throughput.

This fixed experiment is complete. No layer selection, extra sweeps,
complete package build, test/MK measurement or promotion was performed.
Publication remains on hold.
The measured gain is a partial repair at a larger raw size, not equal quality
or a global minimum claim.
