# Input-axis residual: measured full-validation improvement

Quantization and independent native evaluation completed with exit code 0,
including final model restoration and bound-file checks. The primary candidate
improves complete-validation PPL from **8.3598675480 to 8.1094286657 (2.99573%)**.
It passes the declared 1% improvement condition, but remains **10.57041% above
source PPL 7.3341759473**, failing the source+5% quality target.

This is a measured improvement at **4.67715% more raw data**, not equal-capacity
or equal-quality compression. There is no automatic promotion, complete
package build or publication. The historical accepted all-small distribution
remains unchanged.

## Completed implementation and quantization

Follow the [frozen protocol](INPUT_AXIS_RESIDUAL_PROTOCOL.md), SHA256
`bf4af74313812a5f20373e9cfa68654b2cbe39a66841788ae4c66e965aa80088`.
Exactly 56 input projections of shape18560×4096 were requantized from original
BF16 weights and the original TRAIN calibration Hessians. The fixed recipe
uses E8 plus four axis/sign bits per eight-value vector, two LDLQ sweeps,
scale0.9, damping0.01 and seed1000+2×layer. No language-loss training or recipe
search was performed.

Completed proofs:

- All56 files have finite positive axis amplitude, valid20-bit indices and
  exact lengths, with actual JSON headers, code planes, scales and signs counted.
- Stored upper16-bit indices and packed axis/sign nibbles match quantizer
  output exactly. Every decoded FP16 matrix is finite and matches the
  quantizer output's content hash.
- Source, accepted parent/small values, original calibration and all56 Hessians
  are bound by hashes; input identities were rechecked after quantization.
- The independent CPU preflight verifies the actual complete overlay,274 bound
  inputs,61 inherited file entries and all507 expected tensor hashes for each
  arm. Replacement counts are0/56/57/58; inherited tensor counts are507/451/450/449.
  The subsequent native evaluation independently checks the complete model
  contents before and after every scored arm.

Quantization completed with exit code0 in **271.8821 seconds** internally.
The frozen builder and verifier remain separate from the native evaluator.

## Actual storage scopes

The 56 new files occupy **1,331,009,669 bytes**, replacing1,064,932,680 bytes.
The increase is **266,076,989 bytes**:266,076,160 extra code bytes and829 bytes
of net JSON-header growth. The actual168,431-byte overlay manifest makes the
separate overlay directory **1,331,178,100 bytes**; reused parents and W4 files
remain required. Do not add this physical overlay total again to resolved
model data.

| Resolved arm | Actual raw data bytes | Difference from accepted |
| --- | ---: | ---: |
| Accepted baseline |2,886,482,462 |0 |
| Enhanced inputs, W5 embedding / W5 head |3,152,559,451 |+266,076,989 |
| Primary: enhanced inputs, W4 embedding / W5 head |3,021,487,451 |+135,004,989 |
| Enhanced inputs, W4 embedding / W4 head |2,890,415,451 |+3,932,989 |

These totals include actual replacement-file headers but exclude manifests,
tokenizer, software, reports and training/checkpoint material. The primary is
**4.67715% larger** than accepted raw data. No new complete or entropy-coded
distribution has been built; the existing Huffman packer does not support
axis-residual files. Dense FP16 evaluation memory remains a separate scope.

## Completed observed-development comparison

All four fixed arms scored the same **64 previously observed reserved TRAIN
windows /131,008 targets per arm**, with fresh state and64-token CE chunks
plus the63-token tail. They are development comparisons, not fresh evidence.
Only the predeclared primary could enable complete validation.

| Arm | Development PPL | Change versus baseline | Improved windows |
| --- | ---: | ---: | ---: |
| Accepted baseline |8.397361922037003 |— |— |
| Enhanced inputs, W5 / W5 control |8.156725860168214 |-2.865615% |62/64 |
| Primary: enhanced inputs, W4 / W5 |8.168229288617539 |-2.728626% |62/64 |
| Enhanced inputs, W4 / W4 control |8.595606028628822 |+2.360790% |6/64 |

After all four arms, restoring the original input projections and W5
vocabularies reproduced every baseline CE chunk exactly. The primary's PPL
is below the declared8.3133883028 threshold, so its development gate passed.
The slightly better W5/W5 control did not replace the fixed primary and has
no complete-validation result under this protocol.

## Conditional complete validation

Following the passed gate, the evaluator scored source, accepted baseline and
the primary reloaded from actual files on **130 windows /264,764 targets**.
The final partial window contains572 targets with a60-token CE tail. Native
FP16 head/backbone and FP32 full-vocabulary CE were unchanged; aggregates use
summed NLL divided by the exact target count.

| Model | Complete-validation PPL |
| --- | ---: |
| Original source cast to FP16 |7.334175947318572 |
| Historical accepted all-small |8.359867548009992 |
| Reloaded input-axis/W4-embedding/W5-head primary |8.109428665707021 |

The primary improves **2.9957279%** over the paired accepted baseline, reducing
mean NLL by0.0304151659 nats/target. **127 windows improve and3 worsen**.

| Declared full-validation condition | Outcome |
| --- | --- |
| At least1% lower PPL than paired baseline |**PASS** |
| PPL at most5% above source, maximum7.7008847447 |**FAIL**: actual gap10.5704134% |

The current primary is the lowest measured compressed-candidate PPL so far on
this complete-validation protocol, below the prior prototype's8.3065850617.
It uses more raw bytes and is not a promoted replacement for the existing
distribution. Validation has repeatedly informed development and is not
untouched test evidence; no test/MK result is added.

## Final execution and integrity

Every development and complete-validation arm passed actual507-tensor audits
before and after scoring. The primary's56 input matrices were independently
reloaded for complete validation;57 tensors differ from baseline, with450
inherited tensors fixed. Final restoration reestablished all507 baseline
content hashes, and bound-file identities were rechecked before completion.

The final report is complete, the process receipt records exit code0, and the
recorded evaluator PID was absent at the terminal verification. Internal
evaluation time was **636.2684 seconds**; process wall time **638.4628 seconds**.
Peak allocated GPU memory was **33,617,176,576 bytes** for dense FP16 quality
execution, not compressed runtime residency or isolated throughput.

The improvement does not meet the source+5% target or prove recall retention.
No complete axis-residual container has been built, and the historical accepted
all-small2,671,176,471-byte distribution remains intact. Any subsequent precision
experiment or packing extension is separate. No additional tuning was
performed within this fixed recipe, and publication remains on hold.

## Completed receipts

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
|[Quantization](../reports/input_axis_residual_v1_quantization.json) |168,588 |`75345c3a829ab62ce39023d9f08df4a125aa5229820ca2c5188b88b662b697f5` |
|[Overlay manifest](../reports/input_axis_residual_v1_manifest.json) |168,431 |`0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85` |
|[Quantization process](../reports/input_axis_residual_v1_quantization_process.json) |1179 |`02a6a3652fb49336c850348f146bcd81d7f57308cdfaf886db7a9070e9c6b903` |
|[Builder CPU preflight](../reports/input_axis_residual_v1_cpu.json) |29,233 |`63a0af3dd3d420db90c78c78f9f2de508a0914853c8a595efb99a2f1c84e71eb` |
|[Actual-overlay evaluation preflight](../reports/input_axis_residual_v1_eval_cpu.json) |539,754 |`546300a6ceb9ed51db39c8ef73fbfe1d5fafec00c47204bdf977a93800c4f58f` |
|[Completed native evaluation](../reports/input_axis_residual_v1_eval.json) |3,040,232 |`1dd6be0ddb4481e765da2658926ecea92183521ac9862b914fc3d3ba2a80a7eb` |
|[Evaluation process / exit](../reports/input_axis_residual_v1_eval_process.json) |1152 |`70b2edfa29f7c8b3028e5c5b231187f22fe3202369b07cefb21138e075b76ddd` |

Frozen evaluator SHA256:
`c8dece7ec209b74ef65244086c7c245f08a59ea344124559f964e44c8a2540f0`.
