# Why the first 8B E8/W5 candidate loses PPL quality


**Current PPL-priority result:** the [all-small-tensor experiment](SMALL_TENSOR_COMPENSATION_RESULTS.md) completed 1024 updates and full validation. PPL is **8.35987**, down **2.6750%** from norm-v1 and **5.3946%** from original E8/W5, with raw capacity unchanged. It remains **13.9851% above FP16**, so the +5% source target is unmet. MK is deferred under the user's [PPL-first direction](PPL_PRIORITY_CONTINUATION.md); recall recovery is unmeasured for this candidate. Publication remains on hold.

Model publication is on hold at the user's request. The existing GitHub release
is a draft with zero uploaded assets. The original candidate and its quality
reports are preserved; none of the diagnostic interventions creates a replacement
model package.

**Conclusion:** the main measured quality loss comes from E8's lossy projection
quantization. The complete validation split confirms the four-window diagnosis:
E8-only PPL is 8.729770, close to complete E8/W5 at 8.836527; W5-only PPL is
7.415406 versus original 7.334168. The major input-projection sensitivity lies
in x and z, with a substantial separate output-projection contribution. No
packaging, parameter-swapping or quantizer-math defect explaining the large
regression was found. The tiny cross-run numerical discrepancy remains disclosed.

## Established facts before component attribution

The first candidate's full WikiText-2 test PPL is **7.244528 → 8.708514**,
an increase of **20.208155%** or **0.184054679 nats per target token**.
Lower PPL is better. All 147 test windows worsen. Removing the five windows
with the largest excess loss still leaves a 19.676164% increase, so a few bad
passages do not explain the result.

The paired evaluation uses the same tokenizer, text and token hashes, windows,
runtime, FP16 arithmetic, zero initial state and next-token scoring rule. See
the [protocol audit](../reports/ppl_protocol_audit.md) and the
[reproducible statistics script](../scripts/audit_ppl_protocol.py).

### Causes that do not explain the main regression

| Candidate explanation | Evidence |
| --- | --- |
| Huffman or release packaging | Quality was measured from the raw package before entropy coding; its later container is lossless with respect to those raw bytes. |
| Missing tensors or serialization corruption | An independent full decode audit covers all 507 tensors; E8 decoded hashes, independently unpacked W5 values and retained FP16 values match their intended quantized representations. |
| BF16 source converted to FP16 | On the frozen four validation windows, source BF16 PPL is 7.855850 versus FP16 7.857624, only +0.022579%. This is a public-runtime control, not original Megatron parity. |
| SSD prefill versus per-token FP16 state | On the same validation subset, the excess-loss gap differs by only 0.000145752 nats/token, about 0.090% of the prefill degradation. Full-test tokenwise evaluation is still not established. |
| Incorrect non-power-of-two rotation or LDLQ port | Independent Kronecker/inverse/Hessian/objective/LDL checks pass; the actual 18560-axis inverse error is 4.22e-7; buffered LDLQ matches the upstream unbuffered implementation in the synthetic oracle. |

Receipts: [BF16/FP16 control](../reports/ppl_runtime_precision_dev.json),
[full decode audit](../reports/decoded_weight_audit.json),
[quantizer math audit](QUANTIZER_MATH_AUDIT.md).

## Confirmation on the entire validation split

Four configurations were specified before this confirmation and evaluated on
all **264764 next-token targets**, in **130 windows** of up to 2048 targets,
with state reset at each window. No new training, tuning or test-set use occurred.

| Quantized components | PPL | Change versus original | Excess mean NLL, nats/token |
| --- | ---: | ---: | ---: |
| None | 7.334168 | — | 0 |
| Both W5 vocabulary matrices only | 7.415406 | +1.1077% | 0.011015774 |
| E8 projections only | 8.729770 | +19.0288% | 0.174195102 |
| Complete E8/W5 | 8.836527 | +20.4844% | 0.186350029 |

Introducing E8 costs **0.174195102** nats/token with original vocabularies and
**0.175334255** with W5 vocabularies. Introducing both W5 matrices costs
**0.011015774** with original projections and **0.012154926** with E8 projections.
Thus the E8 dominance is not an artifact of the four-window screen. E8 alone
produces **93.48%** of the complete candidate's excess NLL on the original-vocab
path; effects depend on the background and are not additive PPL contributions.
All three quantized configurations worsen all 130 windows relative to original.

The entire-validation result uses different windows from the four-window screen
below; compare rows within each table. Its +20.48% full-candidate PPL degradation
is also consistent with the previously recorded +20.21% on the separate test
split. The confirmation is diagnostic evidence, not a repaired candidate.

Evidence: [full validation receipt](../reports/ppl_components_full_validation.json)
(SHA-256 `ca6254c2032087236ea642d80fc0b4b40c2fb9ea3d699a84789bf94132418ddd`).
All four cases have identical window hashes, exactly 264764 targets, verified
NLL sums and original parameter references restored at completion.

## Four-window factorial and branch interventions

All eight combinations were measured on the same four frozen validation windows
(4096 targets), in one process. Original and quantized artifacts were unchanged.

| Quantized components | PPL | Excess mean NLL over original, nats/token |
| --- | ---: | ---: |
| None | 7.857624 | 0 |
| E8 projections only | 9.121424 | 0.149141698 |
| W5 embedding only | 7.855142 | -0.000315923 |
| W5 output head only | 7.958227 | 0.012721980 |
| E8 + W5 embedding | 9.124854 | 0.149517646 |
| E8 + W5 output head | 9.234861 | 0.161501322 |
| Both W5 vocabulary matrices | 7.954807 | 0.012292180 |
| Complete E8/W5 | 9.237164 | 0.161750693 |

**The dominant measured cause is lossy E8 projection quantization.** E8 alone
produces 92.20% of the complete candidate's excess NLL on this subset. This is a
comparison of two measured interventions, not an additive attribution rule.
Across all four settings of the other components, introducing E8 increases mean
NLL by 0.14878–0.14983; introducing W5 output-head quantization increases it by
0.01223–0.01272. W5 embedding's effect is very small, between -0.00043 and
+0.00038 nats/token; its slightly favorable isolated score is not evidence of
reliable quality improvement.

### Where the E8 error matters

Each intervention below starts again from the complete quantized candidate.
It restores the indicated weights to original FP16 values and changes nothing
else. Loss recovered means `(NLL_full_Q - NLL_intervention) /
(NLL_full_Q - NLL_original)` on these exact 4096 targets.

| Original weights restored | PPL | Excess NLL recovered |
| --- | ---: | ---: |
| All input projections | 8.294958 | 66.51% |
| All output projections | 8.671019 | 39.10% |
| Input-projection gate z rows | 8.746509 | 33.74% |
| Input-projection x rows | 8.775621 | 31.69% |
| Input-projection B+C rows | 9.183800 | 3.58% |
| Input-projection dt rows | 9.212624 | 1.64% |

These are separate conditional interventions; the percentages **must not be
summed**. B alone recovers 1.95% and C alone 1.63%. The results place the major
error in the broad x/z/output pathways. They do not support a diagnosis in which
only the small dt or B/C parameter slices account for the regression.
Layer-by-layer contributions and nonlinear propagation are not separately
identified by this experiment.

A dt FP16 patch on top of the present full E8 base would add **58,720,256 bytes**;
B+C would add **939,524,096 bytes** before metadata. Their limited quality
recovery does not support using them as the principal repair. Restoring all z or
x rows would add 3,758,096,384 bytes each as an additive FP16 patch, so those
restorations locate sensitivity rather than offer a compact solution.

Raw evidence: [all 16 interventions](../reports/ppl_components_dev.json).

## What E8 actually optimizes

The codec stores a 16-bit E8 index for each eight transformed projection weights:
2 bits/weight before metadata. Its local objective is a Hessian-weighted
projection-output error, approximately `tr(deltaW H deltaW^T)`. It does not
directly optimize next-token language loss through the complete recurrent model.

The original-model calibration-output RMS errors have medians of **14.636%**
for input projections and **19.964%** for output projections. These are not PPL
contributions. They establish that the decoded weights retain substantial local
approximation error despite exact serialization. Downstream nonlinearities,
state updates and input-distribution changes can change how that error affects
language predictions.

The old 2.7B result did not establish FP16-equivalent quality: on its historical
sampled protocol, FP16 / W4 / E8-W5 PPL were **8.324626 / 11.143332 / 10.830269**.
E8-W5 was better than W4 there, while remaining 30.099% above FP16. Those data use
a different model and evaluation protocol from this 8B run and cannot be used
to rank the models. Exact historical sources and hashes are recorded in the
[transfer audit](../reports/quantizer_transfer_metadata_v1.json).

## Diagnostic protocol and numerical reproducibility

The component investigation uses only the already frozen four validation windows
(4096 scored targets). It swaps original and decoded quantized parameter
references without changing either source artifact. All eight combinations of
E8 projections, W5 embedding and W5 output head are evaluated. Follow-up
interventions restore projection families or native input-projection row roles.
Their effects must be compared in mean NLL; PPL differences are not additive.

A strict historical endpoint check initially failed for the quantized model:
saved development PPL **9.237663717**, repeated current PPL **9.237163951**.
The difference is -0.005410% PPL, or -0.0000541024 nats/token. The original-model
endpoint reproduced exactly. The failed receipt is retained instead of being
silently accepted or overwritten.

A fresh probe found exact repeated current-Q results and exact direct-Q versus
parameter-swapped results, including first-window hidden values. All 507
parameter strides match; there are no registered buffers or differing module
scalar attributes. The initial probe checked four E8 hashes; the completed
ablation subsequently checked **all 112 decoded E8 hashes**, all matching the
quantization manifest. Retained small tensors were bitwise equal and sampled
W5 rows also matched an independent CPU unpacker, supplementing the prior full
W5 audit. These
checks exclude the parameter-swapping intervention as the source of the small
historical discrepancy. Its cross-run numerical cause is not yet identified.
The resumed ablation binds its current Q anchor explicitly and passed the strict
within-process comparison. The historical gate remains recorded as nonexact.
The separate full-validation process measured development Q PPL 9.237647508;
its direct-Q and swapped-Q NLLs again matched exactly. This is within the
observed approximately 0.0054% cross-process spread. Historical strict comparisons
are recorded as comparisons only, while within-process equality is mandatory.

See [endpoint probe](../reports/ppl_endpoint_probe.json),
[initial failed gate](../reports/ppl_components_dev_failed_endpoint.json), and
[component intervention driver](../scripts/diagnose_ppl_components.py).

## Limits on subsequent conclusions

These interventions diagnose this candidate; restoring FP16 rows is not itself
a deployable low-bit codec. Because the current E8 left rotation mixes native
rows, a row patch must retain the complete E8 base unless the codec is redesigned.
Count the complete patch and decoding costs in any future size comparison.

The first candidate's full test result has already been inspected. Future
candidate selection must use validation data and disclose that the original
test set is consulted. The diagnosis does not establish a global minimum model
size, a compressed inference-memory result, or original Megatron runtime parity.

## Consequences for the next compression iteration

Keep publication on hold. Improving packaging or exact entropy decoding cannot
repair these lossy projection weights. Increasing only output-head precision
would leave the dominant E8 contribution; protecting only dt/B/C would leave
most of it as well.

A useful next candidate should reduce x/z/output projection error per stored
byte, for example through selective additional precision or residual capacity,
or a training/calibration method that optimizes the complete quantized network.
Those are follow-up hypotheses, not validated repairs. This diagnosis does not
prove that every 2-bit method fails, and no claim is made that a particular repair
will meet the +5% PPL target. Under the user's subsequent
[PPL-first direction](PPL_PRIORITY_CONTINUATION.md), new repair experiments use
their declared PPL gates and report recall as unmeasured. The earlier combined
PPL/recall gates below remain recorded with their original outcomes.

## Subsequent same-bitrate check

The completed [two-versus-eight-sweep experiment](E8_REFINEMENT_RESULTS.md)
improved all 112 projection training-output MSEs (median 1.8970%) but reduced
development PPL only 9.237164→9.194595, a 0.460844% gain. Normal recall stayed 10/12
and target-removed matches 0/12. It failed the predeclared 1% advancement threshold;
full validation was skipped and this fixed route stopped. This supports a small
benefit from extra codeword search, insufficient for the tested repair gate.
It does not establish a lower bound for all 2-bit methods. No training or further
candidate selection was performed in this bounded experiment.

The subsequent [norm-only training experiment](NORM_COMPENSATION_RESULTS.md)
recovered 2.9421% development PPL at unchanged raw size, but lost one of the
parent's ten correct public recall cases. It failed its predeclared combined
gate and stopped before full validation. This shows some prose loss is
compensable with existing parameters; it does not establish preserved recall
or repair the original full-test quality failure.

## Reproduce the diagnosis

Use the pinned source/raw artifacts and existing frozen reports. Choose fresh
output names; the drivers refuse to overwrite their evidence files.

```bash
python3 scripts/audit_ppl_protocol.py \
  --baseline reports/baseline_full_prefill.json \
  --candidate reports/e8w5_full_prefill.json
python scripts/diagnose_ppl_components.py \
  --source-dir models/source --raw-dir artifacts/e8w5_v1 \
  --report reports/ppl_components_dev_rerun.json
python scripts/confirm_ppl_components.py \
  --source-dir models/source --raw-dir artifacts/e8w5_v1 \
  --prior-diagnostic reports/ppl_components_dev_rerun.json \
  --report reports/ppl_components_full_validation_rerun.json
```

The measured GPU peak was 42.30 GB for the factorial/row interventions and
33.76 GB for the four-case confirmation. These figures include original and
decoded quantized models used for diagnosis; they are not compact inference
memory measurements. All GPU diagnostic jobs have completed.
