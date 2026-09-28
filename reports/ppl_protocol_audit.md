# Independent PPL protocol audit: first 8B candidate

Scope: read-only analysis of the existing source, calibration, quantization and
evaluation receipts. No new model inference or candidate selection was performed
for this audit. Publication is paused while the PPL regression is investigated.

## Finding

The recorded **7.244528 → 8.708514 PPL (+20.208155%)** is a broad, paired
regression under the stated FP16 reference protocol. No asymmetric tokenizer,
dataset, source mapping, residual configuration, execution mode or scoring
mismatch was found. The evidence supports investigating quantization error first;
it does not yet identify E8 versus W5, a particular layer, or a coding defect as
the dominant cause.

The exact candidate is raw manifest
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.

## Paired protocol and source checks

| Possible cause | Evidence checked | Conclusion and remaining limit |
| --- | --- | --- |
| Different evaluation code/environment | Full reports have identical runtime/evaluation SHA, Python/package/CUDA versions, GPU name and TF32-matmul setting | No recorded asymmetry; shared-GPU timings are irrelevant to this accuracy comparison |
| Different text, token IDs or special tokens | Dataset metadata, fingerprint, text SHA, entire token-stream SHA, tokenizer SHA and all 147 individual window hashes match | Ruled out for the recorded comparison; both use native SentencePiece with no automatic BOS/EOS |
| Off-by-one labels, repeated targets or incorrect aggregate PPL | Evaluator scores `tokens[1:]` from `tokens[:-1]`; starts are `i*2048`; 146 full windows plus one 1955-target window sum to 300963 targets, exactly total tokens minus one | Each next-token target is scored once; aggregate PPL is `exp(sum(NLL)/sum(targets))`, not an arithmetic mean of window PPL |
| Accidental state carry between windows | Prefill PPL calls the backbone without inference parameters for each independent window | Every window starts from zero state for both models |
| Wrong architecture or tied vocabulary | Strict source state loading; 507 tensors / 8236999680 parameters; in-projection 18560×4096, out-projection 4096×8192; eight SSM groups; gated RMSNorm group size 1024; two independent 256000×4096 vocabulary tensors | Supported by tensor inventory, runtime checks and source smoke receipt; shape coverage alone is not full Megatron numerical parity |
| Inconsistent residual precision | Source arguments specify `fp32_residual_connection=False`; frozen runtime uses `residual_in_fp32=False`; calibration and both full evaluations bind to that same runtime SHA | No FP32/FP16 residual mismatch between baseline and candidate. The earlier FP32-residual smoke is not the evaluated baseline |
| Different source model | Full source checkpoint SHA and tokenizer SHA match in baseline, calibration and quantized package; candidate receipt binds all component hashes | No source/package identity mismatch found |
| Corrupt E8/W5 serialization or missing parameters | Independent decoded-weight audit covers 112 E8 projections, two W5 vocabularies and 393 retained FP16 tensors, all 507 tensors | Serialization matches the intended quantized weights. It does not prove that the quantizer math is optimal or correct |
| Huffman coding | Full quality reports evaluated `artifacts/e8w5_v1` directly, before entropy packing | Huffman cannot explain this measured PPL regression |
| Nonfinite output/loss | Evaluator rejects nonfinite hidden states, logits or cumulative NLL; both reports are complete | Catastrophic NaN/Inf failure was not silently averaged into PPL |
| Native NVIDIA BF16 parity | Original tensors are BF16; both evaluation paths use FP16 weights and residuals in the same public runtime | Original Megatron end-to-end and full-test BF16 quality parity remain unmeasured. A later four-window BF16/FP16 control is recorded below; no upstream benchmark equivalence is claimed |

Code inspected: [runtime](../mamba_e8w5/runtime.py),
[evaluation](../mamba_e8w5/evaluation.py),
[calibration](../mamba_e8w5/calibration.py),
[comparison](../mamba_e8w5/compare.py),
[quantizer](../mamba_e8w5/quantize.py).
Source receipts: [arguments](source_args_audit.json),
[tensor inventory](source_tensor_inventory.json),
[source smoke](runtime_source_smoke.json),
[decode audit](decoded_weight_audit.json).

## Distribution across all 147 test windows

All **147/147 windows worsen**; none is within the provisional +5% PPL target.
The aggregate excess loss is **0.184054679 nats per target token**.

| Statistic | Window PPL increase |
| --- | ---: |
| Minimum | 10.604926% |
| 5th percentile | 13.071967% |
| 25th percentile | 15.808032% |
| Median | 19.260382% |
| 75th percentile | 23.613023% |
| 95th percentile | 31.076054% |
| Maximum | 49.937694% |

Percentiles use linear interpolation at `(window_count - 1) * p`.
80 windows increase 10–20%, 57 increase 20–30%, nine increase 30–40%, and one
increases at least 40%. The worst window is zero-based index 59, token offset
120832: PPL 4.984497 → 7.473640.

The five windows with the largest excess NLL contribute only **5.730249%** of
total excess NLL. As a diagnostic, excluding those five leaves 142 windows and
290723 targets, with PPL **7.282037 → 8.714862 (+19.676164%)**. This exclusion is
post hoc and is not a replacement quality score; it shows that a handful of
outlier passages do not explain the regression.

### Dataset position and sequence length

| Consecutive third of test windows | Targets | Baseline PPL | Candidate PPL | Increase |
| --- | ---: | ---: | ---: | ---: |
| Windows 0–48 | 100352 | 7.386203 | 8.765699 | 18.676659% |
| Windows 49–97 | 100352 | 7.180549 | 8.662415 | 20.637228% |
| Windows 98–146 | 100259 | 7.168813 | 8.697734 | 21.327393% |

Pearson correlation of window index with excess NLL/token is **0.146869**;
correlation of baseline NLL/token with excess NLL/token is **0.005752**.
These are descriptive, unadjusted correlations, not causal tests. The state
resets at each window, so dataset position is not recurrent-state age.

146 windows have exactly 2048 targets. The sole shorter window has 1955 targets
and rises **26.963626%** (7.114417 → 9.032721). This does not supply enough length
variation for a length-effect estimate. Existing reports store one summed NLL per
window; **within-window token-position effects cannot be recovered** from them.
Do not infer long-context instability or early/late-token concentration from
these aggregates. Optional follow-up diagnostics can record NLL by 64/128-token
bins on fixed validation windows.

## Prefill versus per-token state rounding

The existing four-window validation check uses the same 4096 targets in both
execution modes:

| Mode | Baseline PPL | E8/W5 PPL | Relative increase | Excess NLL/token |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 7.857624 | 9.237664 | 17.563073% | 0.161804795 |
| FP16 cache written each token | 7.857486 | 9.238848 | 17.580209% | 0.161950547 |

The degradation gap changes by only **0.000145752 nats/token**, about **0.090079%**
of the prefill degradation gap. Thus cache/scan rounding is not a plausible
dominant explanation on this measured validation subset. Candidate tokenwise
evaluation used the native-step CUDA graph; source-model graph parity checks
cover hidden outputs, all 112 cache tensors, reset behavior and one greedy prompt.
This is not full-test tokenwise evidence and does not establish parity for every
possible candidate input.

### Follow-up BF16/FP16 source control

The independently run [precision control](ppl_runtime_precision_dev.json) repeats
the same four validation windows with original source weights in the same public
runtime. Source BF16 PPL is **7.855849780**; source FP16 PPL is
**7.857623554**, exactly reproducing the earlier FP16 endpoint. The FP16 change
is **+0.022579%**, or **0.000225765 nats/token**. This source precision choice
does not explain the much larger measured quantization regression on these
windows. This control covers prefill execution and 4096 validation targets, not
the original Megatron runtime or full-test/per-token BF16 behavior.

## Quantization hypotheses still open

Calibration contains 65536 disjoint training tokens, all 112 projection-input
second-moment matrices, and identical per-hook token counts. It used the original
model's activations. Source/calibration/runtime hashes match the candidate. There
is no evidence that held-out test data entered this initial calibration.

The quantization ledger already shows substantial local reconstruction error:

| Component | Median relative weight norm error | Median calibration-output RMS error | Maximum calibration-output RMS error |
| --- | ---: | ---: | ---: |
| 56 input projections | 37.481253% | 14.636018% | 16.334838% |
| 56 output projections | 42.559695% | 19.963998% | 22.993794% |

W5 vocabulary relative weight errors are **5.547564%** for embedding and
**6.620090%** for the output head. These metrics have different roles; neither
raw weight error nor the local Hessian proxy directly allocates PPL blame. The
calibration-output metric is `sqrt(tr(ΔW H ΔWᵀ) / tr(W H Wᵀ))`, measured with
original-model activations, not with the accumulated quantized model state.

The leading hypothesis is accumulated lossy quantization error, including possible
interaction between W5 vocabulary noise and E8 projection noise. This is an
inference, pending component ablations and the independent codec math audit.
The following remain open:

1. E8 projection-only versus W5 vocabulary-only contributions and their interaction.
2. Sensitivity of the heterogeneous input-projection branches (gate, x, B, C, dt),
   individual layers and the output vocabulary.
3. Whether damping, scale, rotation, diagonal balancing or LDLQ implementation
   choices leave avoidable error; exact disk decoding cannot answer this.
4. Drift from original-model calibration activations as preceding layers become
   quantized; no layerwise recalibration or quantization-aware fine-tuning was used.
5. Context-length and within-window position dependence, which the current report
   granularity does not measure.

Run diagnosis on fixed validation windows first. The original full test result
must remain visible; choosing subsequent candidates after inspecting it makes
that test a consulted test set, not untouched evidence.

## Reproducibility

The [stdlib audit script](../scripts/audit_ppl_protocol.py) recomputes the full-test
aggregate, interpolated percentiles, thirds, correlations and exclusion diagnostic:

```sh
python3 scripts/audit_ppl_protocol.py \
  --baseline reports/baseline_full_prefill.json \
  --candidate reports/e8w5_full_prefill.json
```

For paired window `i`, compute `delta_i = candidate.nll - baseline.nll` and
`relative_i = candidate.ppl / baseline.ppl - 1`. Every aggregate above uses the
summed NLL and target counts of its selected windows. No text decoding, new
dataset download or additional forward pass is needed to reproduce the analysis.

| Input | SHA-256 |
| --- | --- |
| `baseline_full_prefill.json` | `5eb9ab7e72aa8efa558216e82b244508bf23aff4f3406ab03e6fd492c6f3677b` |
| `e8w5_full_prefill.json` | `537588d2d4d55170c9f8846169236cae710bb0eb1f8bb2bb996644c3363dde73` |
| `comparison_full_prefill.json` | `cbc6bfe274ea45dc0385c1303cbde79e4695361d071eea86ca8bac2553746715` |
| `calibration_v1_manifest.json` | `70f2993a00bc441b4dd340cc42dbee3262bd24a8547c665a1f1b236cd276afab` |
| `quantization_v1.json` | `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed` |
| `decoded_weight_audit.json` | `0ff1bc0b9a1399bf7042c764bae31e9af169d6e92b341ee4b3ec921c7b3ad04e` |
| frozen `runtime.py` | `bc31a8d898f78e6be59d74e34d3948166d217a8d36b1817b819bde1f32a0d369` |
| frozen `evaluation.py` | `ff4811cf74d98302c5712d0968f683e840c3a50af2a0dcad9a07628362b45089` |
