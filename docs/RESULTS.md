# First 8B candidate: measured results

Candidate `e8w5_v1`; raw manifest SHA-256:
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.

This first candidate used fixed scale 0.9, damping 0.01 and two LDLQ tuning sweeps, selected before held-out test evaluation. Calibration used 65,536 WikiText-2 train tokens. No fine-tuning, adapter, architecture pruning or tied vocabulary was used.

## Full prefill evaluation

| Metric | FP16 source baseline | E8/W5 | Change |
| --- | ---: | ---: | ---: |
| Full WT2 test PPL | 7.244528 | 8.708514 | +20.208% |
| Public MK normal | 18/48 (37.50%) | 15/48 (31.25%) | -6.25 percentage points |
| Target-removed matches | 0/48 | 0/48 | 0 |

All 300,963 next-token targets were scored once, across 147 windows of at most 2,048 targets, with state reset per window. The worst individual window's PPL increased 49.94%. Of the 48 normal recall cases, 14 were correct for both models, 4 were lost, 1 was gained, and 29 were wrong for both. These small recall counts do not establish statistical equivalence.

The provisional engineering target (PPL increase at most 5%, at most one fewer correct normal recall, no increased target-removed matches) **was not met**. The archive remains an experimental size/quality tradeoff.

Reports: [baseline](../reports/baseline_full_prefill.json), [candidate](../reports/e8w5_full_prefill.json), [paired comparison](../reports/comparison_full_prefill.json).

## Development and state semantics

The separate development prefill screen gave PPL 7.857624 → 9.237664 and normal MK 7/12 → 10/12. Its apparent recall improvement did not reproduce on the larger held-out test. The test results above determine the release description.

The per-token FP16-cache check covers four validation windows and 24 validation recall/control prompts, separately from the full prefill test. It measured PPL **7.857486 → 9.238848**, normal recall **7/12 → 10/12**, and target-removed matches **0/12 → 0/12**. The candidate used the verified native-step CUDA graph implementation; the baseline used the plain native token loop. See [paired tokenwise results](../reports/comparison_dev_tokenwise.json). CUDA graph implementation checks verified bitwise hidden and all 112 cache-tensor equality over 128 source-model tokens, repeat-sequence reset, and identical greedy outputs on a 263-token prompt. This does not extend the measured per-token test scope to the full test split.

## Integrity and memory

- Official source and native tokenizer match the publisher's pinned SHA-256 values.
- All 507 loaded tensors and all 8,236,999,680 parameters passed the [independent decode audit](../reports/decoded_weight_audit.json).
- 112 E8 projections match the quantizer's FP16 result hashes. Both complete 256,000-by-4096 W5 vocabularies match an independent CPU bit decoder. The remaining 393 FP16 tensors match the stored values.
- Raw data files total 2,886,482,462 bytes. This excludes the raw manifest and separately shipped tokenizer/metadata. Use the final release manifest for complete entropy-coded storage accounting.
- The verified local Huffman container is 2,660,128,443 bytes; the immutable prepared distribution is 2,666,260,364 bytes including its manifest. It remains unpublished. The [actual restore](../reports/release_restore_local_v1.json) reproduced all118 files, and [generation with archived software](../reports/restored_inference_local_v1.json) used only restored model/tokenizer inputs.
- Full prefill evaluation: baseline peak allocated 17,076,597,760 B; candidate 17,355,670,016 B allocated and 18,599,641,088 B reserved. The reference fully expands quantized weights to FP16.
- Batch-one FP16 convolution plus SSM state: 122,028,032 B (116.375 MiB).
- The original checkpoint is BF16; this comparison casts it to FP16. The complete original Megatron runtime was not executed. Native BF16 quality equivalence, compressed GPU residency, ASIC power and energy, and global minimum model size are not established.

The corresponding code, calibration hashes, environment versions, dataset/tokenizer identities and quality reports are retained for reproduction. No benchmark timing here is an isolated performance claim.

## Subsequent diagnosis and experiments

The [component diagnosis](PPL_DIAGNOSIS.md) locates the dominant loss in E8 projections: entire-validation PPL is 7.334168 for original, 8.729770 for E8-only, 7.415406 for W5-only and 8.836527 for the full candidate. This does not replace the test results above. A limited BF16/FP16 original-runtime control differs by only0.022579% PPL; original Megatron parity remains unverified.

The [eight-sweep experiment](E8_REFINEMENT_RESULTS.md) generated all112 projections at unchanged raw bitrate. Squared calibration-output errors improved for all112 (median1.8970%), but paired development PPL improved only9.237164→9.194595 (0.460844%); normal MK remained10/12 and controls0/12. This failed the [predeclared](REFINEMENT_PROTOCOL.md)1% PPL advancement threshold. The route is stopped, full validation was skipped, and the original candidate remains unchanged. No refined Huffman size or full-test quality is claimed.
