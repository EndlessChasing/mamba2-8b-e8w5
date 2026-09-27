# Frozen evaluation protocol v1

This protocol is specified before any compressed 8B model is evaluated. It separates numerical implementation checks, development measurements, and final held-out measurements.

## Source and tokenization

- Official NVIDIA source revision `b915550c63ba9359f88f44d1f6a600d85af27302`; exact checkpoint and tokenizer hashes in `reports/source_model.json`.
- Native SentencePiece `encode_as_ids` / `decode_ids`, without automatic BOS/EOS.
- The baseline casts the source checkpoint to FP16. The quantized candidate is decoded to FP16 for this accuracy comparison. Both run the same public `mamba_ssm` implementation, eight SSM groups, grouped gated RMSNorm, FP16 residual accumulation (the source checkpoint has `fp32_residual_connection=False`), and independent input/output vocabulary weights.
- FP16 quality evaluation expands the quantized weights. Its allocated GPU memory is **not** compressed inference residency.

## Calibration and development

Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Join records with two newlines. Store text/token digests and selected window offsets; do not redistribute dataset text.

Calibration uses only the train split: 32 evenly spaced, disjoint windows of 2,048 tokens, totalling 65,536 tokens. Record all 112 FP32 projection-input second-moment matrices. TF32 is disabled. The initial fixed E8 configuration is scale 0.9, damping 0.01, two tuning sweeps; each vocabulary uses W5, group 128.

Development uses four fixed, nonoverlapping validation windows of 1,024 next-token targets. Public synthetic recall uses the validation generator seed, two samples for each combination of three templates and record counts 16/64: 12 normal prompts and 12 target-removed controls. It is a smoke screen, not a final quality result.

## Held-out measurements

1. **Full prose PPL:** the complete WikiText-2 raw test token stream, in windows of at most 2,048 next-token targets, with state reset at each window. Adjacent windows share only their boundary input token, so each next-token target is scored once. No test tokens tune quantization or scales.
2. **Public MK v1:** test seed and key/value ranges disjoint from development, eight samples per combination of three templates and record counts 16/64: 48 normal prompts and 48 controls. Query positions span each prompt. Greedy generation, at most 12 tokens; match the first standalone six-digit integer to the target value. Report per-cell and overall accuracy plus target-removed matches.
3. **Tokenwise check:** separately run at least the fixed development windows and development MK suite with FP16 convolution and SSM cache written after every input token. Record the exact extent of this check. A prefill SSD scan is not evidence of per-token FP16-state behavior.

The public MK task is newly defined here. Its scores are not interchangeable with earlier private Resurface multi-key recall results. Report sample counts and paired outcomes; 48 normal examples are too few to establish a narrow statistical equivalence claim.

## Acceptance and reporting

- Require complete parameter coverage, finite outputs, correct grouped normalization, separate vocabulary matrices, exact E8/W5 serialized reconstruction, and full raw-to-container-to-raw SHA256 equality.
- Report baseline and quantized PPL, relative PPL change, each window, recall successes/counts, and change in percentage points. Show any adverse outcome. The compact release may be marked experimental if quality is poor; passing codec tests does not establish model quality.
- A provisional quality target is aggregate PPL degradation at most 5% and no more than one fewer correct answer among 48 normal MK examples, with no increased target-removed matches. These are engineering targets, not statistical proof of equivalence; actual results and uncertainty take precedence.
- The first candidate is fixed before held-out evaluation. If a held-out failure motivates another candidate, disclose that the test set has been consulted, retain prior results, and do not describe a repeatedly selected result as untouched held-out evidence.
- Include source, tokenizer, code, calibration and compressed-package hashes, execution mode, dataset revision, software versions, peak allocated/reserved GPU memory, state bytes where applicable, and full file/container size accounting.
- A size comparison must identify the exact same-base releases audited and include every required model payload, codebook, scale, offset, table, configuration and tokenizer. No global-smallest claim follows from a limited catalog search.
