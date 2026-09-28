# User-directed PPL-only continuation

The user now explicitly prioritizes PPL research and plans to address MK later
with Resurface. This authorizes an additional full-validation comparison of
the fixed norm candidate, even though the earlier combined development gate
failed. It does not alter training, select another checkpoint, reverse the MK
result or authorize publication.

The original receipt remains immutable:
`reports/norm_compensation_v1_eval.json`, SHA-256
`72350eaa1989601f4412ba9e31c178e22dba95ec0b85f5837f9f08b29b265dad`.
It records PPL improvement from 9.2371639507 to 8.9653964898 and normal MK
decline from 10/12 to 9/12, with target-removed controls remaining 0/12.
That failed combined-gate conclusion remains part of the evidence.

## Fixed continuation scope

- New script: `scripts/evaluate_norm_ppl_continuation.py`.
- New receipt: `reports/norm_compensation_v1_ppl_continuation.json`.
- Same process, fixed arm order: original FP16 source, original two-sweep
  E8/W5 parent, final `norm_compensation_v1` FP16 export.
- Complete WikiText-2 validation: exactly 264,764 scored next-token targets,
  130 windows with at most 2048 targets each, fresh zero state per window.
- Unchanged `evaluation.py` math: native prefill, FP16 head GEMM, FP32
  cross-entropy, logits chunks of 64 tokens, no automatic BOS/EOS tokens.
- Reverify all artifact/training/checkpoint identities; audit the loaded
  parent and reload the actual norm export. Preserve all paired per-window
  losses, input hashes, dataset revision and token-stream identity.
- Carry forward the prior MK evidence explicitly; MK is not rerun or used as
  a blocking criterion in this PPL-only continuation.

Source, original parent, trained candidate and prior reports are unchanged.
The original source checkpoint is a declared reference input, not a candidate
inference dependency. No test split, new training or new compression recipe
is used. Validation has already been consulted during prior diagnosis and
development; it must not be described as untouched test evidence.

Report exact full-validation PPL for all three arms, norm-versus-parent change,
each quantized arm's source-relative gap, and the separate +5% source-PPL
reference. A PPL improvement does not establish recall preservation or repair
the earlier MK regression. No publication follows automatically.
