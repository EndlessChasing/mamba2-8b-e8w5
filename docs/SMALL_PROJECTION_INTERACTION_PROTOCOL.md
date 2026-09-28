# Small-tensor compensation and projection restoration interaction

Declared before this diagnostic's outcomes. This is a read-only, fixed four-arm
intervention on existing weights. No training, checkpoint selection, MK, test
split, export, publication or model mutation is authorized by this protocol.

## Question and fixed arms

Does restoring the original source FP16 projections remain useful after all
393 small tensors have adapted to E8/W5? The embedding and output vocabulary
matrices are **W5 in every arm**, including the projection restoration oracle.

| Arm | 393 small tensors | 112 projections | 2 vocabularies |
|---|---|---|---|
| `original_small_e8` | original source FP16 | parent E8 decoded FP16 | parent W5 decoded FP16 |
| `trained_small_e8` | final all-small FP16 export | parent E8 decoded FP16 | parent W5 decoded FP16 |
| `original_small_source` | original source FP16 | original source FP16 | parent W5 decoded FP16 |
| `trained_small_source` | final all-small FP16 export | original source FP16 | parent W5 decoded FP16 |

Run these four arms in table order, in one process with the frozen native
runtime. Original means the pinned NVIDIA BF16 checkpoint cast to FP16 by the
existing source loader. Source checkpoint SHA-256:
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
Original E8/W5 parent manifest SHA-256:
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
All-small manifest SHA-256:
`edde5734ee447bca45f0e7079287abfe5ef378dd5fd9f0cd57bd7d868bd9a006`.
Completed all-small full-validation receipt SHA-256:
`957de68b6dc8ac3410f9edebc11b3991879a6b3b139f49e55b3ef7f3116c7206`.
The actual small export, training/checkpoint/smoke inputs and frozen code must
match that receipt and the existing strict overlay verifier. Bind this protocol
and diagnostic script hashes in the new receipt before GPU outcomes.

## Parameter and swap integrity

Use two model objects, verified source and verified quantized parent. Retain
their original parameter references; the only additional model-sized state is
the 393-tensor small export, about7.16MB of FP16 tensor values. No third model
or dense trainable parameter copies are needed.

Freshly audit the parent's complete507 tensors /8,236,999,680 parameters:
all112 E8 decoded hashes, independent CPU W5 decoding for both vocabulary
matrices, and all393 stored small tensors. Verify source original-small hashes
against parent small values. Hash all112 actual source FP16 projection tensors.
Apply the strict all-small overlay to the quantized object, verify all393 loaded
values against the actual export, and rehash all114 large E8/W5 tensors to
confirm the overlay did not change them.

Before complete scoring, compare direct quantized objects with the corresponding
reference-swapped source object for both E8 endpoints, original-small and
trained-small. The first128 validation input tokens form a zero-state integrity
probe; require exact FP16 hidden-state and last8-token full-vocabulary-logit
hash equality, finite outputs and exact parameter-reference mappings. This
probe is separate from the complete PPL target accounting.

Each arm must retain W5 embedding/head references and the exact declared112
projection/393-small references, with full507-parameter coverage. Record actual
393 loaded small hashes. After all arms, restore both model objects' original
507 references, verify that restoration, and rehash the frozen parent114 and
source112 tensors. Never change parameter tensor contents in place.

## Frozen full validation

Use exactly the prior receipt's WikiText-2 raw validation dataset, tokenizer and
token-stream hashes:264,765 tokens,264,764 next-token targets,130 contiguous
windows of at most2048 targets, final window572 targets. Each window begins at
zero state. Reuse unchanged `evaluation.py`: prefill SSD computation, FP16 head
GEMM followed by FP32 cross entropy,64-token logit chunks, no automatic special
tokens. Store all130 loss rows for every arm and verify token identity and the
target-weighted aggregate `exp(mean NLL)`.

No other intervention is added after seeing outcomes. Historical same-arm PPL
differences are reported; they are not substituted for current-process arms.

## Conditional effects and interpretation

Let `L(s,p)` be mean NLL, with `s` original/trained small tensors and `p` E8/source
projections. Report:

- Projection restoration at original-small: `L(original,source)-L(original,E8)`.
- Projection restoration at trained-small: `L(trained,source)-L(trained,E8)`.
- Small adaptation under E8: `L(trained,E8)-L(original,E8)`.
- Small adaptation under source projections: `L(trained,source)-L(original,source)`.
- Interaction: the second projection effect minus the first; equivalently,
  the second adaptation effect minus the first.

Negative conditional effects mean lower loss. A positive interaction means
projection restoration removes less loss after adaptation; a negative value
means it removes more. Include all per-window conditional effects, paired PPL
changes and improvement/worsening counts. PPL effects are not additive.

The source-projection arms are diagnostic oracles with FP16 projection storage,
not achieved E8 repairs or deployable same-size candidates. Results may guide a
later teacher target but cannot establish that codebook/codeword optimization
can attain this oracle. Validation has informed development; these are not
untouched test results and say nothing about MK. Record actual GPU allocation
and timings only as diagnostics. Preserve all prior receipts and model files.
