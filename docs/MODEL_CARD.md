# Mamba-2 8B E8/axis + W4/W5 + soft Resurface

Release: **`v0.2.0-resurface` — research prerelease**. The model has completed
paired PPL and independent numeric-recall confirmation. Publication is authorized;
release packaging and public availability are tracked in
[RELEASE_READINESS.md](RELEASE_READINESS.md).

This is an independently modified version of
[NVIDIA's pure Mamba-2 8B base model](https://huggingface.co/nvidia/mamba2-8b-3t-4k).
It is a base language model, not an instruction-tuned assistant. The release
preserves 56 Mamba-2 blocks, width 4096, eight SSM groups, a 256000-token vocabulary,
and separate embedding and output weights. There are **8,236,999,680 base
parameters**, plus **1,154,104 trained adapter parameters**.

## Representation and adaptation

| Component | Released representation |
| --- | --- |
| 112 input/output projection matrices | Rotated E8P12 plus axis residual; 20 code bits per eight values (nominal 2.5 bits/value), with separately counted transforms/scales/headers |
| Embedding | Group-128 W4 with FP16 scales |
| Output vocabulary head | Group-128 W5 with FP16 scales |
| 393 remaining base tensors | FP16; readapted for this compressed base |
| 224 adapter tensors | FP16 soft gated readout at all 56 layers; 1,154,104 parameters |

The adapter is this project's **post-D Resurface-inspired variant**: it operates
between the native D*x addition and grouped gated RMSNorm. It is not an exact
reproduction of the original paper's placement. The same enabled soft sigmoid
policy is used for recall, prose PPL, and inference. There is no task-dependent
bypass, fitted inference threshold, or additional recurrent cache.

The final adapter was trained for 1536 successful updates on 1536 synthetic
numeric-binding examples, with 10,752 answer-token targets and 784,896 prose
regularization targets. The current compressed model served as the prose
teacher. All 507 base tensors were frozen during adapter training; actual export
and final-checkpoint rounding were verified. See the
[training protocol](RESURFACE_READAPTED_TRAINING_PROTOCOL.md) and
[complete results](RESURFACE_READAPTED_RESULTS.md).

Quantization is lossy relative to the original model. This release uses the
existing E8HUF001 container with raw members and **no additional entropy-coding
pass**; wrapping/restoration preserves every encoded file exactly.

## Measured quality

### Complete WikiText-2 validation

| Model | PPL |
| --- | ---: |
| Original NVIDIA weights cast to FP16 | 7.334175947 |
| Compressed/readapted base without adapter | 7.622396588 |
| Released candidate with soft adapter | **7.593163114** |

The adapter improves PPL **0.383521%** over its paired compressed base, with all
130 windows improving. Candidate PPL remains **3.531237% above original FP16**.
The source-relative +5% engineering target is met; this does not establish equal
quality.

The protocol scores **264,764 next-token targets** in 130 reset windows, including
the short final window. Current/adapter/restored runs were paired in one process;
current and restored CE chunks match exactly. The source value is from the pinned
original-source evaluation with identical dataset, tokenizer, window and token
identities, not a new same-process source/adapter comparison. WikiText-2 validation
informed development and is **not untouched test evidence**.

[Full PPL report](../reports/resurface_soft_continuation_v1_full_eval.json);
[independent arithmetic/integrity receipt audit](../reports/resurface_soft_continuation_v1_full_independent_audit.json).

### Independent numeric-binding confirmation

| Model on the same CONFIRM set | Normal recall | Target-removed false matches |
| --- | ---: | ---: |
| Compressed/readapted base | 91/384 (23.6979%) | 0/384 |
| Same base with soft adapter | **340/384 (88.5417%)** | **0/384** |

The paired improvement is **64.84375 percentage points**: 251 gained cases and
two lost cases. The conservative 95% lower bound is +58.502343 points; exact
McNemar p=4.439957888196715e-72. CONFIRM was prepared only after full PPL passed,
using separate numeric instances under the frozen split contract. It shares the
three development template families and covers 16 or 64 records per prompt.
The original uncompressed source was **not evaluated on CONFIRM**.

[Confirmation report](../reports/resurface_soft_continuation_v1_confirm_eval.json);
[independent confirmation audit](../reports/resurface_soft_continuation_v1_confirm_independent_audit.json).

### Observed development comparison with the original source

On the same 384 DEV normal prompts, original FP16 scores **167/384 (43.4896%)**
and the repaired compressed candidate scores **346/384 (90.1042%)**; both have
0/384 target-removed matches. This is observed development evidence, separate
from CONFIRM. The candidate received recall training while the original-source
baseline did not: the comparison is not evidence that compression itself improves
recall or that training budgets are equal.

[Source baseline](../reports/readapted_mk_baseline_v1.json);
[DEV result with preserved strict failure](../reports/resurface_readapted_v1_dev_eval.json).

### Reproducibility qualification

The initial strict DEV run failed historical cross-process MK replay: seven
generated sequences differed and compressed-base correctness changed 111→112/384,
despite identical input and weight hashes. Its failure remains preserved.
The separately declared [continuation protocol](RESURFACE_SOFT_CONTINUATION_PROTOCOL.md)
removed only that historical-equality prerequisite, after the discrepancy was
observed. It kept the same candidate, soft policy, numerical profile and paired
quality thresholds. Same-process restored MK/PPL controls, fresh-cache replays,
507 base tensors, 224 adapter tensors and final file checks passed. Prospective
full PPL and previously unopened CONFIRM then passed. The cause of the historical
sequence differences remains unresolved.

## Storage and execution

| Measured scope before outer packaging | Bytes |
| --- | ---: |
| Encoded base model data | 3,138,928,792 |
| Serialized FP16 adapter file | 2,539,647 |
| **Base plus adapter data** | **3,141,468,439** |
| Adapter tensor payload within that file | 2,308,208 |
| Separate adapter manifest | 294,171 |

Base plus adapter data are **80.930748% smaller** than the original base's
16,473,999,360-byte 16-bit tensor payload. This compares tensor/model-data scopes,
not complete distribution sizes. Outer manifests, tokenizer, software, licenses
and reports are additional; the completed release's `release_manifest.json`
provides the authoritative total and hashes. The older 2.671 GB all-small bundle
contains a different model and must not be used as this release's size.

The reference runtime expands weights to FP16. Batch-one native FP16 convolution
and SSM cache in the measured recall runs is 122,028,032 bytes (116.375 MiB), before
weights, activations and other runtime allocations. Compact files do **not** mean
3 GB GPU residency. There is no compressed GPU matrix kernel, measured ASIC
energy/throughput result, or global-smallest/equal-quality claim here.

The original checkpoint is BF16; the reported comparison uses this project's
FP16 reference runtime. Original Megatron end-to-end numerical parity is not
established. Numeric binding recall does not establish broad task performance,
unseen-template recall, longer-context retention, or arbitrary-prompt reliability.

## Availability and provenance

The release is prepared for
[`v0.2.0-resurface`](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface).
Use the [download and verification instructions](DOWNLOAD.md) after the tag is
public. A repository checkout alone is not a downloaded model. Release assembly,
restore and archived-software checks are recorded in
[RESURFACE_RELEASE.md](RESURFACE_RELEASE.md).

The official model and tokenizer identities are pinned in [SOURCES.md](SOURCES.md).
The final adapter file SHA-256 is
`1e9857feaf80ede6525cce839726883f6230c98a067cb226eb2f095693ea803e`.
The same identity is bound to both full-PPL and CONFIRM reports. No original
full-precision checkpoint is required to use a complete restored release.

## Attribution and licenses

NVIDIA's upstream model and tokenizer are declared **Apache-2.0**; the pinned model
card, attribution and license text are retained. This is an independently modified
model without NVIDIA endorsement.

Repository software and the QuIP#-derived E8/LDLQ implementation are **GPL-3.0**;
matching source and notices accompany the distribution. Model and software scopes
are described separately in [THIRD_PARTY.md](../licenses/THIRD_PARTY.md). This
release's new public adapter implementation and trained artifact are distinct
from excluded private Resurface code, checkpoints and datasets. No Quamba
implementation is distributed. WikiText text and tokenized passages are fetched
separately under upstream terms and are not included in the model distribution.
