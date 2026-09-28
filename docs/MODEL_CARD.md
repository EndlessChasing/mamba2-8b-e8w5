# Mamba-2 8B E8/W5 — unpublished experimental candidate

**Model publication is on hold.** This is a reviewable model-card draft for the
completed two-sweep candidate `e8w5_v1`. It is not a published download, a claim
of equal quality, or the model card of a successfully evaluated refinement.

**The provisional quality target failed:** full WikiText-2 test PPL increased
**20.21%**, and public multi-key recall decreased **6.25 percentage points**.

## Model and method

This is an independent modified version of
[NVIDIA's pure Mamba-2 8B base language model](https://huggingface.co/nvidia/mamba2-8b-3t-4k).
It preserves 56 layers, model width 4096, eight SSM groups, the full 256000-token
vocabulary, separate embedding/output weights, and all **8236999680 parameters**.
It is not an instruction-tuned model.

- **112 projection matrices:** calibrated diagonal balancing, signed DCT/Hadamard
  rotations and E8P12 lattice quantization with LDLQ feedback. Every eight values
  use a 16-bit index, before transformation metadata. This candidate used scale
  override 0.9, damping 0.01 and two tuning sweeps.
- **Two independent vocabulary matrices:** uniform W5, group 128, with FP16 scales.
- **393 remaining parameter tensors:** FP16.
- **Entropy storage:** chunked Huffman, with tables and offsets counted. Huffman
  restores the quantized files exactly; quantization itself is lossy relative to
  the original floating-point model.

Calibration used 65536 WikiText-2 training tokens. No fine-tuning, residual adapter,
architecture pruning, tied vocabulary, or external weight base is required by
the representation. Source, tokenizer and calibration identities are retained in
the [provenance](SOURCES.md) and [quantization manifest](../reports/quantization_v1.json).

## Measured quality: failed target

| Full-test metric | Original source cast to FP16 | E8/W5 decoded to FP16 | Change |
| --- | ---: | ---: | ---: |
| WikiText-2 PPL | 7.244528 | 8.708514 | +20.208155% |
| Public MK normal prompts | 18/48 (37.50%) | 15/48 (31.25%) | −6.25 percentage points |
| Target-removed controls | 0/48 | 0/48 | No additional matches |

The PPL evaluation scores all **300963 next-token targets**, across 147 windows
of at most 2048 targets with state reset at each window. Both models use the same
tokenizer, dataset identities and runtime. Execution uses SSD prefill; this is
not a full-test per-token FP16-cache experiment.

The declared engineering target was PPL increase at most 5%, at most one fewer
normal recall success, and no increase in target-removed matches. **This candidate
fails the PPL and normal-recall conditions.** All 147 prose windows worsen.
Among normal MK cases, 14 succeed for both models, four are lost and one is
gained. Forty-eight examples do not establish statistical equivalence.

The public MK generator is new for this repository. Its scores are not
interchangeable with private Resurface results or other recall benchmarks.
See the [frozen protocol](EVALUATION.md), [paired full-test report](../reports/comparison_full_prefill.json)
and [detailed results](RESULTS.md).

A separate four-window validation check writes FP16 state after every token:
PPL **7.857486 → 9.238848**, normal MK **7/12 → 10/12**, controls **0/12 → 0/12**.
That small validation recall gain did not reproduce on the larger test and does
not override the test result. [Tokenwise report](../reports/comparison_dev_tokenwise.json).

Component diagnosis on the complete validation split attributes the dominant
measured loss to E8 projections. The [diagnosis](PPL_DIAGNOSIS.md) also records
the small historical-versus-current parent endpoint discrepancy, exact decoded
E8 checks and same-process controls. A later eight-sweep refinement has only
training-screen evidence at this draft's snapshot; it is not included in these
quality or storage claims.

## Actual storage

These counts describe the **existing immutable local two-sweep package**, not a
future release amended with this draft:

| Scope | Exact bytes | Decimal GB |
| --- | ---: | ---: |
| Raw quantized data files, excluding raw manifest | 2886482462 | 2.886482462 |
| Complete raw package, including raw manifest | 2886686334 | 2.886686334 |
| Huffman weight container | 2660128443 | 2.660128443 |
| Entire prepared distribution, including tokenizer, code, licenses, reports and its own manifest | 2666260364 | 2.666260364 |

The weight container is split into **1500000000** and **1160128443** byte parts.
Its size is approximately **2.583590 bits per original parameter** including
container overhead; this is not a claim that all weights individually use 2 bits.
No original model checkpoint is omitted from a required weight dependency: the
container holds all encoded or retained model tensors.

The [complete local restore acceptance](../reports/release_restore_local_v1.json)
verified all **118/118 original raw files** byte-for-byte, plus tokenizer and
configuration. Restoration took approximately 109 seconds on the test host;
this is a local file operation, not inference throughput. The subsequent
[restored generation smoke](../reports/restored_inference_local_v1.json) verified
61 archived source files and ran three short prompts using the restored raw
weights/tokenizer. Outputs were nonempty, and a fresh-cache repeat was identical.
No source-checkpoint argument was supplied; the disabled source loader and Python
open guard recorded zero access attempts. This is not an OS filesystem sandbox
or a model-quality benchmark.

Changes to distribution metadata will change complete distribution bytes and
manifest SHA. The current outer manifest SHA is
`d5f5c77eba397c6706a1b59290f37fd1754728b9e00637aed8a390c1d6922ff1`;
the raw model identity is
`ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.

## Runtime memory and use limitations

The public accuracy runtime expands weights to FP16. Full-test candidate
evaluation peaked at **17355670016 B allocated** and **18599641088 B reserved**
GPU memory. Batch-one FP16 convolution plus SSM cache occupies **122028032 B**
(116.375 MiB), before other runtime memory.

Compact disk storage therefore does not establish compact GPU residency. This
candidate includes no compressed GPU matrix-multiplication kernel, measured
ASIC design, power/energy result or isolated inference-throughput claim. The
original checkpoint is BF16; full test comparisons use FP16 in the public runtime,
and original Megatron end-to-end numerical parity has not been demonstrated.

Use the candidate for compression and reconstruction research with its measured
quality loss visible. Do not treat restoration success or a generation smoke as
quality retention. There is no certified global-smallest or equal-quality claim;
the [public comparison](COMPARISON.md) has a finite, dated search scope.

## Availability and loading

The [public repository](https://github.com/EndlessChasing/mamba2-8b-e8w5) exists,
but model assets remain unpublished while diagnosis and refinement proceed.
[Download instructions](DOWNLOAD.md) describe the planned release workflow, not
an available model download. The prepared package contains matching archived
source and the native tokenizer; [packaging documentation](PACKAGING.md) describes
verification and restoration. Original weights and calibration Hessians are
needed to reproduce quantization, not to decode the completed package.

## Attribution and licenses

NVIDIA's upstream model and tokenizer are declared Apache-2.0; the pinned original
model card, attribution and license are retained. This is a modified model from
an independent project, without NVIDIA endorsement.

The software and QuIP#-derived E8/LDLQ implementation use GPL-3.0, with matching
source and attribution. Model and software scopes are recorded separately in
[THIRD_PARTY.md](../licenses/THIRD_PARTY.md), with [Apache-2.0](../licenses/APACHE-2.0.txt)
and [QuIP# GPL-3.0](../licenses/QUIP-SHARP-GPL-3.0.txt) texts included. No Quamba
implementation or private Resurface code, adapter or dataset is distributed.
WikiText is fetched separately under its upstream terms; dataset text is not
included in the repository or model archive.

This draft has not modified the existing outer model card, manifest, weight
container, or archived source. See [RELEASE_READINESS.md](RELEASE_READINESS.md)
for the remaining publication requirements and hold.
