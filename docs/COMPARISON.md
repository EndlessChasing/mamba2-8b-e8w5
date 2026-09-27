# Public size comparison and smallest-model claim

Audit date: 2026-09-27. Target: a compact, reproducible, openly released quantization of the complete pure NVIDIA Mamba-2 8B base model. The release preserves the 56-layer architecture and full vocabulary. A smallest-model claim is a goal until actual packing and quality checks complete.

## Same-base inventory

The table records weight-file byte counts from publisher Hugging Face LFS metadata. It does not substitute for download verification, resident GPU memory measurement, or evaluation under the same tokenizer and protocol.

| Artifact | Exact weight-file bytes | Decimal GB | Evidence |
| --- | ---: | ---: | --- |
| NVIDIA original Megatron checkpoint | 16,474,189,490 | 16.474 | [Publisher](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302) |
| Quamba2 8B W4A8 | 4,245,725,690 | 4.246 | [Publisher](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a8/tree/813a7a8815a2bdbfcdbec1f0550e0beec90dd187) |
| Quamba2 8B W4AX | 4,247,971,564 | 4.248 | [Publisher](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4aX/tree/6d03f97e4b77f2688965653e517bde480a7c88ef) |
| UniQL 8B W4A16 | 4,253,398,040 | 4.253 | [Publisher](https://huggingface.co/ut-enyac/mamba2-8b-converted-uniql-1.0-masked-lora-rft-w4a16/tree/220a411ffc885f1543fc19afcfe90b793fa61153) |
| Quamba2 8B W4A16 | 4,253,601,410 | 4.254 | [Publisher](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a16/tree/997f760f29c10574ee8363b8680d636048dee048) |
| This project E8/W5 | Pending measured output | Pending | No completed 8B size or quality claim in this audit |

Each listed competitor also includes the original 4,573,028-byte SentencePiece model and smaller configuration/license files. The original checkpoint is a serialized Megatron container, so its total byte count includes format overhead. Competitor byte identities and source revisions are recorded in [source_model.json](../reports/source_model.json).

Quamba reports W4A8/W4A16/W4AX support and measured GPU inference in its [paper](https://arxiv.org/abs/2503.22879) and [official project page](https://hychiang.info/projects/quamba2/). Those reported results establish neither this project's quality nor its speed. A smaller E8 package could still require more decode work or more runtime memory.

## Search scope and limits

The audit searched Hugging Face's public model API for `mamba2-8b`, the `ut-enyac` 8B releases, model cards linked to the exact NVIDIA source, and targeted web searches for Mamba2 8B 2-bit, 3-bit, GGUF, and quantization releases. This finite search found the above directly comparable low-bit releases. It does not establish that no smaller artifact exists anywhere.

Codestral Mamba, Nemotron hybrids, Mamba/Llama distillations, Bamba, and 2.7B Mamba models are different bases and are not used to assert this exact model's smallest size. Pruned networks and deltas that require an external base must be reported separately with all dependencies counted.

The Quamba model repositories include a `license.txt` with UT Austin Research License restrictions even where a Hugging Face metadata tag says Apache-2.0. They are public comparison artifacts; this report does not classify those implementations as unrestricted open-source releases. This project starts from the NVIDIA Apache-2.0 checkpoint and uses the GPL-3.0 E8 implementation lineage.

## Gates for a publishable size result

- [x] Freeze exact official source model and tokenizer identity.
- [x] Enumerate publicly accessible same-base comparison artifacts.
- [x] Verify downloaded source bytes and tensor inventory.
- [ ] Complete quantization of every intended projection and both vocabulary matrices.
- [ ] Pack and read back every quantized tensor; publish all stored byte counts and SHA-256 manifests.
- [ ] Report original FP16-cast baseline and E8/W5 quality under a fixed tokenizer and PPL/MK protocol; identify comparison artifacts whose quality was not remeasured.
- [ ] Measure peak runtime memory and decode behavior separately from archive size.
- [ ] Recheck the public inventory on the release date.

Suitable language after these gates: “Smallest weight archive among the enumerated pure NVIDIA Mamba-2 8B releases checked on [date], at [bytes], with the quality and runtime results below.” An unqualified worldwide smallest or equal-quality claim would require stronger evidence.
