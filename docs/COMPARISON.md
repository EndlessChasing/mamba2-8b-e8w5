# Public size comparison and smallest-model claim

Public inventory audit date: 2026-09-27. Target: a compact, reproducible, openly released quantization of the complete pure NVIDIA Mamba-2 8B base model. The candidate preserves the 56-layer architecture and full vocabulary. Its local size, restoration and quality are now measured, but model publication is on hold and its quality-retention target failed. A global smallest-model claim remains unestablished.

## Same-base inventory

Competitor rows record weight-file byte counts from publisher Hugging Face LFS metadata at pinned revisions. This project's row records its verified local Huffman container. These are different storage formats; the table does not substitute for download verification of competitors, resident GPU memory measurements, or equal-quality evaluation under the same tokenizer and protocol.

| Artifact | Exact weight-file bytes | Decimal GB | Evidence |
| --- | ---: | ---: | --- |
| NVIDIA original Megatron checkpoint | 16,474,189,490 | 16.474 | [Publisher](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302) |
| Quamba2 8B W4A8 | 4,245,725,690 | 4.246 | [Publisher](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a8/tree/813a7a8815a2bdbfcdbec1f0550e0beec90dd187) |
| Quamba2 8B W4AX | 4,247,971,564 | 4.248 | [Publisher](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4aX/tree/6d03f97e4b77f2688965653e517bde480a7c88ef) |
| UniQL 8B W4A16 | 4,253,398,040 | 4.253 | [Publisher](https://huggingface.co/ut-enyac/mamba2-8b-converted-uniql-1.0-masked-lora-rft-w4a16/tree/220a411ffc885f1543fc19afcfe90b793fa61153) |
| Quamba2 8B W4A16 | 4,253,601,410 | 4.254 | [Publisher](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a16/tree/997f760f29c10574ee8363b8680d636048dee048) |
| This project E8/W5, unpublished two-sweep candidate | 2,660,128,443 | 2.660 | Verified local Huffman weight container; [restore receipt](../reports/release_restore_local_v1.json), [quality results](RESULTS.md) |

Each listed competitor also includes the original 4,573,028-byte SentencePiece model and smaller configuration/license files. The original checkpoint is a serialized Megatron container, so its total byte count includes format overhead. Competitor byte identities and source revisions are recorded in [source_model.json](../reports/source_model.json).

This project's complete prepared distribution is **2,666,260,364 bytes**, including tokenizer, software, licenses, reports and its manifest. The model container passed complete byte-for-byte restoration and generation using its archived software. Its full-test PPL increased **20.208%** and normal public MK decreased from **18/48 to15/48** relative to its original FP16-cast baseline. Competitor quality was not rerun under this exact protocol, and no equal-quality superiority is established. The separate eight-sweep experiment is not included in these counts or results.

Quamba reports W4A8/W4A16/W4AX support and measured GPU inference in its [paper](https://arxiv.org/abs/2503.22879) and [official project page](https://hychiang.info/projects/quamba2/). Those reported results establish neither this project's quality nor its speed. A smaller E8 package could still require more decode work or more runtime memory.

## Search scope and limits

The audit searched Hugging Face's public model API for `mamba2-8b`, the `ut-enyac` 8B releases, model cards linked to the exact NVIDIA source, and targeted web searches for Mamba2 8B 2-bit, 3-bit, GGUF, and quantization releases. This finite search found the above directly comparable low-bit releases. It does not establish that no smaller artifact exists anywhere.

Codestral Mamba, Nemotron hybrids, Mamba/Llama distillations, Bamba, and 2.7B Mamba models are different bases and are not used to assert this exact model's smallest size. Pruned networks and deltas that require an external base must be reported separately with all dependencies counted.

The Quamba model repositories include a `license.txt` with UT Austin Research License restrictions even where a Hugging Face metadata tag says Apache-2.0. They are public comparison artifacts; this report does not classify those implementations as unrestricted open-source releases. This project starts from the NVIDIA Apache-2.0 checkpoint and uses the GPL-3.0 E8 implementation lineage.

## Gates for a publishable size result

- [x] Freeze exact official source model and tokenizer identity.
- [x] Enumerate publicly accessible same-base comparison artifacts.
- [x] Verify downloaded source bytes and tensor inventory.
- [x] Complete quantization of every intended projection and both vocabulary matrices.
- [x] Pack and read back every quantized tensor; retain exact byte counts and SHA-256 manifests locally.
- [x] Report original FP16-cast baseline and E8/W5 quality under a fixed tokenizer and PPL/MK protocol; competitor quality remains unmeasured in that protocol.
- [x] Measure peak runtime memory and decode behavior separately from archive size.
- [ ] Obtain authorization to resume publication, then publish the complete model bytes, explicit quality failure and manifests.
- [ ] Recheck the public inventory on the release date.

Current suitable language: “An unpublished E8/W5 candidate with a verified 2.660 GB weight container and measured quality loss.” After publication and a refreshed inventory, any smallest-among-enumerated statement must specify date, exact artifacts, storage formats, all required bytes, quality differences and runtime limits. An unqualified worldwide smallest or equal-quality claim is not supported.
