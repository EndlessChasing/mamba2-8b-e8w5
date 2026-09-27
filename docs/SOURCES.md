# Sources and reproducibility

Audit date: 2026-09-27. These are public source and metadata checks. The download and quality reports separately establish what was actually run.

## Selected base model

- Publisher: [NVIDIA](https://huggingface.co/nvidia/mamba2-8b-3t-4k).
- Repository: `nvidia/mamba2-8b-3t-4k`.
- Pinned revision: `b915550c63ba9359f88f44d1f6a600d85af27302`.
- Architecture: pure Mamba-2 base language model. This is not a Mamba/attention hybrid, an instruction model, or a Llama distillation.
- Training description: 3.5 trillion tokens, 4K sequence length, as stated by the publisher.
- License: Apache-2.0, explicitly declared in the [pinned publisher model card](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md). A verbatim card snapshot is in [licenses](../licenses/NVIDIA_MAMBA2_MODEL_CARD.md).

| File | Exact bytes | Published SHA-256 |
| --- | ---: | --- |
| `release/mp_rank_00/model_optim_rng.pt` | 16,474,189,490 | `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb` |
| `mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model` | 4,573,028 | `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09` |

The byte counts and hashes above come from [Hugging Face LFS metadata](https://huggingface.co/api/models/nvidia/mamba2-8b-3t-4k?blobs=true). The remote download receipt now verifies both full-file SHA-256 values; see [reports/source_download.json](../reports/source_download.json). The machine-readable source manifest is [reports/source_model.json](../reports/source_model.json).

Pinned downloads:

- [Original Megatron checkpoint](https://huggingface.co/nvidia/mamba2-8b-3t-4k/resolve/b915550c63ba9359f88f44d1f6a600d85af27302/release/mp_rank_00/model_optim_rng.pt)
- [Original SentencePiece model](https://huggingface.co/nvidia/mamba2-8b-3t-4k/resolve/b915550c63ba9359f88f44d1f6a600d85af27302/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model)

## Architecture checks

Verified geometry is 56 blocks, width 4096, expansion 2, state dimension 128, head dimension 64, 128 heads, 8 B/C groups, convolution width 4, and separate 256,000-by-4096 embedding/output matrices. The official [tensor inventory](../reports/source_tensor_inventory.json) contains 507 BF16 tensors and 8,236,999,680 parameters.

The checkpoint arguments specify normalization epsilon `1e-5` and `fp32_residual_connection=False`. The reference runtime preserves those settings. Its quality baseline casts BF16 weights to FP16, including FP16 residual accumulation; it is not a native BF16 Megatron parity claim. The public [Megatron Mamba layer](https://github.com/NVIDIA/Megatron-LM/blob/7ee599a25ca15f042c1087176b73ebddb0566d02/megatron/core/ssm/mamba_layer.py) also makes FP32 residual promotion conditional.

NVIDIA's [NeMo conversion documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/mamba/checkpointconversion.html) explicitly distinguishes 8 groups for NVIDIA's 8B model from 1 group in the original state-spaces 130M through 2.7B series. The public [HF conversion config](https://huggingface.co/ib-ssm/mamba2-8b-3t-4k-hf/blob/2b35ece89bfc2d66327d771c25f054bc1dc0f73a/config.json) independently records the remaining geometry. That conversion is a metadata cross-check, not this project's weight source.

The original checkpoint schema maps onto the state-spaces API by renaming these namespaces and preserving the tensor contents:

| Original namespace | Runtime namespace |
| --- | --- |
| `embedding.word_embeddings.weight` | `backbone.embedding.weight` |
| `decoder.final_norm.weight` | `backbone.norm_f.weight` |
| `output_layer.weight` | `lm_head.weight` |
| `decoder.layers.<i>.*` | `backbone.layers.<i>.*` |

Loading must reject unrecognized tensor keys, missing keys, and shape mismatches. Merely finding a same-shaped model does not establish numerical parity. A full-precision baseline and recurrent-versus-prefill checks remain necessary.

## Tokenizer

Use the original SentencePiece model with native `encode_as_ids` / `decode_ids`. The NVIDIA [GPTSentencePieceTokenizer implementation](https://github.com/NVIDIA/Megatron-LM/blob/7ee599a25ca15f042c1087176b73ebddb0566d02/megatron/training/tokenizer/tokenizer.py) uses those operations without automatically inserting BOS/EOS. Read the special token IDs from the `.model`; do not copy them from a generic HF wrapper.

A public HF conversion [documents its use of a T5 compatibility wrapper](https://huggingface.co/ib-ssm/mamba2-8b-3t-4k-hf/blob/2b35ece89bfc2d66327d771c25f054bc1dc0f73a/README.md), which is not byte-for-byte equivalent to Megatron tokenization. Another [state-spaces conversion](https://huggingface.co/devingulliver/mamba2-8b/blob/663993b0c882616ab837222433c37b323a4eaf6a/config.json) omits the 8-group setting in its config. Neither is selected here.

## E8 provenance

The E8P12 codebook and LDLQ primitive originate in [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp), pinned at `1d8f873e9a2a8b86b12bb1064c312c5689b77d98`. The relevant original files are `lib/codebook/latticee8_padded12.py` and `lib/algo/quip.py`. QuIP# is GPL-3.0; its license is retained in [licenses/QUIP-SHARP-GPL-3.0.txt](../licenses/QUIP-SHARP-GPL-3.0.txt).

This project adapts those primitives to Mamba projections with its own rotation, calibration, storage, and decoder workflow. It does not claim to implement the complete original QuIP# fine-tuning recipe. The E8/W5 representation is lossy relative to full precision. Huffman packing of an already quantized representation is lossless.

## Attribution and release boundary

See [third-party notes](../licenses/THIRD_PARTY.md). No private Resurface checkpoint, adapter, training data, or private experiment history is included. Research comparisons with Quamba refer to public metadata and publications; Quamba implementation code is not vendored or imported by this release.
