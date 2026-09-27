# Mamba-2 8B E8/W5

Work in progress: a reproducible compact release of NVIDIA’s pure Mamba-2 8B using E8 lattice quantization for projections and W5 for the separate embedding and language-model head.

The goal is a small, openly reproducible full-model package. No 8B compressed size, quality retention, runtime-memory result, or smallest-model claim has been established yet.

Source: [nvidia/mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k), Apache-2.0. Quantization components derived from QuIP# retain GPL-3.0 licensing. Code and model licenses will be documented separately.

Only public upstream model/runtime sources and project-authored compression code will be included; research checkpoints and large model files are excluded from Git.
