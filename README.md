# Mamba-2 8B E8/W5

Work in progress: a reproducible compact release of NVIDIA’s pure Mamba-2 8B using E8 lattice quantization for projections and W5 for the separate embedding and language-model head.

The goal is a small, openly reproducible full-model package. No 8B compressed size, quality retention, runtime-memory result, or smallest-model claim has been established yet.

Source: [nvidia/mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k), Apache-2.0. The software and QuIP#-derived components are distributed under GPL-3.0; upstream model attribution is separate. See [license scope](licenses/THIRD_PARTY.md), [pinned sources](docs/SOURCES.md), and [comparison limits](docs/COMPARISON.md).

Only public upstream model/runtime sources and project-authored compression code will be included; research checkpoints and large model files are excluded from Git.

## Method

- **112 projection matrices:** activation-calibrated diagonal balancing, signed DCT/Hadamard rotations, and QuIP# E8 lattice quantization with LDLQ feedback. Each 8-value vector stores a 16-bit index: 2 bits/weight before scales, balancing factors and signs.
- **Two independent vocabulary matrices:** W5, groups of 128 values, an FP16 scale per group. The 256,000-token embedding and output head are not tied.
- **Remaining parameters:** FP16; the original 56-layer architecture and eight SSM groups are preserved.
- **Entropy coding:** chunked Huffman over quantized symbols with tables and offsets counted. It losslessly restores the already-lossy quantized package.

The reference loader expands the package to FP16 for quality measurements. Compact disk storage does not imply equally compact inference memory. There is no custom compressed GPU matrix-multiplication kernel in this first release.

## Reproduce

Use Linux, a CUDA-capable PyTorch installation, a C++17 compiler, and the public `mamba-ssm` runtime. Large quantization jobs are being run on an RTX PRO 6000 Blackwell with 96 GB VRAM. Smaller cards have not been validated for this pipeline.

```bash
git clone https://github.com/EndlessChasing/mamba2-8b-e8w5.git
cd mamba2-8b-e8w5
python -m pip install -e .
python -m pip install 'mamba-ssm==2.3.2.post1' --no-build-isolation
python -m unittest discover -s tests -v
python scripts/download_source.py --output models/source
python -m mamba_e8w5.calibration \
  --source-dir models/source --out-dir calibration/v1 \
  --nwin 32 --seqlen 2048 \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3
python -m mamba_e8w5.quantize \
  --source-dir models/source --hessian-dir calibration/v1 \
  --out-dir artifacts/e8w5_v1 --scale 0.9 --damping 0.01 --tune-iters 2
```

`mamba-ssm` wheel/build compatibility depends on PyTorch, CUDA and GPU architecture. The initial experiment environment and results will be recorded with the release. The optional `causal-conv1d` extension is not required by this reference pipeline.

Calibration alone writes approximately 18.8 GB of covariance matrices, in addition to the 16.47 GB original checkpoint and generated packages. Do not store these large files in Git.

## Evaluate

The [frozen protocol](docs/EVALUATION.md) specifies train-only calibration, validation screens, full test PPL, public multi-key recall, and a separately reported per-token FP16-state check.

```bash
python -m mamba_e8w5.evaluation \
  --source-dir models/source --report reports/baseline_full_prefill.json \
  --suite full --execution prefill \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3
python -m mamba_e8w5.evaluation \
  --source-dir models/source --raw-dir artifacts/e8w5_v1 \
  --report reports/e8w5_full_prefill.json --suite full --execution prefill \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3
```

Use `--suite dev --execution tokenwise` with a different report path for the explicitly sequential FP16-cache check. A test report states its actual extent; a development screen is not a complete test-set measurement.
