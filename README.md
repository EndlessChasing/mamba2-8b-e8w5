# Mamba-2 8B E8/W5

An experimental compact version of NVIDIA’s pure Mamba-2 8B using E8 lattice quantization for projections and W5 for the separate embedding and language-model head.

All **8,236,999,680 parameters** have been quantized or retained and independently verified. The raw weight data occupies **2,886,482,462 bytes** before entropy coding, tokenizer and release metadata. **Model publication remains on hold after the failed quality gate.** The release is a draft with no uploaded assets.

## Measured quality

| Full test measurement | Original weights cast to FP16 | E8/W5 reconstructed to FP16 |
| --- | ---: | ---: |
| WikiText-2 PPL, 300,963 next-token targets | 7.2445 | 8.7085 |
| Public MK, normal prompts | 18/48 | 15/48 |
| Public MK, target-removed controls | 0/48 | 0/48 |

PPL increases **20.21%** and recall decreases **6.25 percentage points**. This fails the provisional quality-retention target; the release is experimental and does not claim equal quality. Measurements use 2,048-target windows with state reset and SSD prefill; per-token FP16-state validation is reported separately. The MK task is new and has limited sample size. See the [frozen protocol](docs/EVALUATION.md) and [full paired results](reports/comparison_full_prefill.json).

The [PPL diagnosis](docs/PPL_DIAGNOSIS.md) records controlled component and projection-role interventions, independent codec checks and numerical reproducibility limits. Entire-validation confirmation (264,764 targets) gives original PPL7.3342, E8-only8.7298, W5-only7.4154, and complete E8/W5 8.8365. This locates the main quality loss in E8 projection quantization; model publication remains on hold.

A separate [eight-sweep repair experiment](docs/E8_REFINEMENT_RESULTS.md) improved all 112 projection reconstruction proxies at unchanged raw bitrate. Paired development PPL improved only **0.461%**, with recall unchanged, missing the predeclared 1% advancement threshold. This route stopped; no refined full-validation result or entropy-coded package is claimed.

The subsequent [existing-norm compensation experiment](docs/NORM_COMPENSATION_RESULTS.md) trained 692,224 existing parameters at unchanged raw capacity. Development PPL improved **2.942%**, but normal recall fell **10/12→9/12**. Its combined gate failed, so full validation was skipped and this recipe stopped. It is not a promoted replacement for the original candidate.

The decoded FP16 evaluation peaked at **17.36 GB allocated / 18.60 GB reserved** GPU memory. Compact file size is not compressed inference residency. Timings were collected on a shared GPU and are not isolated throughput benchmarks.

The staged complete package occupies **2,666,260,364 bytes**. Its actual restore command reproduced all118 raw files exactly; generation also passed using the archived software, restored tokenizer and restored weights, with no original checkpoint as an input. This establishes package usability, not a quality pass. See the [restore receipt](reports/release_restore_local_v1.json) and [generation smoke](reports/restored_inference_local_v1.json). The package remains unpublished.

Source: [nvidia/mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k), Apache-2.0. The software and QuIP#-derived components are distributed under GPL-3.0; upstream model attribution is separate. See [license scope](licenses/THIRD_PARTY.md), [pinned sources](docs/SOURCES.md), and [comparison limits](docs/COMPARISON.md).

Only public upstream model/runtime sources and project-authored compression code will be included; research checkpoints and large model files are excluded from Git.

Download instructions are in [DOWNLOAD.md](docs/DOWNLOAD.md); model assets will appear in [GitHub Releases](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases). A global smallest-model claim is not established. The public comparison records the exact same-base artifacts audited and which quality measurements were not repeated.

## Method

- **112 projection matrices:** activation-calibrated diagonal balancing, signed DCT/Hadamard rotations, and QuIP# E8 lattice quantization with LDLQ feedback. Each 8-value vector stores a 16-bit index: 2 bits/weight before scales, balancing factors and signs.
- **Two independent vocabulary matrices:** W5, groups of 128 values, an FP16 scale per group. The 256,000-token embedding and output head are not tied.
- **Remaining parameters:** FP16; the original 56-layer architecture and eight SSM groups are preserved.
- **Entropy coding:** chunked Huffman over quantized symbols with tables and offsets counted. It losslessly restores the already-lossy quantized package.

The reference loader expands the package to FP16 for quality measurements. Compact disk storage does not imply equally compact inference memory. There is no custom compressed GPU matrix-multiplication kernel in this first release.

The [checkpoint architecture audit](reports/pure_mamba2_architecture_audit.json) confirms **56 pure Mamba-2 blocks**, with zero attention, independent MLP or MoE blocks. The actual source checkpoint has `hybrid_attention_ratio=0` and `hybrid_mlp_ratio=0`; all 507 tensors map exactly to the parent and norm candidate.

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
