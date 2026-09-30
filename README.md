# Mamba-2 8B E8/W5， 3GB， Recall Recovered

## Current v0.2.0: frozen WikiText-2 test

| Frozen same-base arm | Full test PPL |
| --- | ---: |
| Current readapted E8/W5 base, Resurface disabled | 7.53418476 |
| Identical base, published soft Resurface enabled | **7.50295894** |

The already published `v0.2.0-resurface` model was evaluated on the complete
WikiText-2 **test** split, separately from its earlier validation results.

Both arms score the same **300,963 targets / 147 windows**, including the last
1,955 targets, with the published SSD **parallel prefill** and zero initial state
per window. All 147 windows improve; relative PPL change is **−0.4144552164%**.
All 507 decoded base tensors remain unchanged, all 224 adapter tensors match the
published artifact, and adapter removal restores the first-window score exactly.
The independent CPU audit passed. This measures decoded FP16 execution, not
compressed GPU residency or per-token quantized state. No training or selection
was performed in this test.

See the [complete paired result and audit](evaluation/wt2_test_v1/),
[frozen test protocol](docs/RELEASE_V02_WT2_TEST_V1_PROTOCOL.md), and
[external test runner](scripts/run_release_v02_wt2_test_v1.py).
The original `e8w5_v1` test below describes different pre-repair weights and
remains unchanged. WikiText-2 test was previously exposed elsewhere in this
project; this is a frozen retrospective test of the released v0.2.0 artifact.


**Current release: `v0.2.0-resurface` (research prerelease).** The complete **3,165,987,804-byte (3.166 GB)** package is [publicly available](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface). Exact restoration, archived-source GPU inference and all26 GitHub asset checksums passed. Use the [release guide](docs/RESURFACE_RELEASE.md) and [download instructions](docs/DOWNLOAD.md) for the latest axis-residual, W4-embedding/W5-head model with its enabled soft Resurface adapter. The original `v0.1` recipes below are retained as experiment history.

**Historical validation and MK repair:** the [fixed soft Resurface adapter](docs/RESURFACE_READAPTED_RESULTS.md) improves independent numeric-binding recall **91→340/384 (23.70%→88.54%)**, with removed-target matches **0→0/384**. Complete130-window validation PPL improves **7.622396588→7.593163114**; all130 windows improve, with exact historical/restored-baseline CE replay. Independent confirmation gives251 gained/2 lost and a conservative95% improvement lower bound of **+58.50 percentage points**. Both tasks use the same enabled final adapter, adding **2,539,647 serialized bytes** with all507 base tensors frozen. Native audits and independent receipt checks pass. The [continuation protocol](docs/RESURFACE_SOFT_CONTINUATION_PROTOCOL.md) explicitly preserves a historical cross-process MK replay failure and changes only that prerequisite. Results cover three shared prompt families/N16/64 and previously used WikiText-2 validation. The user authorized publication after reviewing these results.

**Preceding PPL-priority result:** [current-axis small-parameter readaptation](docs/AXIS_SMALL_READAPTATION_RESULTS.md) and independent native evaluation completed. Training the existing393 small tensors for448 updates lowers complete-validation PPL **7.977512009→7.622396588 (4.45146%)**, with **130/130 windows improving**. The candidate is **3.92983% above source7.334175947** and passes the declared **source+5% PPL target**. All112 nominal2.5-bit axis projections, W4 embedding and W5 head remain fixed. Raw model data stay **3,138,928,792 bytes** with **zero replacement-byte delta**; the new provenance manifest adds239,782 bytes separately. That stage did not include MK or a new complete package.

The preceding [output-axis residual experiment](docs/OUTPUT_AXIS_RESIDUAL_RESULTS.md) measured full-validation PPL **8.109428666→7.977512009 (1.62671%)**, with116/130 windows improving. Its paired source+5% target failed before the new readaptation. The [same-capacity joint E8/axis screen](docs/JOINT_AXIS_SELECTION_RESULTS.md) also stopped: all six sampled reconstruction errors improved, but median reduction was only **1.839555%**, with **0/6** reaching2%. It failed the declared expansion gate before all112 rebuilding or PPL; pilot file sizes were unchanged and matrix-stage time was5.4934x greedy replay.

The preceding [input-axis experiment](docs/INPUT_AXIS_RESIDUAL_RESULTS.md) improved full-validation PPL **8.359867548→8.109428666**. The [W4 diagnostic](docs/VOCAB_W4_DIAGNOSTIC_RESULTS.md) motivated retaining W5 for the head. Earlier [scalar-temperature](docs/SCALAR_TEMPERATURE_RESULTS.md), [prototype](docs/PROTOTYPE_COMPENSATION_RESULTS.md), [rank-4](docs/LOW_RANK_COMPENSATION_RESULTS.md) and [teacher-KL](docs/TEACHER_KL_COMPENSATION_RESULTS.md) recipes remain stopped with their recorded negative gates.

The historical [accepted all-small model](docs/SMALL_TENSOR_COMPENSATION_RESULTS.md), **full-validation PPL8.359867548**, and its verified **2,671,176,471-byte** offline distribution remain unchanged. The readapted candidate uses more raw bytes than that historical package's underlying data; the current adapter result is summarized above. This does not establish a global minimum or equal quality. Validation has informed development and is not untouched test evidence. The user-directed MK repair follows the earlier [PPL-first work](docs/PPL_PRIORITY_CONTINUATION.md).

An experimental compact version of NVIDIA’s pure Mamba-2 8B. The original baseline uses E8 lattice quantization for projections and W5 for both vocabulary matrices. The latest PPL candidate adds projection residuals, readapts existing small parameters, and uses W4 for the embedding while retaining W5 for the language-model head.

All **8,236,999,680 base parameters** have been quantized or retained and independently verified. The historical all-small raw weight data occupy **2,886,482,462 bytes** before entropy coding, tokenizer and release metadata; the newer candidate's raw size is given above. The original combined quality gate remains failed; the separate fixed-soft Resurface continuation passes both complete PPL non-regression and independent MK improvement. Its raw base-plus-adapter data are **3,141,468,439 bytes**, excluding manifests/tokenizer/software. The latest complete distribution and restoration evidence are tracked in the [release guide](docs/RESURFACE_RELEASE.md).

## Original candidate: measured test quality

**Historical, different weights:** the following test, runtime-memory and restored-generation measurements describe the original `e8w5_v1` candidate before compensation training. The retained all-small candidate has separate full-validation, complete offline-distribution, CPU restoration and archived-software generation evidence; its test quality remains unmeasured.

| Full test measurement | Original weights cast to FP16 | Original `e8w5_v1` reconstructed to FP16 |
| --- | ---: | ---: |
| WikiText-2 PPL, 300,963 next-token targets | 7.2445 | 8.7085 |
| Public MK, normal prompts | 18/48 | 15/48 |
| Public MK, target-removed controls | 0/48 | 0/48 |

PPL increases **20.21%** and recall decreases **6.25 percentage points**. This fails the provisional quality-retention target; the release is experimental and does not claim equal quality. Measurements use 2,048-target windows with state reset and SSD prefill; per-token FP16-state validation is reported separately. The MK task is new and has limited sample size. See the [frozen protocol](docs/EVALUATION.md) and [full paired results](reports/comparison_full_prefill.json).

The [PPL diagnosis](docs/PPL_DIAGNOSIS.md) records controlled component and projection-role interventions, independent codec checks and numerical reproducibility limits. Entire-validation confirmation (264,764 targets) gives original PPL7.3342, E8-only8.7298, W5-only7.4154, and complete E8/W5 8.8365. This located the main quality loss in E8 projection quantization and led to the subsequent repair experiments.

A separate [eight-sweep repair experiment](docs/E8_REFINEMENT_RESULTS.md) improved all 112 projection reconstruction proxies at unchanged raw bitrate. Paired development PPL improved only **0.461%**, with recall unchanged, missing the predeclared 1% advancement threshold. This route stopped; no refined full-validation result or entropy-coded package is claimed.

The subsequent [existing-norm compensation experiment](docs/NORM_COMPENSATION_RESULTS.md) trained 692,224 existing parameters at unchanged raw capacity. Development PPL improved **2.942%**, but normal recall fell **10/12→9/12**. Its original combined gate failed, so that protocol stopped before full validation. The subsequent user-directed [PPL-only full validation](docs/NORM_PPL_RESULTS.md) measured **8.589640668**, versus **8.836560440** for original E8/W5 and **7.334175947** for the FP16 source, over all **264,764 targets**. The historical combined gate remains failed, and no norm entropy-coded package has been built.

The original `e8w5_v1` decoded FP16 evaluation peaked at **17.36 GB allocated / 18.60 GB reserved** GPU memory. Compact file size is not compressed inference residency. Timings were collected on a shared GPU and are not isolated throughput benchmarks.

The staged original `e8w5_v1` complete release occupies **2,666,260,364 bytes**. Its actual restore command reproduced all118 raw files exactly; generation also passed using the archived software, restored tokenizer and restored weights, with no original checkpoint as an input. This establishes package usability, not a quality pass. See the [restore receipt](reports/release_restore_local_v1.json) and [generation smoke](reports/restored_inference_local_v1.json). The package remains unpublished.

## Historical verified storage

| Artifact scope | Original `e8w5_v1` | Trained all-small |
| --- | ---: | ---: |
|117 raw data files, excluding manifest |2,886,482,462 B |2,886,482,462 B |
|Huffman weight container, including raw manifest |2,660,128,443 B |**2,660,171,471 B** |
|Complete release, including tokenizer/software/licenses/outer metadata |2,666,260,364 B |**2,671,176,471 B** |

The all-small container passed complete readback of **118 files**, **342 independent chunk checks**, and a subsequent identity audit against its pinned resolved manifest. See the [readback receipt](reports/all_small_resolved_v1_huffman.json), [identity receipt](reports/all_small_resolved_v1_huffman_identity.json), and [composition details](docs/ALL_SMALL_COMPOSITION.md). The complete offline distribution additionally includes the tokenizer, fixed corresponding source, licenses, reports and outer metadata. Its actual CPU restore command reproduced **all 118 raw files exactly**; see the [distribution audit](docs/ALL_SMALL_DISTRIBUTION.md) and [build/restore receipt](reports/all_small_distribution_v2_job.json). It retains the previously measured full-validation **PPL8.359867548** through exact file identity; packing did not rerun PPL. Archived-software GPU loading/generation also passed on three short prompts, with exact fresh-cache repetition and no original checkpoint input; see the [generation receipt](reports/all_small_restored_inference_v2.json). This is package-usability evidence, not new PPL/MK or compressed GPU residency. This historical distribution remains unpublished.

Source: [nvidia/mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k), Apache-2.0. The software and QuIP#-derived components are distributed under GPL-3.0; upstream model attribution is separate. See [license scope](licenses/THIRD_PARTY.md), [pinned sources](docs/SOURCES.md), and [comparison limits](docs/COMPARISON.md).

Only public upstream model/runtime sources and project-authored compression code will be included; research checkpoints and large model files are excluded from Git.

Download instructions are in [DOWNLOAD.md](docs/DOWNLOAD.md); the current version is tracked at [v0.2.0-resurface](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface). A global smallest-model claim is not established. The public comparison records the exact same-base artifacts audited and which quality measurements were not repeated.

## Historical validation and recall for the current release

| Metric | Uncompressed source | Current release |
| --- | ---: | ---: |
| Full WikiText-2 validation PPL, 264,764 targets | 7.33418 | 7.59316 (+3.53%) |
| Same 384 development MK prompts | 167/384 (43.49%) | 346/384 (90.10%) |
| Independent 384 confirmation MK prompts | Not measured | 340/384 (88.54%) |
| Serialized model data, excluding tokenizer/metadata/software | 16,474,189,490 B | 3,141,468,439 B |

The independent confirmation compares the current compressed base (91/384) with the enabled adapter (340/384). The source comparison uses historical development runs with identical prompt/token identities. Adapter training adds an adaptation budget, so higher recall is not evidence that quantization itself improves recall. See the [full result and limitations](docs/RESURFACE_READAPTED_RESULTS.md).

- **112 projections:** E8 quantization plus axis residuals, nominal 2.5 bits/weight before metadata.
- **Vocabulary:** W4 embedding and W5 output head; untied 256,000-token matrices.
- **Small base tensors:** 393 readapted tensors; the final base still has 507 tensors and 8,236,999,680 parameters.
- **Soft Resurface:** 224 FP16 adapter tensors, 1,154,104 parameters; the same enabled adapter is used for recall and PPL.
- **Recurrent cache:** FP16, 116.375 MiB at batch size 1; no additional recurrent adapter state.
- **Release transport:** lossless raw members in a split E8HUF001 envelope. This version does not claim additional Huffman size reduction.

## Original method (historical)

- **112 projection matrices:** activation-calibrated diagonal balancing, signed DCT/Hadamard rotations, and QuIP# E8 lattice quantization with LDLQ feedback. Each 8-value vector stores a 16-bit index: 2 bits/weight before scales, balancing factors and signs.
- **Two independent vocabulary matrices:** W5, groups of 128 values, an FP16 scale per group. The 256,000-token embedding and output head are not tied.
- **Remaining parameters:** FP16; the original 56-layer architecture and eight SSM groups are preserved.
- **Entropy coding:** chunked Huffman over quantized symbols with tables and offsets counted. It losslessly restores the already-lossy quantized package.

The reference loader expands the package to FP16 for quality measurements. Compact disk storage does not imply equally compact inference memory. There is no custom compressed GPU matrix-multiplication kernel in this first release.

The [checkpoint architecture audit](reports/pure_mamba2_architecture_audit.json) confirms **56 pure Mamba-2 blocks**, with zero attention, independent MLP or MoE blocks. The actual source checkpoint has `hybrid_attention_ratio=0` and `hybrid_mlp_ratio=0`; all 507 tensors map exactly to the parent and norm candidate.

## Reproduce the original quantization experiment

For the current downloadable model, follow [DOWNLOAD.md](docs/DOWNLOAD.md). The commands in this section recreate the original pre-repair candidate.

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
