# Mamba2-8B E8/axis + soft Resurface

Research prerelease of the pure NVIDIA `mamba2-8b-3t-4k` model:112 nominal2.5-bit E8/axis projections, W4 embedding, W5 head, readapted small tensors and the mandatory soft Resurface-inspired adapter.

## Download

Complete download: **3,165,987,804 bytes (3.166 GB)**, including three weight parts, tokenizer/configuration, corresponding source, licenses and quality evidence. Encoded model data alone:3,141,468,439 bytes. All120 restored files passed exact readback through the public restore command.

Trusted `release_manifest.json` SHA-256:

```
da5931dc8315bf576b773abdf4c77828a2994ccaf4fd14858c7798235b2ef19d
```

```sh
curl -fL https://raw.githubusercontent.com/EndlessChasing/mamba2-8b-e8w5/v0.2.0-resurface/scripts/download_release.py -o download_release.py
python3 download_release.py --tag v0.2.0-resurface \
  --manifest-sha256 da5931dc8315bf576b773abdf4c77828a2994ccaf4fd14858c7798235b2ef19d \
  --output downloaded-model
```

Follow [DOWNLOAD.md](https://github.com/EndlessChasing/mamba2-8b-e8w5/blob/main/docs/DOWNLOAD.md) to restore and generate. Use the shipped `mamba_e8w5.release_generate` loader so the same soft adapter is enabled for every prompt. The original NVIDIA checkpoint is unnecessary for inference.

## Measured quality

| Aligned measurement | Reference | Released model |
| --- | ---: | ---: |
| Complete WikiText-2 validation PPL,264,764 targets | Original FP16 source7.33418 | **7.59316 (+3.53%)** |
| Paired complete-validation PPL | Current compressed base7.62240 | **7.59316 (-0.38%)** |
| Independent numeric-binding MK confirmation | Compressed base91/384 (23.70%) | **340/384 (88.54%)** |
| Target-removed false matches | 0/384 | **0/384** |

All130 PPL windows improve versus the paired compressed base. On the same384 development MK prompts, the original source scored167/384 and the adapted candidate346/384, but adapter training adds an unequal adaptation budget. The source was not evaluated on independent confirmation. Validation informed development, and numeric confirmation uses three shared prompt families with fresh instances. These results do not establish general recall or equal quality to the original model.

The strict historical cross-process MK replay failure remains preserved. The separately declared continuation changed only that prerequisite and required same-process restoration, full-PPL non-regression and fresh confirmation. See the [full results and protocol](https://github.com/EndlessChasing/mamba2-8b-e8w5/blob/main/docs/RESURFACE_READAPTED_RESULTS.md).

## Execution and provenance

The reference loader expands weights to **FP16**; the download size is not GPU residency. Recurrent state remains FP16 (116.375MiB at batch1). No additional entropy compression, optimized compressed GPU kernel, ASIC performance or globally smallest-model claim is made.

Corresponding source commit: `e89c764e33a7fcd72095e1b9b9ea47f22d3ff7bb`. The source archive contains every required inference source plus selected evidence; full historical reports remain in Git. NVIDIA model/tokenizer and Mamba runtime: Apache-2.0. Project/QuIP-derived software: GPL-3.0; license texts and notices are included.

Archived-source restored GPU generation passed: all507 base and224 adapter tensors match the validated model, soft adapter enabled, three prompts and exact fresh-cache repetition. Build/restore and independent package audits passed. [Build](https://github.com/EndlessChasing/mamba2-8b-e8w5/blob/main/reports/resurface_release_v0_2_0_build.json), [independent audit](https://github.com/EndlessChasing/mamba2-8b-e8w5/blob/main/reports/resurface_release_v0_2_0_independent_audit.json), [GPU inference](https://github.com/EndlessChasing/mamba2-8b-e8w5/blob/main/reports/resurface_release_v0_2_0_inference.json).

All **26 GitHub assets** passed service-side SHA-256 verification before publication. The fixed source archive records its build-time status; terminal build, GPU and publication receipts on `main` record the completed release.
