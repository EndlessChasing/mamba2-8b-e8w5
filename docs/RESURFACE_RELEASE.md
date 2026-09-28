# v0.2.0-resurface: release assembly and acceptance

Published: **public research prerelease**
[`v0.2.0-resurface`](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface).
Published on **2026-09-28 at19:36:35 UTC**. Complete packaging, public-command
restoration, archived-source GPU generation and public asset verification passed. The old empty
`v0.1.0-experimental` draft remains separate.

## Model selected for this release

The fixed candidate contains all112 nominal2.5-bit E8/axis projections, W4
embedding, W5 head, final448 readapted393 FP16 tensors, and the final1536 soft
adapter. No candidate or gate setting is selected during packaging.

| Quality scope | Measured result |
| --- | --- |
| Complete validation PPL, paired compressed base→adapter | 7.622396588→7.593163114 (−0.383521%) |
| Original FP16 source PPL, aligned historical run | 7.334175947; candidate +3.531237% |
| Independent CONFIRM normal MK, paired base→adapter | 91→340/384; +64.84375 percentage points |
| CONFIRM removed-target false matches | 0→0/384 |
| Observed DEV normal MK, original source→adapter | 167→346/384; unequal adaptation budget |

The original source was not evaluated on CONFIRM. The corpus validation set
informed development; the fresh numeric instances share the three DEV template
families. The [model card](MODEL_CARD.md) and
[complete results](RESURFACE_READAPTED_RESULTS.md) describe these limits and the
preserved historical MK replay failure.

## Byte scopes

Measured base data are **3,138,928,792 bytes**. The actual adapter file adds
**2,539,647 bytes**, so combined encoded model data are **3,141,468,439 bytes**.
The separate adapter manifest is294,171 bytes. These figures exclude outer
packaging, tokenizer, source, licenses and reports.

The outer container keeps the E8HUF001 format with all members stored raw
(`all-raw-members-no-entropy-coding`). It makes no additional entropy-compression
claim. Its119 payload files include117 base files, the adapter and tokenizer;
the composed raw manifest is counted separately. Complete size and
manifest/container digests are recorded in `release_manifest.json` and release
notes after terminal build/restore verification. The old all-small bundle's2,671,176,471-byte
total belongs to another model. Runtime expands weights toFP16; neither figure
is GPU residency.

## Independent packaging acceptance design

1. Resolve every inference file into a fresh self-contained directory, with no
   symlinks or dependencies on research artifacts outside the directory.
2. Tie actual decoded507 base and224 adapter tensor hashes to the completed
   full-PPL/CONFIRM model identities. Verify model configuration, native tokenizer
   and the single enabled soft policy.
3. Freeze a matching public software snapshot and provenance before assembly.
   Retain corresponding source, applicable licenses and modification notices;
   exclude private2.7B Resurface artifacts, WikiText text/token payloads and
   training checkpoints from inference requirements.
4. Verify the exact raw member ledger against the complete container readback.
   Count all outer assets, split parts and the manifest's own bytes. Reject flat
   upload basename collisions.
5. Exercise the shipped restore command into a new directory, then the archived
   loader using only restored files. Compare actual tensors/output identity and
   verify the soft adapter is active. A generation smoke establishes usability,
   not a new quality score.
6. Publish the pinned assets as the research prerelease and verify their public
   manifest/size/digest identities. Preserve previous artifacts and failed results.

See [readiness](RELEASE_READINESS.md) for status and [DOWNLOAD.md](DOWNLOAD.md)
for the end-user verification flow. Terminal packaging and public release receipts are recorded below.

## Completed build and restored inference

- [Build/process receipt](../reports/resurface_release_v0_2_0_build.json): exit0, CPU-only,72.15s;119 payload files plus the raw manifest restored exactly through the unchanged public CLI.
- [Independent distribution audit](../reports/resurface_release_v0_2_0_independent_audit.json): reconstructed file and tensor ledgers, source provenance, byte totals and download contract all pass.
- [Restored GPU inference](../reports/resurface_release_v0_2_0_inference.json): exit0,110.45s; actual archived code, all507 decoded base and224 adapter hashes exact, soft adapter enabled, three short prompts and exact fresh-cache repetition. No original checkpoint/calibration/training-file access; this is a Python-level access check, not OS isolation.
- [Restore receipt](../reports/resurface_release_v0_2_0_restore.json) binds all120 files to the final release manifest.

The batch1 cache is112 FP16 tensors totaling **122,028,032 bytes**. This smoke
peaked at **22,063,559,680 allocated /23,922,212,864 reserved GPU bytes** and ended
with16,820,608,512 allocated bytes. These are observed reference-loader figures
from the96GB RTX PRO6000, not a minimum-VRAM or throughput benchmark. Packaging
and generation do not remeasure the previously completed PPL/MK scores.

Exact source/tag commit: `e89c764e33a7fcd72095e1b9b9ea47f22d3ff7bb`.
The independent audit checked230 archived files against this Git commit, plus
the producer's snapshot provenance (231 indexed members).

Trusted release-manifest SHA-256:

```text
da5931dc8315bf576b773abdf4c77828a2994ccaf4fd14858c7798235b2ef19d
```

Final raw-manifest SHA-256:

```text
7dd96a44d3de6e634b949e2fb94bbc004c0aecc8404d3848d13f83b396616a1f
```

## Public distribution verification

[GitHub release](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface) — public research prerelease,26 assets totaling **3,165,987,804 bytes**.

The [streaming upload receipt](../reports/resurface_release_v0_2_0_upload.json)
records exact source bytes and all26 GitHub server SHA-256 matches. The
[publication receipt](../reports/resurface_release_v0_2_0_publication.json)
independently confirms a public repository, draft=false, the exact asset
inventory/size/digest ledger, a byte-identical manifest downloaded anonymously,
and HTTP200 for all three weight-part download endpoints. Large weight parts
were not downloaded a second time; server digests match the already verified
streamed bytes. The old v0.1 draft is unchanged.

The source tag and archived code remain fixed at the build commit. Build-time
status text in that archive is historical; these terminal receipts and the
release notes establish final publication status.
