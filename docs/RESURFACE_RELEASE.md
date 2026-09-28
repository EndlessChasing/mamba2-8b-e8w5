# v0.2.0-resurface: release assembly and acceptance

Target: **public research prerelease**
[`v0.2.0-resurface`](https://github.com/EndlessChasing/mamba2-8b-e8w5/releases/tag/v0.2.0-resurface).
Publication is authorized. At this documentation snapshot, assembly and public
asset verification are in progress; no completed upload or public download is
claimed. The old empty `v0.1.0-experimental` draft remains separate.

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
for the end-user verification flow. Exact terminal packaging and public release
receipts will be recorded here when available.
