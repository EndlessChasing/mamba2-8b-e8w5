# Release readiness: v0.2.0-resurface

**Publication is authorized for the latest validated model as a research
prerelease.** This supersedes the earlier publication hold for the current release;
historical failed experiments and their frozen protocols remain unchanged.
The planned tag is `v0.2.0-resurface`. The older empty
`v0.1.0-experimental` draft is retained separately.

## Scientific evidence

| Requirement | Status and evidence |
| --- | --- |
| Official pure Mamba-2 8B source and tokenizer | Complete: [source identities](SOURCES.md), [507-tensor inventory](../reports/source_tensor_inventory.json), [download SHA receipt](../reports/source_download.json) |
| Fixed final adapter and native arithmetic | Complete: [training/export results](RESURFACE_READAPTED_RESULTS.md); final1536 export with 224 tensors/1,154,104 parameters, base507 frozen |
| Same soft adapter enabled for recall and PPL | Complete: [frozen continuation protocol](RESURFACE_SOFT_CONTINUATION_PROTOCOL.md) and matching identities in both terminal evaluations |
| Complete paired PPL | PASS: 7.622396588→7.593163114; 130 windows/264,764 targets, exact restored baseline, [terminal report](../reports/resurface_soft_continuation_v1_full_eval.json), [independent audit](../reports/resurface_soft_continuation_v1_full_independent_audit.json) |
| Source-relative PPL | +3.531237% versus original FP16 7.334175947, within +5%; aligned historical source protocol, not equal-quality or untouched-test evidence |
| Independent recall confirmation | PASS: 91→340/384, removed matches 0→0/384, [terminal report](../reports/resurface_soft_continuation_v1_confirm_eval.json), [independent audit](../reports/resurface_soft_continuation_v1_confirm_independent_audit.json) |
| Historical failure disclosed | Preserved: [strict DEV report](../reports/resurface_readapted_v1_dev_eval.json) failed historical MK replay; the transparent continuation required exact paired restoration and unchanged quality gates |
| Source recall comparison scoped correctly | DEV only: 167→346/384, unequal training budget. Original source was not measured on independent CONFIRM |
| Model/software license scope and corresponding source | Existing GPL-3.0 software and Apache-2.0 upstream model/tokenizer notices retained; [third-party scope](../licenses/THIRD_PARTY.md) |

## Distribution acceptance

Quality validation of the measured model is complete. The following acceptance
checks concern the **new complete package**, not the earlier all-small bundle.
They remain pending until a terminal receipt establishes each result.

- [ ] Compose all latest base files and the actual final soft adapter into one
  self-contained model; bind the 507+224 decoded tensors to the quality reports.
- [ ] Generate an accurate outer model card and file ledger for 112 nominal2.5-bit
  projections, W4 embedding/W5 head, readapted393 and the soft adapter.
- [ ] Pack and read back every raw member; count complete distribution bytes,
  including tokenizer, matching software, licenses, reports and manifest itself.
- [ ] Restore using the shipped software into a fresh path and compare every
  restored member with the pinned raw ledger.
- [ ] Run the new loader from archived software using only restored model files,
  with soft adapter enabled and original-checkpoint access excluded.
- [ ] Check flat GitHub asset basename uniqueness, upload the complete assets and
  verify the public manifest/assets for `v0.2.0-resurface`.

The publisher records exact paths, hashes and completed status in
[RESURFACE_RELEASE.md](RESURFACE_RELEASE.md). No public-availability claim is made
by this checklist before publication verification.

## Immutable evidence identities

| Evidence | SHA-256 |
| --- | --- |
| Adapter file | `1e9857feaf80ede6525cce839726883f6230c98a067cb226eb2f095693ea803e` |
| Adapter manifest | `5660676adb45806052cbaab7f5e50056769c4f5e106d5a49cf148f025bf2b7a8` |
| Full PPL report | `06a71c11fc0a12a52add6e7bf5d28b8a8eb9f832b2ec1cb846d9acaf30afe961` |
| Independent CONFIRM report | `306ae9e8ed5e78756f7c8ea39c8db40dbebf3895ced3c5c279722be5100a344b` |
| Independent full-PPL audit | `14a9bb85a95dcdd6429e12409c73336765e546a9ae09b43b2c0bd28bdcc2d515` |
| Independent CONFIRM audit | `37e3c9e0e5498335fcbf0a30fbc0cfaee34418c80cf5ea0a1cfd99adad6b56da` |

## Publication limits

Publish as a research prerelease with the [model card](MODEL_CARD.md). Do not claim
3 GB GPU residency, universal recall retention, native Megatron parity, a global
minimum model size or equal quality to the uncompressed source. Earlier
[initial E8/W5 results](RESULTS.md) and the
[old all-small distribution](ALL_SMALL_DISTRIBUTION.md) remain historical evidence;
their sizes and restore receipts do not certify the new release.
