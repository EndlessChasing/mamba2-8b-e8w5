# Release readiness: local 8B candidate

**Current PPL-priority result:** the [all-small-tensor experiment](SMALL_TENSOR_COMPENSATION_RESULTS.md) completed 1024 updates and full validation. PPL is **8.35987**, down **2.6750%** from norm-v1 and **5.3946%** from original E8/W5, with raw capacity unchanged. It remains **13.9851% above FP16**, so the +5% source target is unmet. MK is deferred under the user's [PPL-first direction](PPL_PRIORITY_CONTINUATION.md); recall recovery is unmeasured for this candidate. Publication remains on hold.

Evidence snapshot: **2026-09-28 UTC** (2026-09-27 Pacific).
This is a completion checklist, not permission to publish.

**Status: model publication is on hold.** The first E8/W5 candidate is complete,
compact and restorable, but fails the declared quality-retention target. A
refinement experiment is separate from the immutable first candidate. Its
development PPL gain of0.460844% missed the predeclared1% advancement threshold;
the route stopped and full validation was skipped.

## Requirements and evidence

| Requirement | Status | Authoritative evidence | Gap / next action |
| --- | --- | --- | --- |
| New public repository under the requested owner | Complete | [Repository](https://github.com/EndlessChasing/mamba2-8b-e8w5); read-only [GitHub API](https://api.github.com/repos/EndlessChasing/mamba2-8b-e8w5) returned `private:false`, owner/name `EndlessChasing/mamba2-8b-e8w5`, default branch `main` | New local diagnosis/refinement documents are not thereby proven pushed; publication/push hold remains in force |
| Exact official pure Mamba-2 8B source and tokenizer | Complete | [Source identity](../reports/source_model.json), [full download SHA receipt](../reports/source_download.json), [507-tensor inventory](../reports/source_tensor_inventory.json), [source arguments](../reports/source_args_audit.json) | Native original Megatron end-to-end parity is not established |
| Public code provenance, model/software licenses and corresponding source | Complete for the built candidate | [Source notes](SOURCES.md), [third-party scope](../licenses/THIRD_PARTY.md), retained license texts and pinned QuIP# sources; immutable outer manifest includes `source.zip` and its SHA | If distribution metadata changes, retain applicable notices and count the revised files; no private research code/data may enter the package |
| Complete E8/W5 parameter coverage | Complete | [Quantization manifest](../reports/quantization_v1.json), [independent decoded-weight audit](../reports/decoded_weight_audit.json): 112 E8 projections, two W5 vocabularies, 393 FP16 tensors, 8236999680 parameters | This verifies representation/coverage, not quality retention |
| Actual compact package with all bytes counted | Complete locally | Immutable `artifacts/release-e8w5-v0.1.0/release_manifest.json`; [local verifier receipt](../reports/release_restore_local_v1.json), `steps.verify` | Container 2660128443 B; entire prepared distribution 2666260364 B including manifest. No model assets are publicly released |
| Complete source and quantized prose/recall evaluation | Complete; quality target **FAIL** | [Baseline full test](../reports/baseline_full_prefill.json), [candidate full test](../reports/e8w5_full_prefill.json), [paired comparison](../reports/comparison_full_prefill.json), [protocol](EVALUATION.md) | PPL +20.208%; normal MK 18/48→15/48; original source-relative quality gate remains unmet |
| Separately scoped per-token FP16 cache check | Complete for development scope | [Paired tokenwise development report](../reports/comparison_dev_tokenwise.json) | Four validation windows and 24 recall/control cases only; not full-test tokenwise validation |
| Explain the regression before promotion | Evidence collected; repair pending | [PPL diagnosis](PPL_DIAGNOSIS.md), [independent protocol audit](../reports/ppl_protocol_audit.md), [full-validation component confirmation](../reports/ppl_components_full_validation.json), [quantizer math audit](QUANTIZER_MATH_AUDIT.md) | E8 projections dominate the measured loss. This does not prove that the tested quantizer is the best achievable rate/quality tradeoff |
| Exercise the actual end-user restore command | Complete | [Restore acceptance](../reports/release_restore_local_v1.json): all 118 original/restored raw files match SHA and size, tokenizer/config match, temporary container removed | 2886686334 raw bytes verified including original raw manifest; no source checkpoint is used by restoration |
| Generation using only restored files and archived software | Complete for loading/generation smoke | [Completed receipt](../reports/restored_inference_local_v1.json): 61 archived source files verified; three prompts produce nonempty outputs; fresh-cache repeat is exact; no original-checkpoint access attempt recorded | Source loader disabled and Python open guard used, not an OS filesystem sandbox. This confirms end-user loading/generation for the exercised prompts; it does not repair quality |
| Accurate model card and package metadata | Local replacement draft prepared; outer metadata refresh pending | [Model-card draft](MODEL_CARD.md), current immutable model-card/manifest inspection below | Outer v0.1 card lacks explicit numerical FAIL. Refresh only after publication hold is lifted; recalculate all affected hashes and self-inclusive byte totals |
| Accurate repository comparison/progress documents | Complete for current local evidence | [Results](RESULTS.md), [comparison scope](COMPARISON.md), this checklist | Verified size, completed checks and the failed quality target are explicit. These local updates remain unpublished; recheck public competitors before any future publication |
| Smallest open model claim | Not established | [Finite public comparison audit](COMPARISON.md) | A public repository and small local artifact are insufficient. No global-minimum or equal-quality comparison is certified |
| Respect publication hold | In force | User direction; public [GitHub releases API](https://api.github.com/repos/EndlessChasing/mamba2-8b-e8w5/releases) returned an empty list at this snapshot | No upload, release publication or push is authorized while held. Public API does not expose private draft details |

## Immutable candidate identities

| Artifact | SHA-256 |
| --- | --- |
| Quantized raw manifest | `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed` |
| Huffman container, concatenated two parts | `5b12eba1c21e342fc05b5e774396fb23a3b29d7a668b27ce73ecb97e536b658b` |
| Prepared outer release manifest | `d5f5c77eba397c6706a1b59290f37fd1754728b9e00637aed8a390c1d6922ff1` |
| Current outer `MODEL_CARD.md` | `cb632479d3447f2a6f5d821d09571baf613991ab320afb210428558fee8d0c99` |
| Archived `source.zip` | `35f6a56f12cdc9cc6822f0c3a6fdd9e019cb71901ca636acf1f24d2e5e9bee49` |
| Full local restore acceptance report | `06010c123221c73ca3a03606e9926280d28ef2e65b6e6f3fd6bd909034e7cd10` |
| Archived-source restored generation receipt | `ebe5e3070330b111561da4455c92f837cea6e46ae92daa5f1b062a0e0e0525cd` |

The archive was built from code commit
`02462fd58cf6d0fa857251f819bf738c40013d21`, with a clean worktree recorded by the
builder. Later diagnosis/refinement files are not silently attributed to that
archived snapshot. The restore wrapper's initial path-format reporting error is
preserved separately; it happened before restoration, and the completed receipt
contains the subsequent full verification.

The restored generation smoke used the actual archived runtime with matching
frozen SHA, restored raw files and restored tokenizer. Three prompts and a repeat
completed in 68.625 seconds including loading, with peak GPU allocation
17226020864 B. No original-checkpoint argument was supplied. These are acceptance
checks, not a throughput benchmark or new PPL/MK result.

## Current outer metadata assessment

The unchanged outer model card accurately describes lossy E8/W5 quantization,
lossless Huffman restoration, the full parameter count, FP16 expansion, source
and license scope. It links the quality reports and explicitly avoids an
automatic pass/equal-quality/global-smallest claim.

However, users must open reports to discover **PPL +20.21%, MK −6.25 percentage
points and quality target FAIL**. The outer manifest's quality status is only
`reports-attached-no-automatic-pass`; it is not a false pass, but is less explicit
than the known measured failure. The local [draft](MODEL_CARD.md) makes that
failure visible. It does not modify the archived package or its current totals.

Before any later release, apply an explicit metadata amendment or build a new
release version, bind the summary to the shipped candidate-comparison report,
adjust relative links for the archive layout, recalculate asset SHA/bytes and the
manifest's own size, then verify the entire release again. Preserve the old
artifact identity. Do not overwrite its quality history with a later refinement.

## Refinement remains a separate experiment

The [fixed six-matrix screen](../reports/e8_refinement_screen_v1.json) passed its
training-proxy gate: median original-H squared-error improvement **1.657357%**,
six of six matrices meet the individual threshold, worst improvement
**1.035091%**. These are local training reconstruction metrics, not PPL or MK.

All112 eight-sweep projections passed integrity checks. Paired development PPL
changed9.237164→9.194595 (−0.460844%), with normal MK10/12→10/12 and controls0/12→0/12.
The [declared1% advancement gate](REFINEMENT_PROTOCOL.md) failed; the route stopped
and full validation was skipped. See the [completed results](E8_REFINEMENT_RESULTS.md).
This refinement is not a promoted or entropy-packaged replacement candidate.
The original source-relative quality target also remains unmet.

A subsequent [norm-only training recipe](NORM_COMPENSATION_RESULTS.md) completed
128 fixed updates and passed all integrity checks at unchanged raw payload size.
Development PPL improved 2.9421%, but normal MK fell 10/12→9/12, so its combined
advancement gate also failed and full validation was skipped. No candidate has
been promoted and no new entropy-coded distribution has been built. Original
release files and the publication hold remain unchanged.
