# Standalone package composition for the evaluated all-small model

`scripts/compose_all_small_package.py` creates a new flat raw directory for the
already measured `small_compensation_v1` model. It performs CPU identity checks;
it does not train, score, Huffman-pack, publish, or choose a candidate.

## Exactly which model

The script accepts only the pinned original two-sweep `e8w5_v1` parent, final1024
all-small overlay, and completed full-validation report
`reports/small_compensation_v1_eval.json` (SHA starts `957de68b`). That report
measured PPL8.359867548 on264,764 targets; it is not a new composition-time score
and does not establish MK quality, a test result, or publication acceptance.

The output contains116 inherited files (112 E8 projection files, two W5
vocabularies, the E8 codebook and configuration), the trained `other_fp16.pt`,
and a fresh `manifest.json`. The393 actual CPU FP16 tensors are checked by shape,
dtype, finiteness and canonical byte SHA against the evaluated export. All116
inherited file hashes are rechecked, with114 decoded tensor identities bound to
the historical loaded-model audit. CPU composition does not redecode the large
projection or vocabulary tensors.

The new manifest is assembled from an explicit field list. It contains the
replacement archive hash and current393 tensor hashes, unchanged114 decoded
hashes, exact507 tensor mapping, original parent/overlay/report SHA bindings,
and scoped PPL evidence. Original-small hashes, original-small quality claims,
calibration-error estimates and old timing metrics are not copied as current
candidate metadata. Source checkpoint hashes remain provenance only.
The `binding.code_sha256` entries use module-relative `runtime.py` and
`codec.py` filenames, preserving the archived restore verifier's existing
interface. The broader `frozen_code_sha256` uses repository-relative paths.

## Compose a separate resolved directory

From the remote repository, using a new output path:

```sh
CUDA_VISIBLE_DEVICES="" /home/horde/.venvs/lodram/bin/python \
  scripts/compose_all_small_package.py \
  --output artifacts/all_small_resolved_v1 --mode auto
```

`auto` tries a hardlink for each input and falls back to a bounded-memory copy
when the filesystem does not support linking. `copy` always creates independent
files; `hardlink` fails instead of copying. No mode edits, chmods or overwrites
input files. An existing output or `<output>.building` is refused. A failed
staging directory remains available for inspection.

The output is immutable by workflow, not by an operating-system write lock.
Hardlinks share input inodes: never edit them in place or change their modes.
Use `--mode copy` when independent writable inodes are needed. The composer
rehashes both input and output after materialization and only then renames the
completed staging directory.

## Verified raw bytes

Actual CPU composition completed successfully in
`artifacts/all_small_resolved_v1`. All117 files were hardlinked;393 actual small
tensor hashes and116 inherited file hashes passed. The complete receipt and
exact manifest are retained as
[`all_small_resolved_v1_composition.json`](../reports/all_small_resolved_v1_composition.json)
and [`all_small_resolved_v1_manifest.json`](../reports/all_small_resolved_v1_manifest.json).

| Scope | Actual bytes |
| --- | ---: |
|117 resolved raw data files |2,886,482,462 |
|New manifest |246,900 |
|Complete flat raw directory |2,886,729,362 |

Manifest SHA-256:
`bc554db936b13cbbeaeef5267d96b8d6ba7c183bf740c7e64e5f26ac7c478af8`.
Composition receipt SHA-256:
`3ce670e21cd874ea730c8d1de8d875412611c2aa79d65e11df6398c62df9e0fb`.

Disk blocks shared through hardlinks do not reduce this logical inference-file
total. Tokenizer, source, licenses, quality reports, container overhead and the
outer release manifest belong to the later release total and are excluded here.
Composition did not remeasure quality or execute the GPU model.

## Verified weight container; complete release remains unbuilt

The separate CPU-only pack completed, with full trusted readback of118 files and
342 independent chunk checks. A subsequent identity audit matched all118 members
to the pinned resolved manifest and revalidated the raw file ledger. Both
receipts report that CUDA was not initialized:

- [Full readback](../reports/all_small_resolved_v1_huffman.json).
- [Post-readback identity audit](../reports/all_small_resolved_v1_huffman_identity.json).

| Artifact scope | Original `e8w5_v1` | Trained all-small |
| --- | ---: | ---: |
|117 raw data files, excluding manifest |2,886,482,462 B |2,886,482,462 B |
|Huffman weight container, including raw manifest |2,660,128,443 B |2,660,171,471 B |
|Complete release with tokenizer/software/licenses/outer metadata |2,666,260,364 B |Not built |

All-small container SHA-256:
`46be4e12d18ed36e33adf002962f0b7bcafe8478b249b628eca853930d43efcd`.
The container includes its raw manifest, all weight data, Huffman tables,
offsets, scales, signs and codebooks. Its measured size excludes tokenizer,
software, licenses, quality reports and outer release metadata. The complete
release size in the original column belongs to the earlier original package.

Exact file identity binds the all-small container to the existing full-validation
PPL8.359867548. Packing did not remeasure PPL, MK or test quality, and no
all-small archived-software GPU restore smoke has run. The next complete release
build can use `scripts/package_release.py` with the pinned tokenizer and actual
quality report; it must count every additional shipped asset and its manifest.

Following normal release restoration, the existing frozen loader supports this
all-small representation directly:

```python
from mamba_e8w5.runtime import load_quantized_model
model = load_quantized_model("/path/to/restored/raw", device="cuda")
```

Only restored raw files and the shipped code are needed to load weights. The
tokenizer is needed to encode/decode text. Original NVIDIA weights, norm-v1,
training checkpoints, Hessians and training token files are not dependencies
of restored inference. Loading expands weights to FP16; compressed file size
does not describe GPU residency.

## Learned prototypes remain pending

This composer rejects extra files and accepts no prototype branch. A future
prototype artifact requires its own fail-closed schema and a loader that
explicitly applies112 tables; the frozen raw loader ignores extra `.proto`
files. That work remains conditional on completed quality evidence and review.
