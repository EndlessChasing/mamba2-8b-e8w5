# Prototype compensation: completed result

The fixed 128-update recipe, serialized export and independent full validation
completed on 2026-09-28. Both predeclared quality gates **failed**. Stop this
recipe with no extra epochs or intermediate checkpoint selection. Retain the
accepted all-small candidate and its offline distribution. Publication remains
on hold; MK and test evaluation were not performed for this candidate.

## Full-validation quality

| Same-process arm | PPL | Gap versus FP16 source |
| --- | ---: | ---: |
| Original source cast to FP16 | 7.334175947318572 | — |
| Accepted all-small E8/W5 | 8.359867548009992 | +13.9851% |
| All-small plus learned prototypes | **8.306585061749145** | **+13.2586%** |

Prototype PPL decreases **0.6373604%** versus all-small; mean NLL decreases
**0.0063940023 nats/target**. Of 130 paired windows, 127 improve, none tie and
3 worsen. The worsening windows start at tokens204800,206848 and208896.

| Predeclared condition | Required candidate PPL | Result |
| --- | ---: | --- |
| At least1% improvement over all-small | <=8.276268872529892 | **FAIL** |
| At most5% above FP16 source | <=7.700884744684501 | **FAIL** |

The prototype has the lowest measured compact-candidate PPL so far, but the
gain is below the declared advancement threshold. It is not promoted to the
retained all-small package. The full [protocol](PROTOTYPE_COMPENSATION_PROTOCOL.md)
and [evaluation receipt](../reports/prototype_compensation_v1_eval.json)
preserve the thresholds and negative outcome.

All three arms use the same **130 windows /264,764 next-token targets** from
WikiText-2 raw validation, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Per-window input/token hashes,
target counts and ordering match. The tokenizer is unchanged; state resets for
each window, with at most2048 targets, native SSD prefill, FP16 model weights
and FP32 summed cross-entropy in64-token full-vocabulary chunks. This is the
same development validation split used in earlier repair work, not untouched
test evidence. No recall recovery or per-token state-quality result follows.

## Training and integrity

- Exactly128 successful updates in128 attempts, zero overflow retries and
  **262,016 target exposures**:32 fresh TRAIN windows,2047 targets each, four
  fixed seeded passes. Only the final checkpoint was selected.
- Exactly112 zero-initialized FP32 prototype tables of shape256×8:
  **229,376 trainable values**. Forward corrections are FP16. The evaluated
  all-small baseline's507 tensors, original E8 indices, rotations, scales,
  balances and W5 vocabulary matrices remain frozen.
- All112 tables changed. Every update records finite gradients for all112
  tables. The fixed CE/KL recipe, train-only data, code and passed smoke bindings
  match the [training report](../reports/prototype_compensation_v1_train.json).
- Final training export recomputed all507 baseline GPU tensor hashes and found
  them unchanged. The128-token functional/native export check has bitwise-equal
  hidden states and last-eight full-vocabulary logits, with max difference0.
- A fresh evaluator independently read all112 table files, checked them against
  the rounded final128-step masters, decoded the projections and installed
  native FP16 weights. It did not use the training bank's forward path.
  Actual112 candidate projection hashes and395 unchanged tensor hashes cover
  all507 native tensors. The395 are393 small tensors plus two W5 vocabulary
  matrices. This is fresh evaluation evidence, distinct from the earlier
  checkpoint32 startup-hash audit.

The final checkpoint is3,360,571 bytes, SHA256
`21d8aa7ce23899cb0df7db90b8bfe275f006abc18c94f2d2cc336033a1fa3083`.
Native model coverage remains8,236,999,680 parameters; the encoded prototype
tables are additional representation data. Learned corrections make this an
E8-derived representation, rather than an unchanged E8 lattice.

## Measured storage and execution

| Scope | Bytes |
| --- | ---: |
|112 FP16 correction-table values |458,752 |
|112 strict table files, including32-byte headers |462,336 |
|Prototype overlay manifest |142,637 |
|112 table files plus overlay manifest |**604,973** |
|Base logical raw data plus prototype table files, excluding manifests |2,886,944,798 |
|Physical parent + all-small replacement + prototype directories |2,894,670,608 |

The physical directory total includes retained duplicate small-tensor files and
manifests; it is not a complete standalone distribution. Tokenizer, software,
reports and training files are outside these inference-data scopes. No
prototype Huffman/container size was measured. The existing **2,671,176,471-byte**
[all-small distribution](ALL_SMALL_DISTRIBUTION.md) is retained unchanged and
does not contain the prototype overlay.

Training elapsed19,981.123 seconds (about5.55 hours), including load/export
checks. Peak allocated GPU memory was46,710,939,648 bytes and peak reserved was
48,192,552,960 bytes; the peak includes temporary native export verification.
Independent evaluation elapsed312.222 seconds. These are execution observations,
not an isolated throughput benchmark or evidence of compressed GPU residency.

## Evidence and next status

| Receipt | SHA256 |
| --- | --- |
|[Training](../reports/prototype_compensation_v1_train.json) |`ce64a6819e7789244e7364aa687bfc2b842e7d155bd5d1b4baaf97df646303b4` |
|[Overlay manifest](../reports/prototype_compensation_v1_manifest.json) |`44b6535f3e1f5ef7bbc0d23e1f4967a159f6b4273d0479729bfe8c9371c27eae` |
|[Independent validation](../reports/prototype_compensation_v1_eval.json) |`4c6c17edde18cd2ccc911f453e1d14b41cb575dd6d60bfcb15575ce4b2ffcc1d` |
|[Completed job](../reports/prototype_compensation_v1_job.json) |`12b8786fb270c4a539f8c71fa0d7369f4ab3c9809539d6ed75c6edb1fb035733` |

Both job stages exited0; completion is an execution/integrity outcome, not a
quality pass. The [job history](PROTOTYPE_RUNNING_JOB.md) preserves the launch,
early checkpoint audit and terminal status. A separate [rank-4 experiment on
all112 projections](LOW_RANK_COMPENSATION_PROTOCOL.md) is being prepared from
the accepted all-small baseline. It has no GPU training or quality result yet.
