# Eight-sweep refinement: small gain, advancement gate failed

**Status: experiment complete; this route is stopped. Publication remains on hold.**
The fixed change from two to eight additional LDLQ coordinate sweeps improved
all 112 projection reconstruction proxies, but development PPL improved only
0.460844%, below the predeclared 1% threshold. Full validation was therefore
skipped. No test evaluation, candidate promotion or new entropy-coded package
was performed for this candidate.

## Paired language-quality result

Both arms ran in one process using the same four frozen validation windows,
4096 next-token targets, tokenizer, prefill execution and 24 public recall/control
prompts. Both W5 vocabularies and all 393 retained FP16 tensors were unchanged.

| Metric | Parent: two sweeps | Candidate: eight sweeps |
| --- | ---: | ---: |
| Development PPL | 9.237163951 | 9.194595025 |
| Normal MK correct | 10/12 | 10/12 |
| Target-removed matches | 0/12 | 0/12 |

Mean NLL decreased by 0.004619092681 nats/token, giving **0.460844% lower PPL**.
Two prose windows improved and two worsened. All 24 paired recall correctness
outcomes were unchanged. These small recall counts do not establish full-test
recall preservation.

The [frozen protocol](REFINEMENT_PROTOCOL.md) required PPL at most
`0.99 × parent`, or **9.144792311**, plus no loss of normal successes and no
increase in target-removed matches. Recall conditions passed; the PPL condition
failed. The gate was not changed after seeing results. The negative outcome is
retained in the [evaluation receipt](../reports/e8_refinement_cd8_dev.json).

The original candidate's full-test PPL increase of 20.208% and MK 18/48→15/48
remain the authoritative first-candidate results. This four-window experiment
neither replaces those numbers nor establishes a full-corpus repaired score.

## What it shows about the cause

The [component diagnosis](PPL_DIAGNOSIS.md) already isolated E8 projection
quantization as the dominant source of degradation. This experiment tests one
narrow explanation: whether the original two-sweep search left easily recovered
quality at the same bitrate.

More search helped locally, but the measured language gain was small. All 112
original-H squared-output errors improved, with median 1.8970%, minimum 1.0351%
and maximum 2.3663%; input/output medians were 1.6381%/2.0457%. These are training
reconstruction metrics. They are not percentages of language loss recovered.
The six pilot comparisons use independent FP64 reductions; the other 106 use
the parent's recorded FP32 RMS proxy squared, with that precision limitation
disclosed in the [build summary](../reports/e8_refinement_cd8_summary.json).

The evidence supports stopping this fixed two-to-eight-sweep repair. It does
not prove that every longer search, different quantizer or 2-bit representation
must fail. Further quality work would need a distinct declared experiment, such
as training existing retained parameters or allocating additional precision.
The [metadata-training design](METADATA_TRAINING_DESIGN.md) remains an untested
proposal; no training has been run as part of this experiment.

## Integrity, size and computation

- All 112 decoded FP16 hashes match their candidate records. Exactly 507 parameter
  tensors and 8,236,999,680 parameters are resolved; all 395 nonprojection parameter
  objects remain unchanged.
- Logical raw data remains **2,886,482,462B**: 1,535,708,888B of new E8 files plus
  1,350,773,574B of inherited vocabulary, small-tensor, config and codebook files.
- The overlay, including its manifest, occupies 1,535,914,423B. The current
  experimental loader also requires the complete parent directory, so their
  physical total is **4,422,600,757B**, excluding tokenizer/software/calibration.
- No refined Huffman container was built. The first candidate's 2,660,128,443B
  container and 2,666,260,364B complete distribution are not measured sizes of
  this refinement.
- Full generation took 1139.90s; paired evaluation took 120.93s. Evaluation peak
  GPU allocation was 30,166,538,240B while retaining decoded FP16 parent/candidate
  projections. This is diagnostic memory, not compressed inference residency.

The [build documentation](E8_REFINEMENT_BUILD.md) contains reproduction commands
and the durable per-matrix recovery procedure. The original candidate, pilot,
release draft and prepared release files remain unchanged.

## Evidence identities

| Evidence | SHA-256 |
| --- | --- |
| Frozen protocol | `70fcc73a51bea70d0b308881af31482dce0f3e83175c03cbca2484ca230e6b89` |
| Parent raw manifest | `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed` |
| Candidate overlay manifest | `5c7d912bb06e0af107218418f575d04c5b789f910568f0c847fa0322228fd035` |
| Resolved candidate file ledger | `8ccbdd93ce233fc02c981ca601156b28f61d6b742edb59822c8389b2090f079e` |
| CPU integrity receipt | `ad9fbe2afb07196b04e1e9f6784d531a764de14c15f4b04bd2f646e76aec56cc` |
| Build summary | `6b17ccafa330d7b89d5016dd640a7e7b354962487b6924a96bd66719d8027239` |
| Paired development result | `b2cbdd55e1791d32abfcebc811262430e6b724859966ce01e3b2f552ce6d565a` |

Reports: [manifest](../reports/e8_refinement_cd8_manifest.json),
[CPU integrity](../reports/e8_refinement_cd8_integrity.json),
[training summary](../reports/e8_refinement_cd8_summary.json),
[paired development](../reports/e8_refinement_cd8_dev.json).
