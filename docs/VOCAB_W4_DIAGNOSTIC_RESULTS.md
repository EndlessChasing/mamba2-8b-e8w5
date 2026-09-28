# W4 vocabulary diagnostic: completed

The four fixed arms completed with exit code 0. Reducing the embedding to W4
raises PPL by **0.158945%**; reducing the output head to W4 raises it by
**5.307371%**. Both actual W4 files save the same raw byte count, so this
observed-set comparison identifies substantially different quality costs.
It does not promote a W4 model or establish generalization.

## Fixed scope and native file readback

Follow the [frozen protocol](VOCAB_W4_DIAGNOSTIC_PROTOCOL.md), SHA256
`3e0748b1393400a5d4ac168587330ba587d84a3171f86e243aa08dda34e091be`.
Each arm scored the same **64 already-observed reserved TRAIN windows /
131,008 next-token targets**, with fresh state, native FP16 backbone/head and
FP32 full-vocabulary CE in 64-token chunks plus the 63-token tail. No fitting,
temperature correction, validation/test or MK evaluation was performed.

W4 files were written directly from the original BF16 checkpoint tensors,
through the existing writer's FP32 row conversion and FP16 group-128 scales.
They were not requantized from decoded W5 or pre-cast source FP16. Each actual
serialized file was read back and its decoded FP16 tensor identity checked.
All 112 E8 projections and 393 trained small tensors remained fixed.

## Paired quality

| Embedding / output head | PPL | Change from W5/W5 | Mean NLL delta, nats/target | Windows improved |
| --- | ---: | ---: | ---: | ---: |
| W5 / W5 baseline |8.397361922037003 |— |— |— |
| W4 / W5 |8.410709112538152 |+0.158945% |+0.001588189 |18/64 |
| W5 / W4 |8.843041047563577 |+5.307371% |+0.051713228 |0/64 |
| W4 / W4 |8.859629526950979 |+5.504915% |+0.053587350 |0/64 |

The two-replacement interaction is **+0.000285934 nats/target**, defined as
`NLL_both - NLL_embedding - NLL_head + NLL_baseline` using token-weighted means.
It describes non-additivity in these fixed arms, not causal layer attribution.
The diagnostic had no promotion gate; all four arms completed as declared.

Restoring both W5 tensors and repeating the baseline gave exact CE equality
for all **64 windows / 2048 chunks**, including the aggregate PPL. Full
507-tensor content audits passed before and after every arm; expected references
and all baseline values were restored at the end. The historical baseline
PPL8.397298748944545 differs by **+0.0007523%** from this process's baseline.
That drift is disclosed separately from the exact same-process controls.

## Measured raw file sizes

Each vocabulary has 1,048,576,000 values, 8,192,000 FP16 scales and a 79-byte
header. These are actual file sizes, including packed codes, scales and header.

| File scope | W5 bytes | W4 bytes | Raw saving |
| --- | ---: | ---: | ---: |
| Embedding |671,744,079 |540,672,079 |131,072,000 |
| Output head |671,744,079 |540,672,079 |131,072,000 |
| Both vocabulary files |1,343,488,158 |1,081,344,158 |262,144,000 |

W4 embedding SHA256 is
`9857f7d145bb5f833e591356ac1d2d76e1845bd688917ab77da9d476bce7a181`;
W4 head SHA256 is
`4b94775d5ec42ab7f3540809e3d14525a38ce076996ac3080763f79b344d025e`.
Metadata, tokenizer, software and entropy coding remain separate. No new
complete raw package or Huffman distribution was composed from these arms.

Internal runtime was **389.8935 seconds**, process wall time **392.0862 seconds**.
Peak allocated GPU memory was **21,549,383,680 bytes** for the decoded FP16
quality reference, not compressed inference residency or isolated throughput.

## CPU-control correction retained

The first CPU test failed a bitwise independent-oracle comparison at nine
values: expected negative zero versus decoded positive zero; numerical
`torch.equal` still passed. Unsigned offset codes do
not preserve that sign, and decoded zeros are positive zero. Before GPU
execution, the test oracle was corrected to canonicalize its expected zeros
according to the stored format; the bounded CPU checks then passed. The
existing codec/runtime, frozen protocol and model weights were unchanged.
The failed CPU receipt remains available alongside the passed receipt.

## Interpretation and next planned direction

On this observed set, W4 embedding has a much smaller PPL penalty than W4 head
for the same raw saving. This motivates preserving **W5 head** in the next
planned primary candidate: enhanced input projections at a nominal **2.5 bits
per weight**, W4 embedding, and otherwise retained accepted values. Its
[separate protocol](INPUT_AXIS_RESIDUAL_PROTOCOL.md) is declared, with no
implementation or quality benefit yet.

The preliminary code-payload arithmetic adds 266,076,160 bytes for the extra
0.5 bit across the 56 input projections and saves 131,072,000 bytes from W4
embedding: a net **135,004,160 bytes**, about **4.6771%** of the current
2,886,482,462 raw data bytes, before new headers or other metadata. It is not
a same-capacity claim or an actual package measurement. Whether the extra
projection precision compensates for the embedding penalty remains unmeasured.

Accepted all-small **full-validation PPL8.359867548009992** and its verified
distribution remain unchanged. These diagnostic windows were already observed;
no fresh/full-validation/test quality or recall recovery is established.
The earlier scalar-temperature screen remains failed. No candidate is promoted
and no publication follows this diagnostic.

## Evidence

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
|[Four-arm diagnostic](../reports/vocab_w4_v1.json) |1,533,504 |`3ae1bb986ddfa7f2e514fb2451885a83feee5b6c5cc3a1924f248b4a259e353b` |
|[Process / exit](../reports/vocab_w4_v1_process.json) |1062 |`de7707faedbc78e0773160ccf6e91c72180fe75e1ee6ebc684798b266f3e0202` |
|[Passed CPU controls](../reports/vocab_w4_v1_cpu.json) |22,291 |`8ad3659f7b8b5fc2106434a9d5d569d351e7a6322f092fec54bc7caceb6c758a` |
|[Preserved first CPU failure](../reports/vocab_w4_v1_cpu_attempt1.json) |534 |`7693ec1974ac0d376d7811863d005a3b317ccc986b60fe829d80b73fc372e53e` |

Final diagnostic script SHA256:
`c1dd8343a204923d6799f5b4eed31a6f0ffee1bc0774b2deaa15f34e0d54091d`.
