# Same-capacity joint E8/axis selection: final screen

## Outcome

The fixed six-matrix GPU screen and independent actual-file CPU audit completed.
The selector improves every sampled original-H squared error, but the median
reduction is **1.839555%**, below the predeclared **5%** expansion threshold.
**Zero of six** reach the separately required **2%** reduction; four were required.
No matrix worsens. The fixed recipe therefore stops under the
[protocol](JOINT_AXIS_SELECTION_PROTOCOL.md).

No all-112 joint overlay was built, and **no joint-candidate PPL was measured**.
This is a failed reconstruction screen, not measured evidence that PPL worsens
or that every same-capacity method fails. Current best measured complete-validation
PPL remains **7.977512009**, versus source **7.334175947** and the source +5%
target **7.700884745**. Recall/Resurface and publication remain deferred.

## What changed

The prior encoder first selected an E8 point, then corrected its largest residual
coordinate with one signed-axis offset. The new encoder retains that greedy
result and evaluates all 16 signed-coordinate offsets jointly with E8 selection.
It accepts only strictly smaller FP32 Euclidean error for the same LDLQ query.

Each vector still stores **20 bits per eight weights**: the existing 16-bit E8
index plus four axis bits. Stored amplitude, header, balancing, signs, scale and
file length are unchanged. Original BF16 weights, original TRAIN Hessians,
seed, damping .01, scale .9, two tuning sweeps and frozen decoding are retained.
The same-query guarantee does not extend to final original-H error or PPL:
LDLQ feedback queries change, and inverse transforms end in FP16 rounding.

## Actual six-matrix measurements

`J = tr((W_hat_FP16 - W_BF16) H_original (W_hat_FP16 - W_BF16)^T)`.
Reduction is `1 - J_joint/J_greedy`; it is a squared-error reduction, not an RMS
reduction or a PPL reduction. Absolute J values have different matrix scales.

| Matrix | Greedy J | Joint J | J reduction | Greedy / joint seconds |
| --- | ---: | ---: | ---: | ---: |
| layer0.in_proj | 27.20080757 | 26.68166542 | 1.908554% | 4.105 / 31.431 |
| layer0.out_proj | 0.007805784233 | 0.007665010635 | 1.803452% | 5.311 / 18.082 |
| layer18.in_proj | 127.1845551 | 124.9376755 | 1.766629% | 3.513 / 31.043 |
| layer18.out_proj | 1.46556294 | 1.440484166 | 1.711204% | 5.271 / 18.258 |
| layer55.in_proj | 208.1107178 | 204.1126709 | 1.921115% | 3.532 / 30.897 |
| layer55.out_proj | 154.5055389 | 151.6075439 | 1.875658% | 5.292 / 18.743 |

The six matrix stages took **27.0240 s** for greedy replay and **148.4535 s**
for joint selection, a **5.4934x** ratio. This includes per-matrix validation and
is not an isolated inference-throughput benchmark. Total supervised screen wall
time was **227.1302 s**, with exit code 0; the child process is absent.
Largest measured joint quantization allocation was **4,039,436,800 bytes**
(3.7620 GiB), which is not whole-model inference memory.

## Integrity and capacity

- All six greedy files reproduced the current raw bytes, indices, metadata and
  GPU-decoded FP16 identities exactly before any joint matrix was generated.
- All six joint files passed stored-index and finite GPU FP16 readback checks.
- Independent CPU audit checked all 12 actual files, exact fixed metadata,
  complete inventories, 34 input bindings, five inherited files and the 395
  nonprojection identity ledger; it recomputed the gate from GPU-recorded J.
  It did not repeat GPU decoding or dense-H multiplication.
- Six greedy files total **102,815,745 bytes**. Six joint files total exactly
  **102,815,745 bytes**. Raw replacement delta is **0 bytes**.
- Retaining both sets for reproducibility uses **205,631,490 data bytes**, plus
  the **98,893-byte** screen manifest. These are experiment artifacts, not a
  new complete model or Huffman package.
- Existing best-candidate raw model data remain **3,138,928,792 bytes**.
  Joint full-model size, entropy-coded size and runtime residency were not measured.

The independent CPU audit completed in 30.8355 s with CUDA uninitialized.
The implementation and auditor each passed separate bounded CPU checks and
independent code review. Frozen codec/runtime/vendor sources were not edited.

## Reproducible evidence

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
| [joint_axis_screen_v1_cpu.json](../reports/joint_axis_screen_v1_cpu.json) | 50,942 | `aab5bc24a182faeaba5731f30b53174829c5074c83dbd4175f739eab1200c897` |
| [joint_axis_screen_auditor_cpu.json](../reports/joint_axis_screen_auditor_cpu.json) | 902 | `59edc14b6c8742f9c7e56c11d459db1ef8a4334f230da99e3d94358959f38a2c` |
| [joint_axis_screen_v1.json](../reports/joint_axis_screen_v1.json) | 99,050 | `79999e75928e88bd3ada934c4d35f6442d6c139a5504f3b3b3e2277b19b51b04` |
| [joint_axis_screen_v1_manifest.json](../reports/joint_axis_screen_v1_manifest.json) | 98,893 | `8a92439950accd9a035ad3f3c06856a2bbd70f9981d289700ab538d01d59c748` |
| [joint_axis_screen_v1_process.json](../reports/joint_axis_screen_v1_process.json) | 880 | `9393c6209bb5955d50c697fdcadd07206515da86c51d81d6d09127dd9e9f92ef` |
| [joint_axis_screen_v1_audit.json](../reports/joint_axis_screen_v1_audit.json) | 31,289 | `db8613a4aed6171aba85c19995e25999032cb60e5add8708af3c7743d9b22231` |

Screen artifact manifest SHA256:
`8a92439950accd9a035ad3f3c06856a2bbd70f9981d289700ab538d01d59c748`.
Protocol SHA256:
`73c175d7a81886cc7102640966d0b2b8ff47734234d586f7c584f222e0a03bcd`.
Selector SHA256:
`b84525d4ba2da0cc711b5e140bfc55ff631055fcfa0dd583d92237dbac6adbba`.
Screen driver SHA256:
`014822bf77d928e3a1cd8163574c5616bcba23342f607fe4ceb66f479945f34c`.
Independent auditor SHA256:
`6eefc8488e0a7aa2aa73cbc508eb2c4fcaf81f8a96098a8c90df54abf75867cf`.

Scripts require the pinned source, current axis artifacts and TRAIN calibration
on the GPU host. They refuse to overwrite prior reports or artifacts. The final
screen and audit are preserved even though the expansion gate failed.
