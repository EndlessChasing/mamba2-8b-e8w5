# Norm compensation: complete validation PPL

The user requested [PPL-first research](PPL_PRIORITY_CONTINUATION.md), permitting
this separate continuation after the original norm experiment's combined gate
failed. The fixed norm export was not retrained or reselected. The earlier
PPL/MK development result and its failed combined gate remain unchanged.

## Same-process complete validation

| Model | Validation PPL | Change versus original FP16 |
| --- | ---: | ---: |
| Original FP16 source | 7.334175947 | — |
| Original E8/W5, two sweeps | 8.836560440 | +20.4847% |
| E8/W5 with existing-norm compensation | **8.589640668** | **+17.1180%** |

Norm training reduces PPL **2.7943% versus the original quantized parent**.
All 130 windows improve versus the quantized parent. Mean NLL decreases
0.028340806758 nats per target. The remaining source-relative
mean NLL gap is 0.158011843641 nats per target. This confirms the four-window
development gain extends to complete validation, but the +5% source-PPL target
is still not met. It does not show that normalization alone can close the gap.

All arms scored the same **264,764 targets / 130 windows**, at most 2048 targets
per window, fresh zero state, native prefill, FP16 head GEMM and FP32 cross
entropy in chunks of 64 tokens. The receipt preserves every window's token
hash and each arm's NLL. It verifies the fixed norm export, its final training
checkpoint and the loaded parent's 112 E8, two W5 and 393 small tensors.
The final candidate was evaluated after applying its actual FP16 export.

Validation has already been inspected in earlier diagnosis and candidate
research. These results are not untouched test evidence. The original test
result remains separately reported in [RESULTS.md](RESULTS.md). No test split
or additional MK run was used here. The historical development MK result is
still 10/12 to 9/12, and future Resurface recovery is unmeasured.

The logical raw model remains **2,886,482,462 bytes**. A new Huffman container
has not been built, so the original distribution's byte count is not a measured
size for this export. The quality runtime expands parameters to FP16.

## Reproduction and identity

- Script: [evaluate_norm_ppl_continuation.py](../scripts/evaluate_norm_ppl_continuation.py).
- Declared scope: [NORM_PPL_CONTINUATION.md](NORM_PPL_CONTINUATION.md).
- Receipt: [norm_compensation_v1_ppl_continuation.json](../reports/norm_compensation_v1_ppl_continuation.json).
- Receipt SHA-256: `298e423566e1c6dba39570f72b3c80b3185aa3fa8f6a8d4597b1192a42cde68d`.
- Source checkpoint: `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
- Quantized parent manifest: `ef47f52000c14fd644cc0ee459318beb16ce1e078e3586cbe2e946c7506722ed`.
- Norm export manifest: `1fa9d35ff56a08e2f7ddf258bfa1fb61bc2e83ae2af404ab2f68a781821ac4d8`.

Publication remains on hold. The next experiment fits the existing small
tensors under its own [fixed protocol](SMALL_TENSOR_COMPENSATION_PROTOCOL.md).
