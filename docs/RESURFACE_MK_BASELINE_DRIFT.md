# Native MK baseline drift: bounded diagnostic

Status: provisional CPU audit while the frozen DEV evaluator is running. No
GPU probe, warmup, model update, scoring-rule change, or quality promotion was
performed. The historical exact-output requirement remains in force.

## Observed evidence

Historical baseline report: `readapted_mk_baseline_v1.json`, SHA256
`a88f8fe75e82b992b8fc9cebeb6f681847e87aa36ea8bb1771edf979c47d551d`.
Provisional audit: `reports/resurface_mk_baseline_drift_provisional.json`, SHA256
`5bad9a6df90206c54aaa23380b66644cef101a2f86b5bab47c4b71bef0e015e2`.
The audit binds the actual bytes of its live-report snapshot; it does not label
the live evaluation complete.

- All 768 ordered prompt IDs, token hashes, query/answer metadata, and cache
  coverage/dtype metadata are exact across historical and current baselines.
- All 507 previously measured FP16 parameter-content hashes agree, before and
  after each baseline arm. Shape, stride, dtype and trainability match. No
  parameter-pointer alignment differences occur modulo16/128/256/512. These
  are report-ledger comparisons, not a new load of model binaries.
- Both baselines pass their 12 within-arm fresh-cache replays.
- Seven generated outputs differ: four normal prompts and three removed
  controls. Only one correctness result changes: `validation-n16-t0-s6`,
  wrong `650549` becomes correct `663105`. Normal recall is111→112/384;
  removed-control matches remain0/384.
- All64 prose windows, their FP32 CE chunk sums, NLL and aggregate PPL are
  exactly the accepted prior: PPL7.6242504268035765, NLL266121.0063724518,
  131,008 targets.

| Case | First divergent generated-token offset (zero based) |
| --- | ---: |
| validation-n16-t0-s6 | 2 |
| validation-n16-t0-s13-removed | 2 |
| validation-n16-t2-s45 | 9 |
| validation-n64-t0-s46-removed | 2 |
| validation-n64-t1-s12-removed | 3 |
| validation-n64-t1-s18 | 2 |
| validation-n64-t2-s23 | 8 |

All seven complete old/new ID sequences are preserved in the audit JSON.
An output first diverging during a cached step does not establish where its
underlying hidden-state numerical difference first arose.

## Bounded source inspection and limits

The installed Mamba2 source calls `mamba_chunk_scan_combined` with
`return_final_states=ssm_state is not None`, then copies `last_state` into the
FP16 cache (`mamba_ssm/modules/mamba2.py:244–264`). Cached MK generation therefore
exercises a path not covered by cache-free prose PPL alone.

The installed chunk-scan, chunk-state, BMM and state-passing prefill kernels
have Triton autotune configurations. Triton3.6 selects minimum measured time
for each uncached configuration key and stores the result in an in-process
cache (`triton/runtime/autotuner.py:212–242`). Mamba's deterministic utility
can restrict configurations, but the live process had no explicitly set
allowlisted Mamba/Triton/CUBLAS determinism environment variables. The old
process's selected kernel configurations were not recorded. Historical
execution scored the source model first; the new process scores current first.
Different tuning/allocation history is a plausible explanation, not a measured
cause. Equality of shapes and coarse pointer alignment cannot identify a
cuBLAS algorithm or exclude every numerical-path difference.

The cached `selective_state_update` kernel explicitly avoids autotuning because
it overwrites state; it selects block sizes by `dstate`
(`selective_state_update.py:192–198`). It would be inaccurate to blame cached
step autotuning specifically.

Inspected installed file SHA256 values:

- `ssd_chunk_scan.py`: `055b5ce4cb0f30c84c3e031d775f218227fa0b793359254a2465777097b224f9`
- `selective_state_update.py`: `4bc67389730c9fd0ad40569b9ee48de88c699bc9612fbc4607eb5d38c1b4a9b9`
- `utils/determinism.py`: `cb6e1c30392c11200425c2a23ad9fa3d47f50b556d15e9b0caf79b7d483d6f1d`
- `triton/runtime/autotuner.py`: `40f5e142806219942585a16742e9f3721d5bfef88145b571cf389f9aec6460bb`

No per-token hidden/logit/cache-value traces or historical kernel selections
exist in these receipts. Cause attribution remains unresolved. Await the
frozen run's full restored-current arm for same-process paired evidence;
a historical mismatch must remain a failure under its existing protocol.
