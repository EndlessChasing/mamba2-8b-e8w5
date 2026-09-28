# Teacher-KL experiment readiness

**Launch snapshot:** the discarded training-only GPU smoke has started under
supervision (child PID2698330). This records the launch, not continuing process
liveness. No completed smoke, formal448-update training or quality result is
established in this snapshot. The final smoke outcome will be recorded separately.
Publication remains held; MK is deferred.

The [frozen protocol](TEACHER_KL_COMPENSATION_PROTOCOL.md) resets to accepted
all-small with fresh rank-4 factors. Pure teacher→student KL is the optimization
loss; CE is logged without a gradient. Learning rate is1e-4, with one448-window
pass. Both>=1% PPL improvement and lower mean teacher KL on64 reserved windows
are required before any full-validation data are loaded or scored.

## Completed CPU checks and data

- **4 helper/data checks:** independent pure-KL gradient and scaling/tail
  correctness, CE-label independence, split selection and actual token-file
  preparation/readback. [Receipt](../reports/teacher_kl_training_cpu_tests.json).
- **5 trainer checks plus27 bound input checks:** fixed448 schedule, attempt
  accounting and prepared data integration.
  [Receipt](../reports/teacher_kl_compensation_v1_trainer_cpu.json).
- **8 evaluator checks:**224 rounded masters/112 strict files, independent
  native merge, reserved weighted PPL/KL, exact repeated source CE, strict
  two-condition gate and proof that gate failure skips the validation callback.
  Actual data verification was included. [Receipt](../reports/teacher_kl_evaluator_cpu_checks.json).

All were CPU-only; these checks do not replace the native GPU smoke or establish
a PPL improvement. The helper/data, trainer and evaluator recipes were reviewed
before launch, and no previously frozen implementation was modified.

The actual [data manifest](../reports/teacher_kl_compensation_v1_data_manifest.json)
records519 eligible blocks after all prior fitting and observed-diagnostic
intervals are excluded. Reserve64 first, then choose448 training blocks from
the455 remaining;7 are unused. Every stored window contains2048 tokens and2047
targets. The fixed training pass has917,056 target exposures; reserved scoring
has131,008 targets. Internal, train/reserved and every prior-interval overlap
count is zero. Reserved tokens are never supplied to fitting or smoke forwards.

Data manifest SHA256:
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
Training token-file SHA256:
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`.
Reserved token-file SHA256:
`a3fc0803052890d973b74f91581a6bfc1d2f93ed28db5725f23c16eb8ec20546`.

## Frozen execution identities

| Input | SHA256 |
| --- | --- |
| Protocol | `c80fe02fe1dafb24b7a983bf347205c709a7af297cc325ed8761a29ecbff5c6a` |
| Pure-KL helper | `1806f84c38dd7c4f44ca9825a7a45b2e75cdca0356192b9a1804f7abc50c2e77` |
| Data preparation/verifier | `4e528dbaa965110a908c56f2386b12483c23d4bc1e8f80a341875ed00728cd13` |
| Training driver | `ce7e4eb448cb108c58f4406a5e68b7b11546ddf42f3be8064d26342cc8dbd8dc` |
| Independent evaluator | `b899318efba83afe9b8a091f7456d92aa9c929a0e2b43875725a5d4372cea071` |
| Stage supervisor | `b9abd6df7c01750ca551a4ff0668c6e4dc825c3ace8d70e22979dbee950d9a93` |

The supervisor runs exactly one requested phase and records process identity,
exit code and receipt SHA; it cannot resume or automatically launch the next
phase. A reserved quality failure is a successful completed negative experiment
(`complete:true`, `reserved_gate.passed:false`, full validation skipped), not
an execution failure. No quality outcome authorizes publication, extra epochs,
an intermediate checkpoint selection or a rank sweep.
