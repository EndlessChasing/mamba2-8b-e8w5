# Mamba-2 8B E8/W5 execution checklist

Status: publication remains on hold at the user's request. Current best repaired full-validation PPL is 8.359867548 from all-small-tensor compensation, versus 8.589640668 for norm-only, 8.836560440 for original E8/W5 and 7.334175947 for FP16. Raw capacity is unchanged. The >=1% improvement condition passed, but the +5% source target remains unmet (+13.9851%). MK is deferred by explicit user direction, with no assumed Resurface recovery. The GitHub release remains a draft with zero assets.

## Scope

- Official pure NVIDIA Mamba-2 8B, Apache-2.0 source model; 56 layers, width4096, groups8, untied256000-token embedding/head.
- E8 projections using calibrated LDLQ, initial scale0.9, damping0.01, two coordinate sweeps; W5 group128 for each vocabulary matrix separately.
- Keep original architecture and recurrent-state dimensions. Quantization is lossy; subsequent entropy encoding must restore every quantized source byte exactly.
- Use public upstream runtime and tokenizer. Do not distribute private research code, calibration text or private checkpoints.
- File-size minimum is a goal. Any comparative claim must name audited same-base alternatives, date, complete storage scope and quality results.

## Source and repository

- [x] Create public EndlessChasing/mamba2-8b-e8w5 repository.
- [x] Pin official NVIDIA revision and LFS SHA256s.
- [x] Download and verify official checkpoint and native SentencePiece tokenizer on GPU host.
- [x] Record source/runtime/codec licenses and include redistribution notices.
- [x] Push reproducible source and progress documentation; keep weights outside Git.

## Correctness before compression

- [x] Load official checkpoint into public mamba_ssm runtime with exact key/shape coverage.
- [x] Verify grouped gated RMSNorm, 8 SSM groups and independent embedding/output weights.
- [x] Verify native tokenizer and baseline finite-logit/token-cache checks.
- [x] Freeze calibration/development/evaluation inputs and protocol before candidate evaluation.

## Compression

- [x] Collect train-only FP32 input covariances for all112 projections.
- [x] Generate and decode-check every E8 matrix; bind resumption to source/config/calibration/code hashes.
- [x] Stream W5 embedding and lm_head separately, including all scales and rounding rules.
- [x] Save all remaining model parameters and configuration; verify complete parameter coverage.
- [x] Build actual entropy-coded container with all tables, offsets, padding and metadata counted.
- [x] Verify the container against all raw file hashes and independent chunk decoding checks.
- [x] Exercise the end-user restore and generation path locally on the GPU host:118 raw files exact, archived software and restored tokenizer load/generate with no original checkpoint input; retain the publication hold.

## Diagnose PPL degradation before publication

- [x] Stop uploads and publication; verify the release is still a draft with zero assets.
- [x] Audit matched evaluation inputs, source mapping, runtime precision and per-window degradation.
- [x] Reproduce original and quantized development endpoints in one process; preserve and disclose the tiny historical quantized-endpoint discrepancy.
- [x] Measure all eight combinations of E8 projections, W5 embedding and W5 output head; attribute changes in NLL, including interactions.
- [x] Separate input/output projections and restore selected input-projection roles to locate sensitive pathways.
- [x] Audit rotation, balancing, calibration and LDLQ math independently.
- [x] Confirm the dominant component on the entire validation split without further test-set use: original PPL7.334168, E8-only8.729770, W5-only7.415406, complete8.836527; all264764 targets.
- [x] Write a diagnosis supported by controlled interventions and identify any remaining uncertainty: see docs/PPL_DIAGNOSIS.md.
- [x] Evaluate a targeted repair on validation data: all112 eight-sweep projections were verified; paired dev PPL9.237164→9.194595, MK10/12→10/12, controls0/12→0/12. Advancement gate failed; no full validation or promotion. The first candidate's test results have already been seen.
- [ ] Obtain new user direction before resuming model publication.

## Same-bitrate repair experiment

- [x] Freeze the refinement protocol and advancement gates before measuring new results.
- [x] Compare 2 versus 8 LDLQ tuning sweeps on six predeclared train-calibrated projections: layers0/18/55, input/output.
- [x] Require median original-H squared-error reduction >=1%, at least4/6 >=0.5%, and no regression >0.1% before expanding the experiment:PASS, median1.6574%,6/6 improve, exact baseline reproduction.
- [x] After the screen passed, build all112 refined projection files with unchanged index bitrate, sign/balance/scale formats, parent source and calibration; keep v1 immutable. All112 reconstruction proxies improve (median1.8970% squared-error reduction); complete CPU provenance/inventory verification passes.
- [x] Compare parent/candidate PPL and recall in one process. Required >=1% lower development PPL and no worse normal/removed-control recall counts; measured0.460844% improvement, so gateFAIL.
- [x] Apply the conditional stop rule: retain the negative result and stop this route. Full validation was skipped because the development gate failed; see docs/E8_REFINEMENT_RESULTS.md.

This is an improvement screen, not the original-model +5% PPL quality gate. Equal raw bit allocation does not establish equal Huffman archive bytes.

## Existing-norm compensation experiment

- [x] Declare a separate bounded protocol before training: docs/NORM_COMPENSATION_PROTOCOL.md. Use the original two-sweep parent and exactly 113 existing norm tensors / 692,224 parameters.
- [x] Implement an FP16-forward, FP32-master training path with correct checkpoint recomputation, full-vocabulary CE/KL, loss scaling and resumable state; 3 training and 10 overlay/evaluator CPU tests passed.
- [x] Pass training-only GPU smoke checks: native/functional and export/reload outputs bitwise equal; all 113 checkpoint gradients exact; full-vocabulary chunking checks passed; complete 2047-target update had finite gradients and no overflow; all 394 frozen tensor hashes unchanged. Peak allocated 35,774,360,064 bytes; trial state discarded.
- [x] Run the fixed four passes / 128 successful updates on the 32 calibration training windows: 128 attempts, zero overflows, 262,016 target exposures; only the final checkpoint exported.
- [x] Export and independently verify the complete small-tensor replacement: all 394 frozen GPU tensor hashes unchanged, all 113 rounded masters match export, all 280 protected small tensors exact, original E8/W5 inherited. Logical raw data remains 2,886,482,462 bytes.
- [x] Compare original parent and reloaded candidate in one process: PPL 9.237164→8.965396 (−2.9421%); normal MK 10/12→9/12; controls 0/12 unchanged. PPL gate passed; combined gate failed.
- [x] Preserve the negative combined result and stop this fixed recipe. Full validation was skipped; the original combined gate remains failed. The later PPL-only full validation is recorded separately below. See docs/NORM_COMPENSATION_RESULTS.md.

This route adds no inference tensor fields; actual archive and metadata bytes must still be measured. No publication or test-set evaluation is authorized by this experiment.

## Evaluation and release

- [x] Compare original and reconstructed quantized models on the same PPL and multi-key recall protocol.
- [x] Keep development screens distinct from full evaluation and numerical roundtrip checks.
- [x] Measure actual model bytes and evaluation memory; separate compressed file size from inference residency.
- [ ] Release model artifact, tokenizer/config, checksums, source revision and reproducible commands.
- [ ] Publish quality changes, limitations and a bounded comparison to available same-base releases.

Only code, small reports and metadata are retained on the Mac. Original weights, covariances and generated model binaries remain on the GPU host during experiments.

## User-directed PPL priority continuation

- [x] Record the user decision to prioritize PPL and defer recall work, while preserving the earlier failed combined gate: docs/PPL_PRIORITY_CONTINUATION.md.
- [x] Confirm the actual source and candidate architectures are pure Mamba-2: 56 Mamba blocks, zero attention/independent MLP/MoE; reports/pure_mamba2_architecture_audit.json.
- [x] Run separate full-validation PPL for original source, original E8/W5, and fixed norm-v1 export with exact paired inputs: 7.334176 / 8.836560 / 8.589641, all 264,764 targets; docs/NORM_PPL_RESULTS.md.
- [x] Freeze the next all-small-tensor protocol: 393 existing tensors, 256 disjoint train windows excluding the old 32, 1024 fixed updates.
- [x] Pass new CPU and GPU checks covering convolution and recurrent parameter gradients, resumption, and actual FP16 export: 393 finite-gradient tensors, all seven families nonzero, checkpoint gradients exact, reloaded FP16 output exact, 114 frozen weight hashes unchanged; reports/small_compensation_v1_smoke.json.
- [x] Train, export and independently audit the final checkpoint without changing E8/W5 or payload capacity: 1024 updates, zero overflows, 393 changed small tensors, 114 frozen matrices, raw data 2,886,482,462 bytes.
- [x] Measure complete validation PPL, source-relative gap and gain over norm-v1: PPL 8.359868, −2.6750% versus norm-v1, −5.3946% versus original E8/W5, +13.9851% versus FP16. Recall is unmeasured for this candidate. See docs/SMALL_TENSOR_COMPENSATION_RESULTS.md.
- [ ] Consider a separate Resurface recall experiment after the PPL work; no recall recovery is assumed.

## Next PPL hypothesis

- [x] Audit input-balance training versus already trained norm gains; they are redundant in exact arithmetic, with finite-precision differences. Do not count this as 688,128 new independent modes.
- [x] Design a bounded six-projection codeword-reassignment experiment using current-candidate inputs and native teacher outputs: docs/PROJECTION_CROSSMOMENT_PILOT_PROTOCOL.md. E8 raw bytes remain fixed; pilot inputs are disjoint TRAIN intervals.
- [x] Implement the anchored cross-moment solve and check it against independent augmented least squares, rank-deficient inputs and a self-target identity (five CPU tests passed).
- [x] Complete the four-arm full-validation projection/small-tensor interaction diagnostic: trained-small+E8 PPL8.359912 versus trained-small+source-projections7.412137, W5 fixed; all130 windows improve on restoration. The trained small tensors remain compatible with source projections. This diagnostic does not produce a compact candidate. See docs/PROJECTION_REPAIR_RESULTS.md.
- [x] Pass12 paired-statistics/anchored-solve CPU checks, including native teacher rounding and independent heldout scoring.
- [ ] Pass the discarded GPU pairing check, then compare current E8, H-refresh control and anchored cross-moment repair on six fixed projections (run started).
- [ ] Apply the declared native teacher-error and joint heldout TRAIN PPL advancement rule. Do not infer full-validation gain from local MSE.
