# Mamba-2 8B E8/W5 execution checklist

Status: publication remains on hold at the user's request. Prototype compensation completed 128 updates, export and independent full validation: PPL8.306585062 versus all-small8.359867548 and FP16 source7.334175947. This is the lowest measured compact-candidate PPL, but its0.6374% gain misses the declared >=1% advancement gate and its+13.2586% source gap misses the+5% target. Both gates failed; stop this fixed recipe and retain the accepted all-small model/distribution. Its own earlier advancement gate passed, with raw capacity unchanged. MK remains deferred with no assumed Resurface recovery. The GitHub release remains a draft with zero assets.

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
- [x] Compose the measured all-small candidate into a standalone flat raw package:393 actual small tensors and116 inherited files verified,507 tensor identities bound to the evaluation. Raw data2,886,482,462 bytes plus246,900-byte manifest; no original checkpoint required by the loader.
- [x] Build and fully read back its actual Huffman weights container:2,660,171,471 bytes,118 files and342 independently checked chunks. A separate post-build audit matches every container member to the pinned manifest ledger. See docs/ALL_SMALL_COMPOSITION.md; this excludes tokenizer/software/licenses/outer release metadata and does not establish a new quality result.
- [x] Build the currently measured all-small model's complete offline distribution from fixed exported source:2,671,176,471 bytes including20 assets and the outer manifest; actual CPU restoration reproduces all118 raw files. Preserve the first permission-failure receipt and successful second-attempt evidence. See docs/ALL_SMALL_DISTRIBUTION.md. No PPL remeasurement or GPU generation is inferred.
- [x] Verify the retained all-small candidate's archived-software loading/generation:209 archived files checked, restored raw manifest pinned, three short prompts generated and fresh-cache repetition exact. No original checkpoint argument or intercepted access attempt;51.8446 seconds. See reports/all_small_restored_inference_v2.json. This adds usability evidence, not PPL/MK or compressed GPU-memory validation. Publication remains held.
- [ ] Consider a separate Resurface recall experiment after the PPL work; no recall recovery is assumed.

## Next PPL hypothesis

- [x] Audit input-balance training versus already trained norm gains; they are redundant in exact arithmetic, with finite-precision differences. Do not count this as 688,128 new independent modes.
- [x] Design a bounded six-projection codeword-reassignment experiment using current-candidate inputs and native teacher outputs: docs/PROJECTION_CROSSMOMENT_PILOT_PROTOCOL.md. E8 raw bytes remain fixed; pilot inputs are disjoint TRAIN intervals.
- [x] Implement the anchored cross-moment solve and check it against independent augmented least squares, rank-deficient inputs and a self-target identity (five CPU tests passed).
- [x] Complete the four-arm full-validation projection/small-tensor interaction diagnostic: trained-small+E8 PPL8.359912 versus trained-small+source-projections7.412137, W5 fixed; all130 windows improve on restoration. The trained small tensors remain compatible with source projections. This diagnostic does not produce a compact candidate. See docs/PROJECTION_REPAIR_RESULTS.md.
- [x] Pass12 paired-statistics/anchored-solve CPU checks, including native teacher rounding and independent heldout scoring.
- [x] Pass the discarded GPU pairing check, then compare current E8, H-refresh control and anchored cross-moment repair on six fixed projections. Every replacement file has exact readback and unchanged raw capacity.
- [x] Apply the declared native teacher-error and joint heldout TRAIN PPL advancement rule: FAIL. Median local teacher MSE improves17.434%, but layer0.out_proj worsens4.381% and joint PPL8.423374→9.147629 (+8.598%). Stop this route without112-matrix expansion/full validation; docs/PROJECTION_REPAIR_RESULTS.md.
- [x] Establish an exact zero-correction decoder for fixed indices plus learned256×8 prototype corrections: all65,536 zero/nonzero codes checked, strict4128-byte files, seven codec CPU tests and one additional shared-occurrence gradient test passed.
- [x] Declare a separate language-loss experiment: docs/PROTOTYPE_COMPENSATION_PROTOCOL.md. All112 tables /229,376 FP32 masters; fixed128 CE/KL updates; baseline507 tensors frozen;112 serialized files add462,336 bytes plus manifest.
- [x] Prepare32 new TRAIN windows excluding original32/all-small256/cross-moment48 intervals; zero overlap. Six prototype-bank CPU tests pass, including full toy and single-block checkpoint gradients plus serialized native parity.
- [x] Pass discarded GPU smoke:112 zero-decode checks, real-block replay gradients, one full2047-target update, all112 serialized FP16 table/native-output parity, all507 baseline hashes unchanged. First update154.61s and36.57GB peak allocation. See docs/PROTOTYPE_SMOKE_RESULTS.md.
- [x] Complete the bounded decoder performance probe. The faster explicit-MM alternative failed decoded FP16 bitwise parity; retain the original validated decoder unchanged. See [probe results](docs/PROTOTYPE_DECODER_PERFORMANCE.md).
- [x] Verify the actual first periodic checkpoint on CPU:32 successful updates,zero overflows,65,504 targets,112 finite master tables/FP16 casts,optimizer step32,exact schedule/history/journal prefix and fileSHA. Saved frozen hashes remain startup evidence; no intermediate PPL or model selection. See docs/PROTOTYPE_RUNNING_JOB.md.
- [x] Complete the declared 128-update recipe and independently verify final export:128 successes/128 attempts,zero overflows,262,016 target exposures; all507 baseline tensor hashes unchanged. All112 table files match rounded final masters and add462,336 bytes plus142,637-byte manifest. Native export hidden/last-eight logits are bitwise equal on the128-token check. See the [completed job record](docs/PROTOTYPE_RUNNING_JOB.md).
- [x] Run independent complete paired validation for source/all-small/reloaded prototype:130 windows,264,764 targets, PPL7.334175947 /8.359867548 /8.306585062. All112 candidate projection identities and395 unchanged tensors verified;127 windows improve,3 worsen.
- [x] Apply both declared quality gates:FAIL. Prototype gain0.6374% is below1%; source gap+13.2586% is above5%. Stop this fixed recipe with no extra epochs or intermediate checkpoint selection; retain the all-small model/distribution. See [final results](docs/PROTOTYPE_COMPENSATION_RESULTS.md). Execution and integrity completion do not constitute a quality pass.

## Separate low-rank repair

The [rank-4 protocol](docs/LOW_RANK_COMPENSATION_PROTOCOL.md) covers all112
projections starting from the accepted all-small candidate, with the original
E8/W5 and small tensors frozen. The discarded GPU smoke passed, and the fixed
training job launched at2026-09-28 10:28:31UTC; no quality result is established.
See the [job record and status command](docs/LOW_RANK_RUNNING_JOB.md).
The [conditional capacity note](docs/CONDITIONAL_LOW_RANK_REPAIR.md)
and [CPU factor representation](docs/LOW_RANK_RESIDUAL_IMPLEMENTATION.md)
record the storage and native FP16 merge contract. This does not extend the
completed prototype recipe or promote its failed candidate.

- [x] Freeze the all112 rank-4 recipe, native FP16 merge, final-only evaluation and separate quality gates.
- [x] Prepare256 fresh TRAIN windows with zero prior overlap; fixed1024-update schedule and2,096,128 target exposures.
- [x] Pass nine bank CPU tests, four driver accounting tests and22-path input preflight.
- [x] Pass discarded GPU smoke: exact initial native output/replay gradients, one2047-target update in2.062858s without overflow, expected initial B/A gradients and both families after the update,112-file export/native parity and507 unchanged tensor hashes. See [smoke results](docs/LOW_RANK_SMOKE_RESULTS.md).
- [x] Prepare independent native evaluation, pass six CPU integrity/gate tests and launch the fixed sequential training/validation job after readiness checks. Publication remains held.
- [ ] Complete exactly1024 successful updates from fresh initialization, with no intermediate PPL selection.
- [ ] Independently verify final224 masters/112 factor files and native same-process source/all-small/candidate full-validation PPL; apply both predeclared gates. MK remains deferred.
