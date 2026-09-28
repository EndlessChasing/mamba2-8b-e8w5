# Mamba-2 8B E8/W5 execution checklist

Status: the user authorized publication of the latest fixed-soft Resurface candidate. Full validation PPL is **7.593163114** versus source **7.334175947** (+3.53%). Independent MK confirmation is **340/384** versus compressed base **91/384**, with removed-target matches **0/384** for both. The current raw base-plus-adapter data are **3,141,468,439 bytes**. Release packaging, independent restore and archived-source inference checks are in progress for **v0.2.0-resurface**; older candidate failures and artifacts remain retained.

## Current publication checklist

- [x] Receive explicit user authorization to publish after the source comparison.
- [x] Fix release identity to the validated axis/readapted base plus enabled soft adapter.
- [ ] Build complete portable package, tokenizer, corresponding source and licenses.
- [ ] Restore all released files exactly and run inference using the archived source.
- [ ] Upload and verify each asset checksum, then publish the research prerelease.
- [ ] Confirm public release URL and publish exact download byte counts.

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
- [x] Obtain new user direction before resuming model publication: explicit publication authorization received after the final source comparison.

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
E8/W5 and small tensors frozen. The fixed training and independent validation
completed with exit code zero, but both quality gates failed: PPL 8.630189440
versus base 8.359867548 and source 7.334175947. See the
[negative result](docs/LOW_RANK_COMPENSATION_RESULTS.md) and
[completed job record](docs/LOW_RANK_RUNNING_JOB.md).
The [conditional capacity note](docs/CONDITIONAL_LOW_RANK_REPAIR.md)
and [CPU factor representation](docs/LOW_RANK_RESIDUAL_IMPLEMENTATION.md)
record the storage and native FP16 merge contract. This does not extend the
completed prototype recipe or promote its failed candidate.

- [x] Freeze the all112 rank-4 recipe, native FP16 merge, final-only evaluation and separate quality gates.
- [x] Prepare256 fresh TRAIN windows with zero prior overlap; fixed1024-update schedule and2,096,128 target exposures.
- [x] Pass nine bank CPU tests, four driver accounting tests and22-path input preflight.
- [x] Pass discarded GPU smoke: exact initial native output/replay gradients, one2047-target update in2.062858s without overflow, expected initial B/A gradients and both families after the update,112-file export/native parity and507 unchanged tensor hashes. See [smoke results](docs/LOW_RANK_SMOKE_RESULTS.md).
- [x] Prepare independent native evaluation, pass six CPU integrity/gate tests and launch the fixed sequential training/validation job after readiness checks. Publication remains held.
- [x] Complete exactly 1,024 successful updates from fresh initialization: 1,024 attempts, zero overflows, 2,096,128 target exposures and no intermediate PPL selection.
- [x] Independently verify final 224 rounded masters, 112 factor files, native export parity and all 507 frozen tensor hashes. Factor files occupy 15,658,496 bytes plus a 198,517-byte manifest; no new complete distribution or compact inference-memory result is claimed.
- [x] Measure native same-process source/all-small/candidate full-validation PPL on 264,764 targets and 130 windows: 7.334175947 / 8.359867548 / 8.630189440. Four windows improve and 126 worsen. Both the >=1% advancement and source+5% gates fail.
- [x] Preserve the negative result and stop this fixed recipe without extra epochs, checkpoint selection or a rank sweep. Retain the accepted all-small model/distribution; MK remains deferred and publication held.
- [x] Complete the final-only fit-versus-fresh TRAIN diagnostic: fitting PPL8.714097→4.247930, fresh PPL8.510325→8.691613; all64 fresh teacher-KL values worsen. All320 repeated source CE rows are exact; the112 merged/507 complete candidate hashes match final validation. This supports overfitting without isolating causality. See [diagnostic results](docs/LOW_RANK_GENERALIZATION_RESULTS.md).

## Separate teacher-KL compensation

- [x] Declare the [fixed protocol](docs/TEACHER_KL_COMPENSATION_PROTOCOL.md): reset accepted all-small and fresh rank4 factors, pure teacher KL with CE logging only, learning rate1e-4,448 fresh TRAIN windows once and64 reserved windows. Exclude every previous fitting and observed diagnostic interval;7 eligible windows remain unused.
- [x] Prepare and verify the new token sets, pure-KL gradients and independent evaluator:4 helper/data,5 trainer and8 evaluator CPU checks pass. The discarded GPU smoke also passes:one2047-target pure-KL update in2.194529s with zero overflow,112-file native export parity and507 frozen hashes unchanged; trial state discarded. See [smoke results](docs/TEACHER_KL_SMOKE_RESULTS.md).
- [x] Complete exactly448 successful updates:448 attempts,zero overflows,917,056 targets. Independently verify final224 rounded masters/112 files,507 baseline and112 merged/395 inherited candidate tensors; no intermediate quality selection.
- [x] Evaluate all64 reserved windows/131,008 targets. PPL8.397298749→8.571699071 (+2.0769%) fails the>=1% gain requirement; mean teacher KL.248099897→.194325164 passes. Combined gateFAIL; all64 source CE repeats are exact. See [completed result](docs/TEACHER_KL_COMPENSATION_RESULTS.md).
- [x] Apply the conditional stop rule:skip full validation after the failed reserved gate, retain empty full-validation results and leave this candidate's full-validation/source+5% target unmeasured. Stop the fixed recipe without extra updates, earlier checkpoint selection, promotion, MK or publication. Teacher matching generalized but target-token PPL worsened; this does not validate pure KL as a repair or isolate a cause.


## Fixed-beta output diagnostic

- [x] Preserve the v1 exact historical CE-control failure and its zero accepted rows. Complete the bounded window-0 replay: all nine same-process comparisons exact; earlier cross-process drift remains unexplained.
- [x] Freeze the [separate v2 protocol](docs/OUTPUT_CALIBRATION_DIAGNOSTIC_V2_PROTOCOL.md): beta1 only, same64 observed reserved TRAIN windows, strict same-process controls and a descriptive0.1% historical PPL drift ceiling; no fit or advancement gate.
- [x] Complete paired source/base and source/candidate scoring on131,008 targets. First-window frozen/new CE+KL and all64 repeated source controls are exact; all507 identities and112/395 merge/inheritance checks pass. All three aggregate PPL drifts are0. See [diagnostic results](docs/OUTPUT_CALIBRATION_DIAGNOSTIC_RESULTS.md).
- [x] Record entropy, slope, curvature, strict ranks/ties and fixed source-surprisal bins: base slope+0.0728683 locally favors higher T; candidate slope-0.0359664 favors lower T. Candidate top1 falls53.4715%→52.9273%; six NLL windows improve and58 worsen, with all six difficulty-bin totals worse.
- [x] Keep interpretation descriptive: no beta fit, Newton candidate, sweep, measured finite gain or dominant-cause claim. Retain the accepted model/distribution and failed teacher-KL gate; no new validation/test/MK or publication.

## Separate scalar-temperature experiment

- [x] Declare the [bounded scalar-temperature protocol](docs/SCALAR_TEMPERATURE_PROTOCOL.md) for accepted all-small only: a fixed TRAIN fitting budget, a separate seven-window screen and conditional complete validation. This declaration does not fit a scalar or establish a PPL gain.
- [x] Pass bounded CPU derivative/tail, scalar roundtrip and input checks; complete the exact same-process beta1 CE control and unchanged507 model-content audits.
- [x] Complete three fixed updates on64 TRAIN windows/131,008 targets per pass, then the fourth final TRAIN evaluation. Export/reload beta0.9864529967308044 (T1.013733044872986), four payload bytes plus96,550-byte manifest; fit PPL8.185296703→8.182022042 improves0.0400066%.
- [x] Score the seven separate, previously unused TRAIN windows/14,329 targets: source7.008538754, base8.219889723, scaled8.215166553. Five windows improve, two worsen and argmax changes0. The0.0574603% reduction fails the1% screen. See [final results](docs/SCALAR_TEMPERATURE_RESULTS.md).
- [x] Apply the declared stop: full validation not performed, no fitting expansion/intermediate selection/promotion. Complete-validation/source+5% results remain unmeasured. Retain the accepted model/distribution; no new MK or publication. The small benefit does not prove all calibration methods ineffective.

## W4 vocabulary penalty diagnostic

- [x] Declare and complete the [fixed four-arm protocol](docs/VOCAB_W4_DIAGNOSTIC_PROTOCOL.md) on64 already-observed reserved TRAIN windows/131,008 targets per arm; no fitting or full validation.
- [x] Write/read actual group128 W4 files from original BF16 checkpoint values through writer FP32 row conversion. Each file is540,672,079 bytes and saves131,072,000 bytes against W5; both save262,144,000 raw bytes. No new complete or entropy-coded package is claimed.
- [x] Preserve the initial CPU signed-zero oracle failure, correct expected zero representation to the existing unsigned-code format, and pass bounded CPU checks. Codec/runtime, protocol and model weights remain unchanged.
- [x] Verify all507 expected hashes before/after every arm, with all112 E8 and393 small tensors fixed. Restore W5/W5 and reproduce all64 baseline windows/2048 CE chunks exactly.
- [x] Measure baseline8.397361922, W4 embedding8.410709113 (+0.158945%), W4 head8.843041048 (+5.307371%) and both8.859629527 (+5.504915%). Embedding improves18/64 windows; head/both improve0/64. See [final diagnostic results](docs/VOCAB_W4_DIAGNOSTIC_RESULTS.md).
- [x] Retain diagnostic limits: observed TRAIN, no fresh/full-validation/test/MK quality, no promotion/publication, no guarantee vocabulary savings produce a useful combined model.
- [x] Declare the separate [input-axis residual protocol](docs/INPUT_AXIS_RESIDUAL_PROTOCOL.md) for nominal2.5-bit enhanced input projections with W4 embedding/W5 head.
- [x] Implement and independently review the strict20-bit quantizer/reader and native evaluator; preserve frozen codec/runtime and pass bounded CPU checks.
- [x] Complete all56 input projections from original BF16 weights/original TRAIN Hessians, with exact packed-index and finite FP16 readback. Actual new files1,331,009,669 bytes replace1,064,932,680 bytes; delta266,076,989 includes829 header bytes. See [quantization status and storage](docs/INPUT_AXIS_RESIDUAL_RESULTS.md).
- [x] Pass independent actual-overlay CPU preflight:274 bound inputs,56 new files,61 inherited file entries and complete507-tensor ledgers; expected changes0/56/57/58. This is integrity evidence, not PPL.
- [x] Complete all four observed-development arms and exact baseline repeat: baseline8.397361922, enhanced/W5/W5 control8.156725860, fixed primary8.168229289 and enhanced/W4/W4 control8.595606029. The primary's2.728626% improvement passes its1% gate; controls do not replace it.
- [x] Reload the primary from actual files and complete source/baseline/primary validation on130 windows/264,764 targets:7.334175947 /8.359867548 /8.109428666.127 windows improve,3 worsen;1% improvementPASS, source+5% targetFAIL at10.57041% gap.
- [x] Finish native507-tensor audits, final baseline restoration and bound-file rechecks; terminal report complete, exit0 and evaluator process absent. See [final result and receipts](docs/INPUT_AXIS_RESIDUAL_RESULTS.md).
- [x] Retain the measured improvement and its limits: primary raw data3,021,487,451 bytes are4.67715% larger, no complete axis package/test/MK/promotion/publication. Keep the historical accepted all-small distribution unchanged.

## Output-axis residual experiment

- [x] Freeze the [output-axis protocol](docs/OUTPUT_AXIS_RESIDUAL_PROTOCOL.md) before implementation: retain the measured input-axis candidate and enhance exactly 56 output projections, using original weights/Hessians and the same fixed axis4 recipe.
- [x] Implement and independently review the separate output quantizer/verifier, preserving the completed input experiment and original codec/runtime.
- [x] Complete the CUDA-hidden CPU preflight: 314 bound files, 56 output Hessians and 61 inherited files pass; evaluator CPU tests and independent code review also pass.
- [x] Complete all 56 output matrices with exact stored-index and finite FP16 readback; process exited 0. New files total 588,217,549 bytes and replace 470,776,208 bytes. Resolved raw candidate data are 3,138,928,792 bytes, adding 117,441,341 bytes including changed headers.
- [x] Pass the independent actual-overlay CPU preflight with 392 bound files, full paired-model tensor ledgers and no CUDA initialization; start the native PPL evaluation.
- [x] Independently score paired input-only baseline, output-enhanced candidate and exact baseline repeat on 64 observed TRAIN windows /131,008 targets: PPL 8.168229289→8.013059657 (1.89967% improvement; 61 improve, 3 worsen). All 64 repeated baseline windows /2,048 CE chunks are exact; the fixed 1% gate passes.
- [x] After the development gate passed, reload actual output files and complete same-process source/paired-input/candidate validation on 130 windows /264,764 targets: PPL 7.334175947 /8.109428666 /7.977512009. The further 1.62671% reduction passes the improvement condition; 116 windows improve and 14 worsen. The source +5% target still fails at an 8.77176% gap.
- [x] Finish all 507 actual tensor audits, exact token/window pairing, final paired-baseline restoration and bound-file rechecks. The final report is complete, the process exited 0 and is absent. See [final results and receipts](docs/OUTPUT_AXIS_RESIDUAL_RESULTS.md).
- [x] Record actual raw data 3,138,928,792 bytes: 117,440,512 extra code bytes plus 829 changed-header bytes above the paired baseline. Historical cumulative PPL improvement versus all-small is 4.57370%, not a fourth same-process arm. Retain the tested gain and unchanged historical distribution; no new complete package, promotion, test/MK measurement or publication. This fixed recipe is complete.

## PPL objective

- [x] Reduce complete-validation PPL from7.977512009 to at most7.700884745 (paired source+5%): the fixed current-axis small-tensor readaptation reaches7.622396588, source+3.92983%, with zero raw weight-file byte delta. This PPL target is met; recall/Resurface and publication remain deferred.

## Same-capacity joint E8/axis selection

- [x] Freeze the [joint-selection protocol](docs/JOINT_AXIS_SELECTION_PROTOCOL.md) before implementation or measurements: enumerate 16 signed-axis candidates with the greedy incumbent, preserving the current 20-bit files' metadata and decode format.
- [x] Pass CUDA-hidden CPU checks and independent implementation review: 34 bound inputs, six Hessians, five inherited files and 395 nonprojection identities; CPU receipt complete with CUDA uninitialized.
- [x] Reproduce all six fixed current files (layers 0/18/55, input/output) exactly: indices, full raw bytes/metadata and GPU-decoded FP16 identities. All replay checks passed before joint selection.
- [x] Complete the six-matrix original-TRAIN-H screen: all six improve by 1.71120%–1.92112%; median 1.839555%, zero of six at least 2%. The 5%/four-at-2% expansion gate fails. Screen exit 0, process absent; independent CPU audit of all 12 files and gate passes. See [final screen results](docs/JOINT_AXIS_SELECTION_RESULTS.md).
- [x] Apply the declared stop: no all-112 joint overlay, PPL evaluation, checkpoint selection or publication. Pilot replacement data have exactly zero byte delta; the six joint matrix stages cost 5.4934x the greedy replay time. Current best complete-validation PPL remains 7.977512009; the remaining PPL objective is still open.

## Current-axis small-tensor readaptation

- [x] Freeze the [diagnosis and readaptation protocol](docs/AXIS_SMALL_READAPTATION_PROTOCOL.md) before measurements, following the user's option1: fixed current-axis baseline,393 existing small tensors,448 TRAIN windows once,lr3e-5 and mixed CE/KL.
- [x] Complete current-old393 / current-source393 / restored-old393 diagnostic on64 observed reserved TRAIN windows: PPL8.013059657 /8.190893923 /8.013059657. Original small values worsen PPL2.21931%; current adaptation remains useful. Fixed114/507 audits and all64 repeated windows pass, exit0. The diagnostic does not select training initialization. See [results](docs/AXIS_SMALL_READAPTATION_RESULTS.md).
- [x] Pass CPU input checks and current-axis native/functional/checkpoint/chunk/export smoke. Checkpoint and chunk hidden gradients match exactly; one2047-target update succeeds with zero overflow; actual FP16 native export matches. Fixed114 and underlying/teacher507 audits pass; all smoke updates discarded. Start formal training from fresh process/optimizer/scaler/RNG.
- [x] Starting from current old-trained393, finish448 successful updates/448 attempts/zero overflows and917,056 targets. All393 changed exports equal final448 rounded masters; independent CPU verification passes with CUDA uninitialized. Fixed114, underlying507 and teacher507 audits pass. Actual replacement file7,283,930 bytes gives zero raw-byte delta; total3,138,928,792 bytes plus239,782-byte new manifest. Native PPL evaluation follows.
- [x] Independently score current/readapted/restored-current on64 reserved windows:8.013059657→7.624250427,4.85219% gain,64/64 improve and exact baseline repeat. The1% gate enables complete source/current/readapted validation:7.334175947 /7.977512009 /7.622396588 on130 windows/264,764 targets. Paired improvement4.45146%,130/130 improve; source+3.92983% passes the+5% target. All507/114, tail, restoration and final input checks pass; reportcomplete/exit0/processabsent. No new complete package, MK or publication. See [final results](docs/AXIS_SMALL_READAPTATION_RESULTS.md).

## User-directed MK validation and Resurface repair

- [x] Freeze the [Stage1 MK baseline protocol](docs/MK_RESURFACE_PROTOCOL.md) before measurements: original source and actual final448 readapted current model,384 normal+384 removed-binding prompts per arm, native prefill/recurrent decode, no fitting or publication.
- [x] Pass current-file/prompt CPU verification and independent baseline evaluator review. Preserve the initial Wilson endpoint self-test failure and corrected attempt2: all actual final448/export/prompt checks pass, CUDA uninitialized, exit0.
- [x] Complete paired MK baseline: source167/384(43.4896%), current111/384(28.9063%); both removed-target0/384. Current loses14.5833pp, paired12 gained/68 lost, exact McNemar p1.20165e-10. All507 audits and12 prompt replays per arm pass, exit0/processabsent. See [MK and repair results](docs/RESURFACE_READAPTED_RESULTS.md).
- [x] Declare the separate [native no-EMA Resurface recipe](docs/RESURFACE_READAPTED_TRAINING_PROTOCOL.md) before training: post-D soft gated readout,1,154,104 new parameters,1536 fixed updates and prose CE/KL/closure protection. All507 base tensors stay frozen. Prepare and independently retokenize1536 disjoint TRAIN cases: all answers require7 tokens,10,752 answer targets, maximum prompt1236 tokens. Mixed first-digit value pools avoid a fixed answer-prefix shortcut; confirmation is not yet prepared or scored.
- [x] Validate adapter identity, cache-step placement, checkpoint gradients and multi-token answer loss on actual8B. Zero/FP16-export/recurrent-cache and restored-base outputs match exactly; all224 checkpoint gradients match exactly; one MK+511-prose trial succeeds with zero overflow and is discarded. Underlying and teacher507 audits pass. GPU smoke exit0/processabsent.
- [x] Fit only declared TRAIN data with frozen backbone:1536 successful updates,1541 attempts,five identical-pair overflow retries,10,752 MK answer/784,896 prose targets. All224 finalFP16 tensors match actual checkpoint1536; native export and underlying/teacher507 audits pass. Independent actual-file CPUverification passes with CUDAuninitialized. Candidate file2,539,647 bytes plus294,171-byte manifest; no intermediate selection or publication. Native joint DEV evaluation started.
- [x] Complete all DEV measurements: same-process MK112→346/384, removed-target0→0, PPL7.624250427→7.595032520. Restored768 MK rows and64 PPL rows/CE chunks are exact; final507/224 and169-file checks agree. Preserve original strict exit1/complete:false from seven historical cross-process MK sequence differences (one correctness change111→112). No promotion.
- [x] Declare the separate [fixed-soft continuation protocol](docs/RESURFACE_SOFT_CONTINUATION_PROTOCOL.md), explicitly removing only historical cross-process MK equality while retaining the failed original receipt, same candidate/profile and all quality thresholds. Hard0 is untriggered.
- [x] Pass terminal independent CPU reanalysis of the unchanged failed strict report: all3×768 MK/12 replays,3×64 PPL, restored/historical chunks,507/224/export and bound files pass. Paired236 gained/2 lost, removed0→0 and PPL7.624250427→7.595032520 pass unchanged thresholds. Original strict failure remains retained; CPU exit0/CUDA uninitialized.
- [x] Complete full130-window same-enabled-model PPL:7.622396588→7.593163114, all130 windows improve,264,764 targets/4,137 CE chunks and572-target tail accounted. Historical/restored baseline exact;507/224/167-file checks and independent receipt audit pass, exit0/processabsent. Only then prepare previously absent CONFIRM:768 unique disjoint-instance prompts, exact retokenization, CPU exit0.
- [x] Complete once-only independent384+384 CONFIRM: normal91→340/384 (23.6979%→88.5417%),251 gained/2 lost, controls0→0, McNemar p4.43996e-72, conservative95% improvement lower bound+58.502343pp. All768 restored rows/12 replays,507/224 and170-file checks pass; exit0/processabsent. Independent terminal CPU audit passes. Same final soft adapter passes MK plus full PPL non-regression; no publication/new complete distribution, and the original strict DEV replay failure remains preserved.
