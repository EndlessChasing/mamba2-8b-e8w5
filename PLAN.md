# Mamba-2 8B E8/W5 execution checklist

Status: publication is on hold at the user's request. Diagnosis identifies E8 projection quantization as the dominant quality loss, confirmed over the entire validation split. Local package restoration and generation passed. The bounded eight-sweep experiment stopped after missing its predeclared advancement threshold. A distinct norm-only compensation experiment on the original parent is now being implemented under a fixed protocol. The GitHub release remains a draft with zero assets.

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
- [ ] Implement an FP16-forward, FP32-master training path with correct checkpoint recomputation, full-vocabulary CE/KL, loss scaling and resumable state.
- [ ] Pass training-only smoke checks for forward/export parity, checkpoint gradients, loss chunking, selected/frozen gradients and memory.
- [ ] Run the fixed four passes / 128 successful updates on the 32 calibration training windows; preserve all attempts and use only the final checkpoint.
- [ ] Export and independently verify the complete small-tensor replacement, with every non-norm tensor unchanged and all original E8/W5 files inherited.
- [ ] Compare original parent and reloaded candidate on frozen development PPL/MK in one process; require at least 1% PPL gain and recall/control nonregression.
- [ ] If that gate passes, confirm original/parent/candidate on full validation; otherwise preserve the negative result and stop this fixed recipe. Keep the source-relative quality target separate.

This route adds no inference tensor fields; actual archive and metadata bytes must still be measured. No publication or test-set evaluation is authorized by this experiment.

## Evaluation and release

- [x] Compare original and reconstructed quantized models on the same PPL and multi-key recall protocol.
- [x] Keep development screens distinct from full evaluation and numerical roundtrip checks.
- [x] Measure actual model bytes and evaluation memory; separate compressed file size from inference residency.
- [ ] Release model artifact, tokenizer/config, checksums, source revision and reproducible commands.
- [ ] Publish quality changes, limitations and a bounded comparison to available same-base releases.

Only code, small reports and metadata are retained on the Mac. Original weights, covariances and generated model binaries remain on the GPU host during experiments.
