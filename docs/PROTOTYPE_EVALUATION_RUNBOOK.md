# Independent prototype compensation evaluation

The sole eligible export is the final128-successful-update candidate from
[the fixed protocol](PROTOTYPE_COMPENSATION_PROTOCOL.md). Use
`scripts/evaluate_prototype_compensation.py`; the report refuses overwrite.
GPU scoring requires the explicit training-process handoff from the root task.

## CPU integrity

The evaluator independently checks the source/parent/all-small/protocol/data and
software bindings, every4128-byte table file and its payload hash, the exact112
file inventory, final checkpoint provenance, all128 scheduled successful updates
and up to8 correctly retried overflows. All112 FP32 checkpoint masters must round
bitwise to the actual serialized FP16 tables. A non-final checkpoint, altered
table, changed file ledger or mismatched smoke/training receipt is ineligible.

The CPU check does not import or invoke the prototype training bank. It also
reconstructs the declared fresh32 TRAIN window selection and checks token hashes.

```sh
/home/horde/.venvs/lodram/bin/python scripts/evaluate_prototype_compensation.py \
  --verify-only \
  --overlay-dir artifacts/prototype_compensation_v1 \
  --training-report reports/prototype_compensation_v1_train.json \
  --training-checkpoint EXACT_FINAL_CHECKPOINT_FROM_TRAINING_RECEIPT \
  --smoke-report reports/prototype_compensation_v1_smoke.json \
  --report reports/prototype_compensation_v1_integrity.json
```

Run from the remote repository. Defaults point to the pinned source, original
E8/W5 parent, all-small overlay, fresh prototype training data and protocol.
Checkpoint filenames must come from the completed training receipt. No resume,
checkpoint search or historical-report substitution is performed.

## Native quality check after GPU handoff

Use the same command without `--verify-only`, with the distinct report path
`reports/prototype_compensation_v1_eval.json`. The integrity checks run again.
Keep logs on the GPU host and copy small JSON receipts to the Mac.

Three arms run in one process: source FP16, unchanged best all-small E8/W5, and
the reloaded prototype candidate. Every arm uses the frozen complete validation
stream:264,764 targets,130 windows,2048 maximum targets/window, fresh state,
prefill execution and64-token logit chunks. The original source is freed before
loading the E8/W5 baseline.

The freshly loaded parent receives a full507-parameter audit, including an
independent CPU W5 decoder. The all-small baseline is then checked against its
actual393-tensor export and pinned507-tensor hash ledger. Candidate projections
are decoded one at a time using `read_e8`, `read_prototypes` and the foundation's
`decode_e8_with_prototypes`, then installed in the native model. No training-bank
forward is used for PPL. Record all112 decoded FP16 hashes, verify all395 retained
small/vocabulary tensors by actual content, and check507 tensors /8,236,999,680
parameters after installation.

Report complete paired window losses, aggregate target-weighted NLL/PPL, the
predeclared1% PPL gain against the same-process all-small baseline, and the
separate source+5% target. Count462,336 added table-file bytes and actual manifest
bytes. These are expanded-FP16 quality measurements, not a compact-residency or
throughput result. No test split, MK, further training or publication is included.
