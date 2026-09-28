# Teacher-KL compensation: completed negative result

Formal training and independent native evaluation completed with exit code0.
The **reserved advancement gate failed**: teacher KL improved, but PPL worsened.
Full validation was skipped exactly as declared. Stop this fixed recipe with
no extra updates, intermediate checkpoint selection or promotion. The accepted
all-small model and its offline distribution remain unchanged; publication
remains held and MK deferred.

## Fixed training and final export

The [frozen protocol](TEACHER_KL_COMPENSATION_PROTOCOL.md) resets to accepted
all-small plus fresh rank-4 factors on all112 projections. All507 base tensors
remain frozen. The224 FP32 masters contain7,827,456 values; native forward casts
factors to FP16, forms their product and sum with the base in FP32, then casts
the effective projection to FP16. No failed candidate's factors are inherited.

Training uses pure teacher→student KL at temperature1 across the full256,000
token vocabulary. CE is logged with gradient coefficient0. AdamW learning rate
is1e-4. The448 new TRAIN windows are visited once in the fixed seeded order;
each contributes2047 targets. The64 reserved windows are excluded from fitting.

Measured completion:

- **448 successful updates /448 attempts /zero overflow retries**.
- **917,056 successful and attempted target exposures**.
- Every training record reports finite factor gradients, optimization loss
  equal to teacher KL and CE logging only. Only the final448-step checkpoint
  was exported; no intermediate quality measurement or checkpoint selection.
- All112 factor-file readbacks match the rounded final224 masters. The
  training report binds the final checkpoint, file ledger and passed smoke.
- All507 frozen base hashes are unchanged before/after export and agree with
  the overlay manifest's baseline ledger.
- On the128-token export check, native hidden states and last-eight-token
  full-vocabulary logits match the functional path bitwise; maximum difference0.

The final training checkpoint remains on the GPU host:100,339,579 bytes,
SHA256 `89497a03a8ddb082bc571fd2f21692cfda938249d89364aefedbc89a8d2431e0`.
The independent evaluator additionally verified the actual final224 rounded
masters/112 files,507 loaded baseline tensors,112 merged projections and395
inherited tensors. The complete507 candidate content hashes agree before and
after reserved scoring; the evaluation binds this exact final export.

## Storage and execution

| Scope | Bytes |
| --- | ---: |
| Additional FP16 factor values |15,654,912 |
|112 headers,32 bytes each |3584 |
| Complete factor files |15,658,496 |
| Actual overlay manifest |199,125 |
| Factor files plus overlay manifest |**15,857,621** |
| Existing logical base data |2,886,482,462 |
| Logical base data plus factor files, excluding manifests |**2,902,140,958** |

The logical-data total excludes tokenizer, software, licenses, metadata,
reports and optimizer checkpoints. No new complete container or distribution
has been built for this candidate. The accepted all-small complete distribution
remains2,671,176,471 bytes and does not contain these factors.

Training elapsed**750.5572 seconds**, including loading/export checks. Peak
allocated GPU memory was46,416,439,808 bytes and peak reserved was47,745,859,584
bytes; this includes temporary native export verification. These measurements
use dense FP16 base weights and do not establish compressed GPU residency or
isolated throughput.

## Reserved quality: combined gate failed

All three native models were scored on the same**64 reserved TRAIN windows /
131,008 targets**. Each window scores2047 targets with fresh state, FP16 head
GEMMs and FP32 full-vocabulary CE/KL in64-token chunks plus the63-token tail.
These windows were excluded from fitting. Source CE repeated exactly across
both student passes, per chunk and for all64 windows. Token hashes, starts and
target counts match the frozen reserved manifest.

| Native model | Reserved PPL | Mean teacher→student KL |
| --- | ---: | ---: |
| Original source cast to FP16 |7.369086030597358 |0 by definition |
| Accepted all-small |8.397298748944545 |0.24809989710231495 |
| Final teacher-KL candidate |8.571699071423234 |0.19432516351822648 |

Candidate PPL worsens**2.0768622%** versus the accepted base. Its mean teacher
KL improves**21.6746295%**. Six windows improve and58 worsen in target-token
NLL; all64 improve in teacher KL. The evaluator took285.5543 seconds.

| Predeclared reserved condition | Threshold | Outcome |
| --- | --- | --- |
| At least1% PPL improvement |Candidate PPL <=8.313325761455099 |**FAIL** |
| Strictly lower mean teacher KL |Candidate KL <0.24809989710231495 |**PASS** |
| Both conditions required |PPL and KL conditions together |**FAIL** |

The failed gate correctly prevented validation data loading/forward computation.
The full-validation result collection is empty. **This candidate's full-validation
PPL and source+5% target are unmeasured**, not an additional measured failure.
There is no test or MK result for it.

Teacher-distribution matching generalized to the reserved windows, while
prediction of the observed target tokens worsened. KL weights the complete
vocabulary by teacher probabilities; target-token CE scores the observed next
word. Improvement of one does not ensure improvement of the other. These
measurements do not support simple teacher-distribution drift as the explanation
for this candidate's regression, and pure KL is not a validated PPL cure. The
experiment changes loss, learning rate, exposure and data together, so it does
not establish one causal mechanism.

The accepted all-small**full-validation PPL8.359867548009992** and its existing
2,671,176,471-byte distribution remain unchanged. The prior prototype's
full-validation PPL8.306585061749145 is still the lowest measured compact-candidate
value on that split, but failed its1% advancement gate and was not promoted.
Those full-validation values are separate from the reserved-set scores above.
No extra epochs, checkpoint selection, rank sweep or publication follows.

## Completed receipts

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
|[Training](../reports/teacher_kl_compensation_v1_train.json) |29,116,870 |`216bfd2deed4b0bb70f24b152b01aba02ef95924fb56af9b07facde4deadfa04` |
|[Candidate manifest](../reports/teacher_kl_compensation_v1_manifest.json) |199,125 |`cd1c260c3e92146618322431d845101ea7b84468c5fa759769f843a899445300` |
|[Training process/exit](../reports/teacher_kl_compensation_v1_train_process.json) |2817 |`bd78082e4b059871876ea708a2f06c831a35738b0f13f97c674353e11122b49f` |
|[Independent reserved evaluation](../reports/teacher_kl_compensation_v1_eval.json) |1,182,228 |`d8acf27dfa500d18a6d4b25dc6d95e3edaa10a0ccc27f49369cb02ef2649e59a` |
|[Evaluation process/exit](../reports/teacher_kl_compensation_v1_eval_process.json) |3297 |`7b4d1cf16d1f0dd7679b5bd79420da0c36386a9f6872f03f35bc23b70511305e` |

See [readiness](TEACHER_KL_READINESS.md) and the
[discarded smoke result](TEACHER_KL_SMOKE_RESULTS.md) for pre-training evidence.
