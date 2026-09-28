# Fixed soft Resurface: prospective continuation after a replay discrepancy

Declared after observing DEV results and **before full candidate PPL or fresh
CONFIRM evaluation**. Effective only after review, source/protocol pinning and
the terminal CPU audit below passes. No new fitting, candidate or inference
policy is selected. Publication remains on hold.

## Evidence and the one changed prerequisite

The ongoing strict DEV run measured current112/384 versus active346/384 normal
recall, removed-target matches0→0, and PPL7.6242504268035765→7.59503251999968
on64 windows /131,008 targets. These are already observed development results;
they do not establish independent recall generalization. Complete restored
controls and final integrity checks are still prerequisites, not assumed facts.

The historical baseline had111/384 normal recall and seven different generated
sequences despite matching prompt/weight identities. See
[the drift record](RESURFACE_MK_BASELINE_DRIFT.md). Its cause remains unresolved.
Exact historical cross-process generated-output replay is stronger than the
same-process paired quality comparison. This continuation removes **only that
historical MK equality prerequisite**, explicitly after observing its failure.
It retains every numerical MK/PPL quality threshold and requires independent
confirmation. This is a transparent protocol amendment, not a pass under the
original strict protocol.

Keep the original report, failure, process exit, code and
[training/evaluation protocol](RESURFACE_READAPTED_TRAINING_PROTOCOL.md)
unchanged. Do not rewrite `complete:false`, synthesize a passed original gate,
or silently substitute its baseline. The hard0 draft is **not triggered**:
paired soft PPL improved. Do not run its different numerical profile/probe.

## Immutable candidate and execution

| Bound item | SHA256 |
| --- | --- |
| Final1536 training report | `e64effbceda78b82488fd69608d6338fd2bb6e4b535c1d8eca12ddee789f70b6` |
| Final checkpoint | `03ed8da1d7c49fde4b70c053719eff19b715393d7b96dc41427f3cfc62af3fba` |
| Actual FP16 adapter | `1e9857feaf80ede6525cce839726883f6230c98a067cb226eb2f095693ea803e` |
| Adapter manifest | `5660676adb45806052cbaab7f5e50056769c4f5e106d5a49cf148f025bf2b7a8` |
| Accepted current PPL report | `3c6d840315b7552be5530ea884d0a9c620219df8dbbdad1a98b4cdd6e89ad1b4` |
| Original training protocol | `469063147d892282a2bdad8558afca8466d6f365f1efe14ecbba7417e9857dd6` |

Use all224 actual FP16 adapter tensors, verified against final1536 FP32 masters,
and the same507 base tensors. Keep native soft sigmoid gates enabled for every
MK and PPL token, native FP16 computation/cache policy and full256K head. No
retraining, threshold, scale, precision, kernel-profile or task-bypass change.
Preserve the strict runner's Torch precision flags, installed native sources
and environment policy. Record them; do not enable the prospective deterministic
profile or select autotune configurations to improve scores. Independently
rehash actual files and tensor contents using frozen helpers.

## Stage A: terminal DEV audit, CPU only

A new audit receipt must bind the original terminal report and process receipt
by actual SHA/bytes. It must independently establish all of the following:

- The process terminated; the only evaluation failure is the known historical
  current-MK output/row mismatch. No model, file, numerical, coverage, replay or
  cleanup failure is allowed. Preserve the original nonzero exit and failure.
- All three768-prompt arms finished, with384 normal/384 removed each, exact
  ordered input identities and12 successful fresh-cache replays per arm.
  Initial-current and restored-current complete rows, outputs, generated IDs,
  scores, summaries and cells are identical.
- All three64-window PPL arms contain exactly131,008 targets and the frozen
  window/token identities,64-token chunks and63-target tails. Recompute NLL,
  weighted PPL and the gates. Require initial/restored current rows and every
  CE chunk bitwise identical, and current equal to the accepted historical DEV
  PPL rows. Check these explicitly: the original historical MK exception occurs
  before its later PPL-restoration assertion.
- Actual initial/final and per-arm507 base content and identity checks,
  before/after224 active-adapter content checks, hook cleanup, actual final
  export/master proof and final bound-input rechecks are complete and agree.
- Recomputed normal MK gain is positive, removed-target matches do not
  increase, and active PPL is **≤ paired current PPL, with no epsilon**.

Report the seven historical sequence differences and the111→112 historical
baseline count separately. The new receipt may state that the **paired DEV
quality checks pass** while the **original strict run failed**. It must not
claim the latter completed successfully. If any audit prerequisite fails, stop.

## Stage B: complete paired PPL

Only after Stage A passes, load the existing complete validation130 windows /
264,764 targets. Use the same model/profile and native scorer in the order
**current → active soft → restored current**. Keep the actual soft export
enabled for all candidate windows. Verify all input identities, finite values,
full-vocabulary chunk/tail counts,507 base and224 adapter contents, immutable
parameter identities and final bound files.

Require current/restored rows and CE chunks to match exactly, and current to
match every accepted historical full-validation row, including aggregate
PPL7.622396587826496. Require active PPL **≤ this paired current PPL, without
epsilon or source+5% allowance**. A replay, identity or quality failure stops
continuation. No retries to obtain a passing numerical result.

These prose validation data have informed prior development; disclose that
history. This stage supplies the complete same-policy PPL measurement, not an
untouched generalization claim.

## Stage C: untouched confirmation, once

Only after Stage B passes, open the originally declared CONFIRM raw/token
split: seed2026092802,384 normal +384 removed prompts, disjoint numeric pools
and unchanged tokenizer/templates/control construction. Verify actual split
hashes and non-overlap. No CONFIRM examples or scores may have been inspected
or fitted before this stage; if that prerequisite is false, stop rather than
rename the data as fresh.

Score **current → active soft → restored current**, all with fresh native FP16
caches, full-vocabulary greedy max12-token generation, frozen six-digit
matching,12 exact within-arm replays and complete base restoration. Require:

- Positive normal gain and exact two-sided McNemar p<0.05.
- Positive conservative95% lower bound: one-sided97.5% Clopper–Pearson lower
  bound on gained/384 minus one-sided97.5% upper bound on lost/384.
- No increase in target-removed false recalls.
- Complete actual507/224 identities, same policy and final file/hook checks.

Bind both later-stage receipts to the same final export, native code/profile,
this protocol and passed prior-stage receipt. No tuning, checkpoint selection,
threshold change, extra confirmation attempt or promotion before both complete
PPL and confirmation gates pass. Even a pass establishes only these finite
tests; it is not universal recall preservation or exact bitwise PPL equality.
No publication or compact-runtime/performance/ASIC claim is authorized.
