# Conditional Resurface hard0 evaluation — draft

**Status: prospective draft, not frozen or authorized for execution.** No hard0
result exists. Activate only after the current soft run terminates and its
paired measurements show **positive normal MK gain and worse prose PPL**,
with intact model/data identities and exact same-process restored controls.
An incomplete measurement does not establish this trigger. Root must bind the
terminal evidence and freeze this separate protocol before any GPU run.

Preserve the [soft protocol](RESURFACE_READAPTED_TRAINING_PROTOCOL.md), code,
reports and failures unchanged, including failure caused only by historical
MK replay. This experiment cannot turn a soft failure into a soft pass.

## One fixed candidate and policy

Use the unchanged actual final1536 FP16 export: all224 adapter tensors and the
same507-tensor readapted base. Verify the saved final checkpoint masters rounded
to the actual export. No training, threshold search, scale search, extra
updates or checkpoint selection.

The sole change is the inference gate **1[router_logit > 0]**, with zero closed.
Retain the frozen native post-D placement, router arithmetic, head mixing,
FP16 weights, normalization and cached generation. For a closed position,
return its original normalization input exactly; avoid adding a computed zero
or reconstructing that value. Apply this same hard0 policy to every token in
MK prefill, cached decode and prose PPL. No task flags, prompt-specific bypass
or disabling the adapter during PPL. Bind the new inference code separately
from the unchanged soft export and training provenance.

## Fixed numerical profile and prerequisite

The installed Mamba deterministic utility can change both cached MK and
cache-free PPL arithmetic. This is a new, explicitly recorded inference
profile; it is not proof of the cause of the historical drift.

Before importing any project/Mamba module or creating a CUDA context:

- Set `MAMBA_DETERMINISTIC=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Unset `TRITON_CACHE_AUTOTUNING` and
  `TRITON_AUTOTUNE_BLOCK_SIZE_{M,N,K,DSTATE}`. Do not install an internal
  deterministic override. The pinned installed helper then restricts each
  affected autotuner to its fixed single configuration at import time.
- After importing Torch, enable `torch.use_deterministic_algorithms(True)`,
  disable cuDNN benchmarking, enable cuDNN deterministic mode, disable both
  TF32 flags, and select highest float32 matmul precision before model imports.
  Retain FP16 computation and disabled autocast. Unsupported operations fail;
  do not silently fall back or change this profile after observing results.

Record actual flags, installed library/source hashes and selected kernel
configurations. Deterministic settings alone are not a reproducibility proof.

Run a **baseline-only bounded prerequisite**, without the adapter: two fresh
processes, each with two fresh-cache repeats, in the following fixed order.
The first seven prompts are the previously observed differences:

1. `validation-n16-t0-s6`
2. `validation-n16-t0-s13-removed`
3. `validation-n16-t2-s45`
4. `validation-n64-t0-s46-removed`
5. `validation-n64-t1-s12-removed`
6. `validation-n64-t1-s18`
7. `validation-n64-t2-s23`

Append the six unchanged normal controls `validation-n{16,64}-t{0,1,2}-s0`
in numeric size/template order. These observed DEV examples are diagnostics,
not fresh confirmation. Require exact generated IDs, last-prefill hidden and
prefill cache hashes, and per-generated-step hidden/logit hashes within and
across processes. Record top-two logits/ties on any mismatch; stop rather than
selecting a successful repeat. Use passive observation of the native path.

In those same two processes, score the **first four fixed DEV prose windows**
twice using the unchanged64-token full-vocabulary scorer. Require exact per-window hidden
hashes, every CE chunk, target counts, NLL and PPL within and across processes.
Report differences from historical PPL separately. Failure of any prerequisite
blocks hard0 quality evaluation; it does not authorize a looser tolerance.
This13-prompt/four-window check is a bounded reproducibility diagnostic, not
the complete quality gate. The regular stages below still score all64 DEV
and, conditionally, all130 full-validation prose windows.

## Staged paired quality evaluation

Every stage uses **current → hard0 → restored-current** in one process with
the fixed profile. Audit all507 base and224 adapter tensors, base parameter
identity/storage/version, actual files and hook removal before/after. Use
fresh FP16 caches for every MK prompt and the existing12 within-arm replays.
Require complete restored-current MK outputs and PPL rows to equal the initial
current arm exactly. Finite checks and full token/tail accounting remain.
Historical legacy MK is reported descriptively, without requiring equality
to a different numerical profile.

| Stage | Inputs and mandatory gate |
| --- | --- |
| DEV | Existing384 normal +384 removed prompts;64 prose windows /131,008 targets. Normal correct count must increase, removed-target matches must not increase, and hard0 PPL must be **≤ paired current PPL, with no epsilon**. |
| Full PPL | Load only after DEV passes. All130 validation windows /264,764 targets, including final tail. Hard0 PPL must be **≤ paired current PPL, with no epsilon**. |
| CONFIRM | Load its independent384 normal +384 removed prompts only after full PPL passes. Positive normal gain; exact two-sided McNemar p<0.05; positive conservative95% lower bound on paired accuracy improvement; no increase in removed-target matches. |

The confirmation bound remains the one-sided97.5% Clopper–Pearson lower bound
on gained/384 minus the one-sided97.5% upper bound on lost/384, giving at least95%
coverage by Bonferroni. Bind each passed prior-stage report to the exact same
adapter, base, hard0 implementation, numerical profile and protocol. Do not
open later-stage raw/token files early. A failure stops this fixed recipe.

Measure both base and candidate under the new profile: do not substitute the
legacy base PPL into a new-profile comparison. Report the legacy DEV reference
7.6242504268035765 and full reference7.622396587826496 separately, with measured
base drift. Preserve any old-profile soft paired PPL regression as a genuine
soft quality failure. A hard0 success would establish only this distinct
inference policy/profile; no claim of old-profile no-regression follows.

## Exact-identity claims and limits

No PPL increase is the quality gate; bitwise identity is a stronger, separately
audited statement. To claim that the adapter was exactly inactive on tested
prose, require **zero open gates across all56 layers and every scored token**,
exact-return branch coverage, identical per-window hidden hashes, every CE
chunk/NLL/PPL bit pattern and target plan, plus exact restored controls and
unchanged weights. Rounded summary equality alone is insufficient. If gates
open, report their counts even when PPL passes; do not claim an inactive path.
MK may open gates, so prose inactivity never means the entire model is globally
identical to the base. Report any generation cache/hidden identity claim only
for traces actually compared.

Record unchanged adapter payload2,308,208B plus actual serialized file and
metadata bytes. This is expanded-FP16 reference evaluation, with no new compact
runtime, speed, energy or ASIC claim. No publication, automatic promotion,
new training or additional confirmation attempt is authorized by this draft.
