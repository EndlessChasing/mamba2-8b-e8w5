# User-directed PPL priority continuation

The user explicitly changed the research priority after seeing the norm-only
tradeoff: focus on fixing PPL; revisit recall with Resurface later. This permits
new PPL measurements and training despite the earlier MK advancement failure.
It does not retrospectively pass the original gate or authorize publication.

## Preserved history and current objective

- The original norm protocol and failed combined-gate report remain immutable.
- Its candidate is still the fixed final 128-update FP16 export; no checkpoint
  reselection or retraining changes that measured result.
- Evaluate that export on the complete validation split under a separate receipt,
  with the original FP16 source and original E8/W5 parent in the same process.
  Keep all paired token identities and per-window NLL values.
- Primary research target: recover PPL at the existing E8/W5 raw capacity. Report
  changes against both the original FP16 source and the preceding candidate.
  The existing source-relative +5% PPL target remains a useful numerical target.
- MK is recorded evidence, no longer a blocker for these PPL experiments. The
  previous 10/12 to 9/12 result remains visible. Future Resurface improvement is
  a hypothesis; no recall recovery is assumed, and none has been measured here.
- Do not run the test split during this continuation. Validation has already
  been inspected and is being used for research, not as an untouched test set.
- Publication, public pushes and artifact uploads remain on hold.

## Next bounded training experiment

See [all-small-tensor protocol](SMALL_TENSOR_COMPENSATION_PROTOCOL.md). It expands
training to the existing small FP16 tensors while leaving E8/W5 unchanged.
Only the declared final checkpoint is evaluated. Model architecture stays pure
Mamba-2 and no inference adapter or tensor field is introduced.
