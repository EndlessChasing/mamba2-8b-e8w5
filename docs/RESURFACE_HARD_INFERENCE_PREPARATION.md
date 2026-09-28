# Fixed-hard0 inference preparation

Status: **untriggered fallback draft; CPU implementation checks only.** The current
soft adapter, its protocol and exports are unchanged. The reported soft DEV
result improved both PPL and recall, so the proposed fallback condition was not
met. No hard0 GPU, PPL, MK or cached-generation experiment has been run.

## Fixed policy

`mamba_e8w5/resurface_hard_inference.py` reuses an actual unchanged FP16 **soft**
adapter export through the frozen native reader. The explicit inference policy
is `RESURFACE_POST_D_HARD0_GATE1_EXACT_CLOSED_V1`: router logit strictly greater
than zero opens the gate with value one; zero or negative logits close it. No
scale, threshold argument, task flag, fitting or threshold search is provided.

This changes the inference policy, not the adapter tensors. Relative to the
soft gate, an open correction is amplified: its old sigmoid value is between
0.5 and 1. Recall improvement is therefore not guaranteed. A future experiment
would need a separately declared protocol and policy manifest; the
unchanged soft export alone does not describe this behavior.

## Native hook and audit API

```python
from mamba_e8w5.resurface_hard_inference import install_fp16

with install_fp16(model, adapter_path,
                  expected_sha256=adapter_sha,
                  expected_binding=adapter_binding,
                  expected_base_hashes=base507_hashes) as bank:
    # Invoke the existing native backbone/prefill/step APIs normally.
    hidden = model.backbone(input_ids)
    stats = bank.gate_stats()
    bank.assert_adapter_unchanged()
    bank.assert_base_frozen(check_values=True)
```

The bank inherits native hook ownership, geometry guards and base identity/content
auditing. External adapter parameters are FP16, frozen and sealed after file
loading. Its source receipt records both the stored soft mode and effective hard0
policy. Re-export is rejected. Context exit removes every installed hook.
Forwards must be serialized. `forward_hidden(ids, use_checkpoint=False)` returns
native hidden values and scalar statistics; it does not return the training
`GateCapture` object and does not support checkpointing or gradients.

Every fully closed norm call returns its original input `y` object without an
add, multiplication, view or copy. Mixed calls compute head mixing only on open
rows and use `torch.where` for closed rows, preserving their FP16 bits including
negative zero. Mixed output strides need not equal the original strides. Router
logits and open-position corrected values must be finite. A failed invocation
is counted and its native hook frame is removed.

`gate_stats()` retains Python scalar counts and extrema per layer, including
closed/open/zero decisions and failed calls. It retains no token logits or masks
and introduces no recurrent state. `reset_gate_stats()` is allowed between
invocations. `validate_gate_coverage(calls_per_layer=..., tokens_per_layer=...)`
requires exact counts and no failed calls, so an empty run cannot establish that
all gates closed.

## Limits of local identity

A closed gate preserves the readout that reached that norm. An earlier open
adapter can already have changed downstream inputs or recurrent caches; closing
a later gate does not undo those effects. CPU toy step routing is not a native
8B cache-equivalence result.

If a future authorized full-PPL experiment needs to establish exact baseline
behavior on the existing 130-window protocol, it must independently verify all
264,764 input-token decisions per layer (14,826,784 across 56 layers), actual
507 base hashes, same-process hidden and native CE controls, and fresh window
state. Closed counts alone do not prove the score. These finite windows also do
not establish equality for arbitrary text or generation. No such experiment or
claim is part of this preparation.

## CPU evidence

`tests/test_resurface_hard_inference.py` passed all six tests with CUDA hidden and
uninitialized, exit 0, in 0.019344729 seconds. Tests cover:

- Fully closed original object, noncontiguous strides and signed zero.
- Mixed selection, preservation of closed bits, and avoiding overflow from
  mixing a closed row.
- Strictly positive gate value one without rescaling.
- Rejection of NaN and infinite router logits.
- Real FP16 file reuse, hook routing, exact scalar coverage and immutable hashes.
- Failed-invocation cleanup, hook removal and identity mismatch rejection.

The exact command, source hashes and result are retained in
`reports/resurface_hard_inference_cpu_checks.json`. The toy fixture is newly
created for the test; it does not load the current trained adapter.

| File | SHA-256 |
|---|---|
| New implementation | `8793ea76856f40edd9cec43b0a716a6a16b8385ce26d5351514e6c6ca6614891` |
| CPU tests | `db4a7603eabbb917cdf5e1856a725f3a7ceff278447088c4b8ea204b8c4938eb` |
| Frozen native dependency | `2dd08c7ee8958c832f0ae9da7cf5f261dceeff3e528e493c905c7c52df7166d7` |
