# Fixed-index prototype training implementation

Implementation: `mamba_e8w5/prototype_training.py`. This is a new experimental
bank, separate from the frozen norm-only and all-small training helpers. No
quality improvement is established by the bank or its CPU tests.

## What is trainable

There is one FP32 master of shape `[256,8]` for each of the 112 E8 projections:
229,376 scalar masters in total. All initialize to zero. The existing 507 model
parameter objects remain frozen, including all 393 trained small tensors and
both W5 vocabulary matrices. Indices, signs, balance, scale, base codebook and
rotation implementation are unchanged.

Each checkpoint replay casts that block's two tables to FP16, calls
`decode_e8_with_prototypes` and substitutes the resulting FP16 projection
weights through `torch.func.functional_call`. No decoded training weight is
cached on the bank. Non-reentrant checkpointing bounds the saved activations;
the native projection-weight derivatives are temporary nonleaf tensors within
the currently replayed block. Only the small masters retain optimizer gradients.

The prototype decoder changes absolute magnitudes using the original code's
fixed prototype ID, signs and parity. It does not recompute parity or the
quarter offsets after an update. Nonzero corrections are E8-derived and do not
in general remain on the original strict E8 lattice.

This is a small increase in stored capacity, not zero-byte compensation:
112 FP16 payloads contain 458,752 bytes; their 32-byte file headers bring the
table files to 462,336 bytes. Manifests are additional. The original 16-bit E8
indices still encode eight weights per code. No entropy-compressed container
size is assumed.

## API

```python
payloads = load_projection_payloads(raw_dir, device="cuda", layers=56)
bank = PrototypeMasters(best_small_model, payloads, codebook)

hidden = bank.forward(ids, use_checkpoint=True)
hidden, residual = bank.forward_block(index, hidden, residual,
                                     use_checkpoint=True)
masters = bank.state_dict()       # CPU FP32 copies, checkpoint labels below
tables = bank.export_tables()    # CPU FP16, finite, same labels
receipt = gradient_receipt(bank)
bank.assert_frozen()
```

Keys are `layer0.in_proj`, `layer0.out_proj`, ..., `layer55.out_proj`.
`load_projection_payloads` reads each original file once. It does not verify a
package manifest; the driver must do so before constructing the bank. Storing
the original int32 indices on the GPU costs about 3.07 GB; leaving them on CPU
trades that memory for repeated transfers. Long indexing casts are local to a
projection rather than retained for all matrices.

`load_state_dict` validates exact keys, shapes, FP32 masters and finite FP16
casts before modifying any master. Its presence supports checkpoints and CPU
provenance verification; it does not authorize resuming or selecting a training
run outside its declared protocol.

`export_parity(bank, tables, ids)` accepts tables read back from their serialized
files. It compares the live bank with an ordinary native model after installing
all decoded FP16 weights, requiring identical hidden output and final eight
head-logit positions. It restores all original references in `finally`.
Installing all decoded projections temporarily requires about 12.27 GB beyond
the already resident frozen projection weights.

`assert_frozen` verifies complete model coverage, identities, version counters,
requires-grad flags and absence of model gradients. It also guards the payload
and base-codebook identities/versions. The independent driver must still hash
all original tensor contents before/after training; these guards do not claim
to replace a content audit.

## Gradients and finite precision

The executed path is FP32 master → FP16 table → FP32 decoded grid and inverse
rotations → FP16 weight → native FP16 model. Ordinary autograd treats the casts
as identity derivatives, with the corresponding gradient dtype conversions.
This is a surrogate gradient for a function containing discrete rounding, not
the mathematical derivative of the serialized model's piecewise-constant loss.
In particular, an FP16 table-gradient cast can overflow or underflow even
though the optimizer master is FP32. The declared dynamic loss scaling and
finite-gradient guards therefore remain necessary.

Use the frozen `norm_training.chunked_loss_backward`: teacher runs without
gradients; the full-vocabulary KL is summed across vocabulary and averaged over
all target positions; short final chunks receive their actual token weight.
Scaled hidden gradients enter one backbone backward. This implementation does
not introduce a second loss scaling step or any custom decoder backward.

For the real model, training must use checkpoints. A full 56-block forward
with gradients and `use_checkpoint=False` could retain every dense decoder
graph. The GPU smoke instead compares checkpoint/no-checkpoint gradients on
individual blocks 0, 18 and 55 using captured native block inputs. A separate
full 2047-target update must measure peak memory and timing before a run.

Two full FP16 models use about 32.95 GB. Master, gradient and two Adam moments
together require about 3.67 MB. Additional memory is from original indices,
checkpoint boundaries, full-vocabulary chunks, transient decoded weights and
decoder/native backward. The 96 GB device appears sufficient, but this is a
planning estimate until the complete-step smoke is measured. With the current
dense Hadamard implementation, all 112 forward inverse-decodes require roughly
83 TFLOP of FP32 arithmetic; forward, checkpoint replay and decoder adjoints
can total roughly 250 TFLOP per update before model GEMMs. This is an arithmetic
estimate, not measured wall time. Do not replace those transforms with a faster
implementation while claiming bitwise identity without a separate proof.

## CPU checks

Six synthetic tests exercise an actual pinned E8 codebook and a two-layer
native-style FP16 residual model:

- Zero-table/native output identity and gradients reaching all four toy tables.
- Checkpoint/no-checkpoint hidden outputs and every master gradient.
- Isolated-block output, residual, input and both table-gradient parity.
- Serialized table readback, native installation, hidden/head equality and
  restoration of every original parameter reference.
- Master/payload loading, exact 112/229376 inventory, fixed permutation schedule.
- Detection of frozen-parameter mutation and nonfinite FP16 export.

These tests are CPU-only and do not validate Mamba GPU backward, GPU memory,
training stability, heldout PPL or recall.

## Why language loss is the next experiment

The six-projection cross-moment pilot reduced aggregate local teacher-output
error but worsened joint language PPL. Accordingly, local MSE is not a promotion
criterion for this route. CE/KL can weight prototype changes by their effect on
the complete language model while retaining the small representation.

For a possible future local-control solver, let `Q` be the expanded rotated
grid, `s` the fixed scale, `B` the diagonal input balance, `Du/Dv` the fixed
sign diagonals and `Rn/Rm` the forward row-rotation matrices. The pre-half
decoder is `T(Q)=s Dv Rm Q Rn.T Du B^-1`. Its real-linear adjoint is
`T*(G)=s Rm.T Dv G B^-1 Du Rn`. The prototype adjoint scatter-adds each
coordinate's signed gradient to its original `[code>>8, coordinate]` entry.
With linear table-to-weight map `F`, a ridge normal operator is
`F*(F(v)H + lambda F(v)) + eta v`. It has only 2048 unknowns per projection but
each application still expands a full matrix and applies H. Such CG fits the
real-linear proxy; applying FP16 rounding inside that operator would break its
linearity. It is not the recommended next quality experiment after the pilot's
observed mismatch between local MSE and PPL.
