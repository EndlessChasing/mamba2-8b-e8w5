# Mamba-2 8B E8/W5 — experimental quantized release

This is an independent modified version of NVIDIA's pure Mamba-2 8B base model.
Projection weights use E8P12 with calibrated LDLQ and rotations; the independent
input embedding and output head use group-128 W5. Smaller tensors are FP16.
Huffman coding is lossless relative to these quantized files. Quantization is
lossy relative to the original floating-point checkpoint.

## Quality and scope

Supplied measurement reports: [reports/quality-01-small_compensation_v1_eval.json](reports/quality-01-small_compensation_v1_eval.json)

This all-small candidate contains 393 trained
small FP16 tensors. Its 112 E8 projection files and
2 W5 vocabulary files are unchanged from the original E8/W5 parent.

Selected report arm: **`small_e8w5`**. Full-validation PPL is
**8.359867548** over **264,764 targets** in
**130 windows**. Same-process source FP16 PPL is
**7.334175947**; the candidate's relative change is **+13.9851%**.
The source+5% target (PPL <= 7.700884745) is **unmet**.
These are the supplied prior validation measurements; packaging does not rerun PPL.
Validation has informed development, so these are not untouched test results.
No MK or test score is claimed for this trained all-small candidate.
Original-candidate MK/test results describe the earlier model.


The packaging tool does not assign a quality pass, claim equal-quality accuracy,
or certify a globally smallest model. Inspect the supplied protocol, exact
measurements, source audit, and all file counts. No quality value is inferred
from codec integrity tests. Raw manifest parameter count:
`8236999680`.

## Complete storage accounting

`release_manifest.json` records the real container bytes, every required part,
tokenizer, configuration, code, license, provenance, and quality-report byte.
It also includes its own metadata length in total release bytes. Both codebooks
and all weight scales/signs/offsets are inside the container. No external weight
base or adapter is required. The source checkpoint is needed to reproduce
quantization, not to restore or load this release.

## Loading

Extract `source.zip` into a software checkout. Follow `docs/PACKAGING.md` there.
Run the `restore` subcommand to verify all assets and reconstruct exact raw files.
The public reference runtime decodes weights into FP16; its memory use is
different from the compressed archive size. An online Huffman GPU inference
kernel is not part of this release.

## Attribution and licenses

NVIDIA model/tokenizer provenance and Apache-2.0 declaration are retained in
`licenses/NVIDIA_MAMBA2_MODEL_CARD.md` and `licenses/APACHE-2.0.txt`.
The E8/LDLQ implementation is derived from GPL-3.0 QuIP#, whose license and
attribution are included. `source.zip` contains the matching public source,
vendored primitive pin, and a SHA ledger. See `licenses/THIRD_PARTY.md` for the
separate model and software scopes. There is no NVIDIA or QuIP# endorsement.
