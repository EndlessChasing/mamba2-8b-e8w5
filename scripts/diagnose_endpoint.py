#!/usr/bin/env python3
"""Read-only probe for a failed strict full-Q endpoint reproducibility gate."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_e8w5.runtime import load_quantized_model, load_source_model, SentencePieceTokenizer, sha256_file
from mamba_e8w5.calibration import load_wikitext_tokens
from mamba_e8w5.evaluation import evaluate_ppl, ppl_windows
from audit_decoded import tensor_sha_fp16

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
raw = ROOT/'artifacts/e8w5_v1'
manifest = json.loads((raw/'manifest.json').read_text())
q = load_quantized_model(raw)
report = {'complete': False, 'script_sha256': sha256_file(__file__), 'sample_weight_checks': {}}
for label in ('layer0.in_proj', 'layer0.out_proj', 'layer55.in_proj', 'layer55.out_proj'):
    entry = manifest['matrices'][label]
    p = q.get_parameter(entry['source_key'])
    actual = tensor_sha_fp16(p)
    report['sample_weight_checks'][label] = {'actual': actual, 'expected': entry['decoded_fp16_sha256'],
                                            'equal': actual == entry['decoded_fp16_sha256'], 'stride': list(p.stride())}
    print(label, report['sample_weight_checks'][label], flush=True)
tokenizer = SentencePieceTokenizer(ROOT/'models/source')
ids, report['dataset'] = load_wikitext_tokens(tokenizer, 'validation', 'b08601e04326c79dfdd32d625aee71d232d685c3')
windows = ppl_windows(ids, 1024, 4)
report['direct_quantized_first'] = evaluate_ppl(q, windows)
report['direct_quantized_second'] = evaluate_ppl(q, windows)
s = load_source_model(ROOT/'models/source')
report['buffer_names'] = {'source': list(dict(s.named_buffers())), 'quantized': list(dict(q.named_buffers()))}
report['parameter_stride_differences'] = {name: {'source': list(p.stride()), 'quantized': list(q.get_parameter(name).stride())}
    for name, p in s.named_parameters() if p.stride() != q.get_parameter(name).stride()}
diffs = {}
qmodules = dict(q.named_modules())
for name, sm in s.named_modules():
    qm = qmodules[name]
    for field, value in vars(sm).items():
        if isinstance(value, (float, int, bool, str, type(None))):
            if value != getattr(qm, field, '<missing>'):
                diffs[name+'.'+field] = [str(value), str(getattr(qm, field, '<missing>'))]
report['module_scalar_differences'] = diffs
for name, p in q.named_parameters():
    parent, leaf = name.rsplit('.', 1)
    setattr(s.get_submodule(parent), leaf, p)
with torch.inference_mode():
    tokens = windows[0][1][:-1].cuda()[None]
    a = q.backbone(tokens)
    b = s.backbone(tokens)
    report['hidden_parity'] = {'bitwise_equal': torch.equal(a,b), 'max_absolute_error': float((a-b).abs().max())}
    report['hidden_sha256'] = {'direct': tensor_sha_fp16(a), 'swapped': tensor_sha_fp16(b)}
report['swapped_all_parameters'] = evaluate_ppl(s, windows)
report['complete'] = True
(ROOT/'reports/ppl_endpoint_probe.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('dataset', 'sample_weight_checks')}, indent=2), flush=True)
