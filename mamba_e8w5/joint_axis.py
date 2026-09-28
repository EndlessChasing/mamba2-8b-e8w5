"""Same-format joint E8/axis candidate search; frozen codec remains authoritative."""
import math
import torch
from . import codec

ROW_BATCH = 4096


class JointAxisResidualCodebook:
    codesz = 8
    idx_dtype = torch.int64

    def __init__(self, base, amplitude):
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise ValueError('A finite positive stored amplitude is required')
        self.base, self.amplitude = base, float(amplitude)
        self.greedy = codec.AxisResidualCodebook(base, self.amplitude)
        self.calls = self.rows = 0
        self.totals = None

    def quantize(self, x, return_idx=True, **kwargs):
        if (x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 8
                or not len(x) or not torch.isfinite(x).all()):
            raise ValueError('Expected finite nonempty FP32 [rows,8] queries')
        # The incumbent MUST see the entire original query, not each row batch.
        original, incumbent = self.greedy.quantize(x, return_idx=True, **kwargs)
        values = codec.decode_codes(incumbent, self.base, self.amplitude)
        if not torch.equal(values, original) or not torch.isfinite(values).all():
            raise ValueError('Greedy code/decode mismatch')
        codes = incumbent.clone()
        offsets = torch.zeros(16, 8, dtype=torch.float32, device=x.device)
        nibble = torch.arange(16, device=x.device, dtype=torch.int64)
        offsets[nibble, nibble // 2] = (1 - 2*(nibble % 2)).float()*self.amplitude
        totals = torch.zeros(3, dtype=torch.float64, device=x.device)
        for start in range(0, len(x), ROW_BATCH):
            query = x[start:start+ROW_BATCH]
            shifted = (query[:, None, :] - offsets[None, :, :]).reshape(-1, 8)
            _, base_idx = self.base.quantize(shifted, return_idx=True, **kwargs)
            if int(base_idx.min()) < 0 or int(base_idx.max()) >= 65536:
                raise ValueError('Base search returned a non-uint16 code')
            candidates = (base_idx.long().reshape(-1, 16) << 4) | nibble[None, :]
            decoded = codec.decode_codes(candidates, self.base, self.amplitude)
            errors = (query[:, None, :] - decoded).square().sum(-1)
            old_error = (query - values[start:start+len(query)]).square().sum(-1)
            if not torch.isfinite(decoded).all() or not torch.isfinite(errors).all() or not torch.isfinite(old_error).all():
                raise ValueError('Nonfinite candidate/error')
            # min returns the first minimum nibble; strict < keeps greedy ties.
            best_error, choice = errors.min(-1)
            use = best_error < old_error
            row = torch.arange(len(query), device=x.device)
            next_codes = torch.where(use, candidates[row, choice], codes[start:start+len(query)])
            next_values = codec.decode_codes(next_codes, self.base, self.amplitude)
            final_error = (query - next_values).square().sum(-1)
            if not torch.all(final_error <= old_error):
                raise ValueError('Local FP32 incumbent error increased')
            codes[start:start+len(query)] = next_codes
            values[start:start+len(query)] = next_values
            totals += torch.stack((old_error.double().sum(), final_error.double().sum(), use.double().sum()))
        self.calls += 1
        self.rows += len(x)
        self.totals = totals if self.totals is None else self.totals + totals
        return (values, codes) if return_idx else values

    def receipt(self):
        old, new, changed = self.totals.cpu().tolist() if self.totals is not None else (0., 0., 0.)
        return {'calls': self.calls, 'query_rows': self.rows, 'strictly_improved_rows': int(changed),
                'greedy_local_squared_error_sum': old, 'joint_local_squared_error_sum': new,
                'local_nonregression_checked': True, 'row_batch': ROW_BATCH,
                'shift_order': list(range(16)), 'incumbent_scope': 'full original query',
                'tie_rule': 'greedy incumbent first, then first nibble; strict-less only'}


def joint_ldlq_adapter(frozen_ldlq, receipts):
    """Explicit injection seam into frozen vector_quantize; no monkeypatch."""
    def apply(wr, hr, l, d, quantizer, args, buf_cols=128):
        if (type(quantizer) is not codec.AxisResidualCodebook or buf_cols != 128
                or args.quip_tune_iters != 2 or args.resid_scale_override != -1):
            raise ValueError('Unexpected frozen LDLQ invocation')
        joint = JointAxisResidualCodebook(quantizer.base, quantizer.amplitude)
        result = frozen_ldlq(wr, hr, l, d, joint, args, buf_cols=buf_cols)
        receipts.append(joint.receipt())
        return result
    return apply
