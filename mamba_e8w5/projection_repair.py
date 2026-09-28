"""Anchored cross-moment projection fitting; separate from the frozen codec.

All statistics are normalized per token, without centering or added bias.
The returned dense matrix is a fitting target, not a compressed artifact.
"""
from __future__ import annotations

import torch


def fresh_train_starts(token_count, excluded_starts, *, seqlen=2048, count=48):
    """Select spread grid windows disjoint from every excluded token interval."""
    if token_count <= 0 or seqlen <= 0 or count != 48:
        raise ValueError('The pilot requires exactly48 positive-length windows')
    excluded = sorted(set(int(x) for x in excluded_starts))
    if any(x < 0 or x + seqlen > token_count for x in excluded):
        raise ValueError('Invalid excluded interval')
    grid = list(range(0, token_count-seqlen+1, seqlen))
    eligible = [start for start in grid
                if not any(start < old+seqlen and old < start+seqlen for old in excluded)]
    if len(eligible) < count:
        raise ValueError('Insufficient unused training intervals')
    selected = [eligible[i*(len(eligible)-1)//(count-1)] for i in range(count)]
    if len(set(selected)) != count:
        raise ValueError('Repeated selected interval')
    fit = [value for i, value in enumerate(selected) if i % 3 != 2]
    heldout = [value for i, value in enumerate(selected) if i % 3 == 2]
    return fit, heldout, {'grid_windows': len(grid), 'excluded_grid_windows': len(grid)-len(eligible),
        'eligible_windows': len(eligible), 'selected_windows': count,
        'fit_windows': len(fit), 'heldout_windows': len(heldout),
        'selection': '48 spread ranks floor(i*(eligible_count-1)/47); every third is heldout',
        'seqlen': seqlen, 'overlap_with_excluded_tokens': 0}


@torch.no_grad()
def anchored_ridge_target(h, k, anchor, *, damping=.01, row_chunk=512):
    """Solve W*(H+lambda I)=K+lambda*anchor with fixed lambda=.01*mean(diag H).

    Use the *raw undamped* H with the frozen E8 codec afterwards. Passing
    the regularized matrix would apply the ridge penalty a second time.
    """
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or k.ndim != 2
            or anchor.shape != k.shape or k.shape[1] != h.shape[0]
            or h.device != k.device or k.device != anchor.device
            or h.dtype != torch.float32 or k.dtype != torch.float32
            or damping != .01 or row_chunk <= 0):
        raise ValueError('Invalid projection statistics or fixed ridge protocol')
    if not all(bool(torch.isfinite(x).all()) for x in (h, k, anchor)):
        raise ValueError('Nonfinite fitting inputs')
    scale = h.diag().mean()
    if not bool(scale > 0):
        raise ValueError('Nonpositive mean input energy')
    symmetric = (h + h.T) * .5
    normalized = symmetric / scale
    normalized.diagonal().add_(damping)
    cholesky = torch.linalg.cholesky(normalized)
    target = torch.empty_like(k)
    residual_sq = torch.zeros((), device=h.device, dtype=torch.float64)
    rhs_sq = torch.zeros_like(residual_sq)
    for first in range(0, len(k), row_chunk):
        rows = slice(first, first+row_chunk)
        rhs = k[rows]/scale + damping*anchor[rows].float()
        solution = torch.cholesky_solve(rhs.T.contiguous(), cholesky).T
        target[rows] = solution
        residual_sq += (solution @ normalized - rhs).square().sum(dtype=torch.float64)
        rhs_sq += rhs.square().sum(dtype=torch.float64)
    relative_residual = float((residual_sq/rhs_sq.clamp_min(1e-30)).sqrt())
    if not bool(torch.isfinite(target).all()) or relative_residual > 1e-4:
        raise RuntimeError('Anchored solve failed its fixed residual bound')
    return target, {'damping': damping, 'mean_input_energy': float(scale),
        'lambda': float(scale)*damping, 'relative_normal_equation_residual': relative_residual,
        'anchor': 'current decoded FP16 E8 projection', 'centering': False,
        'solver': 'FP32 Cholesky; no inverse; no adaptive damping',
        'subsequent_codec_hessian': 'raw undamped H'}
