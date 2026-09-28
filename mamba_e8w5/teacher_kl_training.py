"""Pure teacher-KL staged backward, separate from all frozen training helpers.

The native FP16 head GEMMs precede FP32 full-vocabulary KL(teacher || student)
at temperature one. CE is detached logging only. All chunk sums divide by the
total target count, including a short last chunk. Apply loss scaling once to
each chunk's hidden derivative, then backpropagate the assembled derivative
through the retained backbone graph once. No teacher gradient is constructed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def teacher_kl_backward(student_hidden, teacher_hidden, targets, student_head,
                        teacher_head, scaler, chunk_tokens=64):
    if (student_hidden.ndim != 3 or student_hidden.shape != teacher_hidden.shape
            or student_hidden.shape[:2] != targets.shape or targets.dtype != torch.int64
            or student_hidden.dtype != torch.float16 or teacher_hidden.dtype != torch.float16
            or student_hidden.device != teacher_hidden.device or targets.device != student_hidden.device
            or targets.numel() == 0 or type(chunk_tokens) is not int or chunk_tokens <= 0):
        raise ValueError('Invalid native FP16 hidden/target/chunk geometry')
    if not student_hidden.requires_grad:
        raise ValueError('Student hidden states must retain the backbone graph')
    if not torch.isfinite(student_hidden).all() or not torch.isfinite(teacher_hidden).all():
        raise FloatingPointError('Nonfinite teacher/student hidden states')
    count = targets.numel()
    gradient = torch.empty_like(student_hidden)
    ce_parts, kl_parts = [], []
    with torch.autocast(device_type=student_hidden.device.type, enabled=False):
        for first in range(0, student_hidden.shape[1], chunk_tokens):
            end = min(first + chunk_tokens, student_hidden.shape[1])
            chunk = student_hidden[:, first:end].detach().requires_grad_(True)
            with torch.no_grad():
                teacher_logits = teacher_head(teacher_hidden[:, first:end])
                if teacher_logits.dtype != torch.float16 or not torch.isfinite(teacher_logits).all():
                    raise FloatingPointError('Invalid native FP16 teacher logits')
                teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
                del teacher_logits
            student_logits = student_head(chunk)
            if student_logits.dtype != torch.float16 or not torch.isfinite(student_logits).all():
                raise FloatingPointError('Invalid native FP16 student logits')
            student_logits = student_logits.float()
            if student_logits.shape != teacher_logp.shape:
                raise ValueError('Teacher/student full-vocabulary logits differ in shape')
            # The observed next-token labels are used only inside this no-grad scope.
            with torch.no_grad():
                ce = F.cross_entropy(student_logits.detach().reshape(-1, student_logits.shape[-1]),
                    targets[:, first:end].reshape(-1), reduction='sum')
                if not torch.isfinite(ce):
                    raise FloatingPointError('Nonfinite detached CE log')
                ce_parts.append(float(ce))
            student_logp = F.log_softmax(student_logits, dim=-1)
            kl = F.kl_div(student_logp, teacher_logp, reduction='sum', log_target=True)
            loss = kl / count
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pure teacher-KL objective')
            scaled = scaler.scale(loss) if scaler is not None else loss
            grad, = torch.autograd.grad(scaled, chunk)
            gradient[:, first:end].copy_(grad)
            kl_parts.append(float(kl.detach()))
            del chunk, grad, student_logits, student_logp, teacher_logp, ce, kl, loss, scaled
    # Backward nonfinites are reported to the scaler guard, not hidden or clipped.
    student_hidden.backward(gradient)
    mean_kl = math.fsum(kl_parts) / count
    return {'ce': math.fsum(ce_parts) / count, 'ce_logging_only': True,
            'teacher_to_student_kl': mean_kl, 'loss': mean_kl, 'targets': count,
            'kl_coefficient': 1.0, 'ce_gradient_coefficient': 0.0, 'temperature': 1.0,
            'scaled_hidden_gradient_finite': bool(torch.isfinite(gradient).all()),
            'scaled_hidden_gradient_nonzero': int(torch.count_nonzero(gradient)),
            'scaled_hidden_gradient_elements': gradient.numel()}
