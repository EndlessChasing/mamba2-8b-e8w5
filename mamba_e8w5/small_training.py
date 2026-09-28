"""All existing small-tensor compensation, building on immutable norm-v1 math."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .norm_overlay import other_shapes
from .norm_training import NormMasters, HYPERPARAMETERS as NORM_HYPERPARAMETERS

HYPERPARAMETERS = copy.deepcopy(NORM_HYPERPARAMETERS)
HYPERPARAMETERS.update({
    'windows_per_epoch':256, 'successful_updates':1024, 'maximum_attempts':1032,
    'sampler':'four sequential torch.randperm(256) draws from one seeded CPU Generator',
    'selection':'final1024 successful updates only; no validation during training',
    'selected_tensors':393, 'selected_parameters':3580928,
    'initialization':'norm_compensation_v1 exported FP16 small tensors; fresh optimizer and scaler',
    'checkpoint_interval_successes':32,
    'training_data':'256 disjoint train windows excluding all32 original calibration windows',
    'stability':'native FP16 casts and native A/dt formulas; finite checks; no new clamps or reparameterization',
})


def selected_inventory():
    return {name:{'shape':shape,'numel':math.prod(shape)} for name,shape in sorted(other_shapes().items())}


def training_schedule():
    generator=torch.Generator(device='cpu').manual_seed(HYPERPARAMETERS['seed'])
    return [index for _ in range(4) for index in torch.randperm(256,generator=generator).tolist()]


def eligible_window_starts(token_count, excluded_starts, window_tokens=2048, count=256):
    """Fixed token-grid selection, independent of token contents or quality."""
    excluded=[(int(start),int(start)+window_tokens) for start in excluded_starts]
    grid=list(range(0,token_count-window_tokens+1,window_tokens))
    eligible=[start for start in grid if all(start+window_tokens<=left or start>=right for left,right in excluded)]
    if len(eligible)<count:
        raise ValueError('Insufficient internally disjoint, calibration-disjoint training blocks')
    starts=[eligible[index*(len(eligible)-1)//(count-1)] for index in range(count)]
    if len(set(starts))!=count or any(b-a<window_tokens for a,b in zip(starts,starts[1:])):
        raise ValueError('Selected training windows overlap')
    overlap=sum(max(0,min(start+window_tokens,right)-max(start,left))
                for start in starts for left,right in excluded)
    if overlap: raise ValueError('Selected windows overlap old calibration tokens')
    return starts,{'grid_blocks':len(grid),'excluded_grid_blocks':len(grid)-len(eligible),
                  'eligible_blocks':len(eligible),'selected_blocks':len(starts),
                  'internal_overlap_tokens':0,'original_calibration_overlap_tokens':overlap,
                  'minimum_start_gap':min(b-a for a,b in zip(starts,starts[1:]))}


class SmallMasters(NormMasters):
    """FP32 leaves for all393 small tensors, including native SSM/conv parameters."""
    def __init__(self,model,inventory=None):
        super().__init__(model,selected_inventory() if inventory is None else inventory)
        self.block_selected=[]
        mapped=set()
        for index,layer in enumerate(model.backbone.layers):
            prefix=f'backbone.layers.{index}.'
            local={name[len(prefix):]:name for name in self.masters if name.startswith(prefix)}
            if not set(local).issubset(self.block_frozen[index]):
                raise ValueError(f'Selected block parameter missing: {index}')
            self.block_selected.append(local); mapped.update(local.values())
        mapped.add('backbone.norm_f.weight')
        if mapped!=set(self.masters): raise ValueError('Not every selected small tensor participates in the functional mapping')

    def forward(self,input_ids,use_checkpoint=True):
        hidden=self.model.backbone.embedding(input_ids)
        residual=None
        for index,layer in enumerate(self.model.backbone.layers):
            def run_block(h,r,index=index,layer=layer):
                values=dict(self.block_frozen[index])
                for local,global_name in self.block_selected[index].items():
                    values[local]=self.masters[global_name].to(torch.float16)
                return functional_call(layer,values,(h,r),{'inference_params':None},tie_weights=False,strict=True)
            if use_checkpoint and torch.is_grad_enabled():
                hidden,residual=checkpoint(run_block,hidden,residual,use_reentrant=False,preserve_rng_state=True)
            else: hidden,residual=run_block(hidden,residual)
        residual=hidden+residual if residual is not None else hidden
        final=dict(self.final_frozen)
        final['weight']=self.masters['backbone.norm_f.weight'].to(torch.float16)
        return functional_call(self.model.backbone.norm_f,final,
            (residual.to(dtype=self.model.backbone.norm_f.weight.dtype),),tie_weights=False,strict=True)

    @torch.no_grad()
    def stability_receipt(self):
        """Diagnostics use the exact serialized FP16 values and native FP32 exp.

        Bias softplus is a parameter diagnostic, not the token-dependent delta.
        Actual token-dependent forward states/logits are separately finite-checked.
        """
        a_values=[];dt_values=[]
        for name,master in self.masters.items():
            rounded=master.half()
            if not torch.isfinite(master).all() or not torch.isfinite(rounded).all():
                raise FloatingPointError(f'Nonfinite master or FP16 cast: {name}')
            if name.endswith('.A_log'):
                effective=-torch.exp(rounded.float())
                if not torch.isfinite(effective).all() or not (effective<0).all():
                    raise FloatingPointError(f'Nonfinite or nonpositive native decay magnitude: {name}')
                a_values.append(effective)
            elif name.endswith('.dt_bias'):
                diagnostic=F.softplus(rounded.float())
                if not torch.isfinite(diagnostic).all() or not (diagnostic>0).all():
                    raise FloatingPointError(f'Nonfinite or nonpositive dt bias softplus diagnostic: {name}')
                dt_values.append(diagnostic)
        result={'all_masters_and_fp16_casts_finite':True,'native_A_finite':True,'dt_bias_softplus_finite':True}
        for label,values in [('native_A',a_values),('softplus_dt_bias_only',dt_values)]:
            if values:
                joined=torch.cat(values)
                result[label]={'minimum':float(joined.min()),'maximum':float(joined.max()),
                               'zeros':int(torch.count_nonzero(joined==0))}
        return result


def parameter_family(name):
    if name.endswith('.mixer.norm.weight'): return 'gated_norm'
    if name.endswith('.norm.weight') or name=='backbone.norm_f.weight': return 'block_and_final_norm'
    if '.conv1d.' in name: return 'conv_weight' if name.endswith('.weight') else 'conv_bias'
    return name.rsplit('.',1)[1]


@torch.no_grad()
def family_ranges(bank, gradients=False):
    groups={}
    for name,master in bank.masters.items():
        value=master.grad if gradients else master
        if value is None: raise RuntimeError(f'Missing gradient: {name}')
        groups.setdefault(parameter_family(name),[]).append(value.detach().flatten())
    result={}
    for name,values in groups.items():
        joined=torch.cat(values)
        finite=bool(torch.isfinite(joined).all())
        result[name]={'tensors':len(values),'parameters':joined.numel(),'finite':finite,
                      'nonzero':int(torch.count_nonzero(joined)),
                      'minimum':float(joined.min()) if finite else None,
                      'maximum':float(joined.max()) if finite else None}
    return result
