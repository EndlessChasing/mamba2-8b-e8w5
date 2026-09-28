"""FP32 masters for fixed-index E8-derived prototype compensation.

Every checkpoint replay rounds the small tables to FP16 and executes the exact
learned decoder. Only the table masters are leaves requiring gradients; native
decoded projection-weight gradients are transient within the current block.
The existing model's parameters, including its trained small tensors, stay frozen.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .codec import read_e8
from .learned_e8_codebook import decode_e8_with_prototypes, validate_prototypes
from .norm_training import HYPERPARAMETERS as NORM_HYPERPARAMETERS, tensor_hash

HYPERPARAMETERS=copy.deepcopy(NORM_HYPERPARAMETERS)
HYPERPARAMETERS.update({
    'learning_rate':1e-3, 'selected_tensors':112, 'selected_parameters':229376,
    'initialization':'zero prototype corrections; best all-small FP16 model remains frozen',
    'forward_parameter_dtype':'FP16 prototype table and decoded FP16 projection weights',
    'trainable_family':'one256x8 prototype correction per original E8 projection; fixed original indices',
    'checkpoint_interval_successes':32,
    'rounding_backward':'ordinary autograd identity through FP16 casts; surrogate for discrete rounding',
    'training_data':'32 fresh TRAIN windows excluding original32, all-small256 and cross-moment48',
})


def selected_inventory(layers=56):
    return {f'layer{index}.{part}':{'shape':[256,8],'numel':2048,
        'source_key':f'backbone.layers.{index}.mixer.{part}.weight'}
        for index in range(layers) for part in ('in_proj','out_proj')}


def training_schedule():
    generator=torch.Generator(device='cpu').manual_seed(HYPERPARAMETERS['seed'])
    return [index for _ in range(4) for index in torch.randperm(32,generator=generator).tolist()]


def load_projection_payloads(raw_dir,device='cpu',layers=56):
    """Read each original .e8 once; caller must verify the raw manifest first.

    Default CPU storage saves GPU memory. Passing the model device retains
    int32 codes there; the decoder converts only the current projection to long.
    This helper does not establish source/package provenance by itself.
    """
    result={}
    for label in selected_inventory(layers):
        payload=read_e8(Path(raw_dir)/f'{label}.e8')
        result[label]={key:value.to(device) if isinstance(value,torch.Tensor) else value
                       for key,value in payload.items()}
    return result


def _identity(value):
    return (id(value),value.data_ptr(),value._version,tuple(value.shape),
            tuple(value.stride()),value.dtype,str(value.device))


class PrototypeMasters:
    def __init__(self,model,payloads,cb,*,payload_device=None):
        self.model=model.eval().requires_grad_(False)
        self.cb=cb
        self.inventory=selected_inventory(len(model.backbone.layers))
        if set(payloads)!=set(self.inventory):
            raise ValueError('E8 payloads must exactly cover both projections in every native block')
        if model.backbone.fused_add_norm or model.backbone.residual_in_fp32:
            raise ValueError('Prototype compensation requires the native unfused FP16 residual path')
        parameters=dict(model.named_parameters())
        self.payloads={};self.masters={}
        for label,entry in self.inventory.items():
            original=parameters[entry['source_key']]
            payload=payloads[label]
            indices=payload['indices']
            if original.dtype!=torch.float16 or tuple(original.shape)!=(indices.shape[0],indices.shape[1]*8):
                raise ValueError(f'E8 payload/native projection geometry differs: {label}')
            self.payloads[label]={key:value.detach().to(payload_device or value.device) if isinstance(value,torch.Tensor) else value
                                  for key,value in payload.items()}
            self.masters[label]=torch.nn.Parameter(torch.zeros(256,8,device=original.device,dtype=torch.float32))
        self.frozen_identities={name:_identity(value) for name,value in parameters.items()}
        self.payload_identities=self._payload_identities()
        self.codebook_identities=self._codebook_identities()
        self.block_frozen=[dict(layer.named_parameters()) for layer in model.backbone.layers]
        for index,layer in enumerate(model.backbone.layers):
            if layer.fused_add_norm or layer.residual_in_fp32 or layer.mlp is not None or layer.mixer.use_mem_eff_path:
                raise ValueError('Unsupported native block path')
            for part in ('in_proj','out_proj'):
                module=getattr(layer.mixer,part)
                if not isinstance(module,torch.nn.Linear) or module.bias is not None:
                    raise ValueError(f'Expected bias-free native Linear in block{index}')

    def _payload_identities(self):
        return {label:{key:_identity(value) if isinstance(value,torch.Tensor) else value
                       for key,value in payload.items()} for label,payload in self.payloads.items()}

    def _codebook_identities(self):
        return {name:_identity(getattr(self.cb,name)) for name in ('grid','grid_packed_abs')}

    def parameters(self):
        return list(self.masters.values())

    def state_dict(self):
        return {name:value.detach().cpu().clone() for name,value in self.masters.items()}

    def load_state_dict(self,state):
        if set(state)!=set(self.masters):raise ValueError('Checkpoint prototype keys differ')
        for name,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or
                    tuple(value.shape)!=(256,8) or not torch.isfinite(value).all() or not torch.isfinite(value.half()).all()):
                raise ValueError(f'Invalid FP32 prototype master: {name}')
        with torch.no_grad():
            for name,value in state.items():self.masters[name].copy_(value)

    def export_tables(self):
        result={}
        for label,master in self.masters.items():
            if not torch.isfinite(master).all():raise FloatingPointError(f'Nonfinite master: {label}')
            table=master.detach().cpu().half()
            validate_prototypes(table)
            result[label]=table
        return result

    def assert_frozen(self):
        parameters=dict(self.model.named_parameters())
        if set(parameters)!=set(self.frozen_identities):raise RuntimeError('Frozen parameter coverage changed')
        for name,value in parameters.items():
            if _identity(value)!=self.frozen_identities[name] or value.requires_grad or value.grad is not None:
                raise RuntimeError(f'Frozen native parameter changed: {name}')
        if self._payload_identities()!=self.payload_identities:raise RuntimeError('Frozen E8 payload identity/version changed')
        if self._codebook_identities()!=self.codebook_identities:raise RuntimeError('Frozen base codebook identity/version changed')

    def decode(self,label,table=None):
        """The FP16 table cast occurs inside every forward/checkpoint replay."""
        if label not in self.masters:raise KeyError(label)
        correction=self.masters[label].to(torch.float16) if table is None else table
        return decode_e8_with_prototypes(self.payloads[label],self.cb,correction,
                                         device=self.masters[label].device)

    def forward_block(self,index,hidden,residual,use_checkpoint=True):
        """Expose the same native block path for bounded replay-gradient smoke."""
        layer=self.model.backbone.layers[index]
        def run_block(h,r):
            values=dict(self.block_frozen[index])
            for part in ('in_proj','out_proj'):
                values[f'mixer.{part}.weight']=self.decode(f'layer{index}.{part}')
            return functional_call(layer,values,(h,r),{'inference_params':None},tie_weights=False,strict=True)
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(run_block,hidden,residual,use_reentrant=False,preserve_rng_state=True)
        return run_block(hidden,residual)

    def forward(self,input_ids,use_checkpoint=True):
        """Use checkpoints for real-model training; no-checkpoint is for no_grad/parity.

        No decoded weight is cached on the bank. During checkpointed backward,
        one native block's dense weight gradients are transient nonleaf tensors.
        Disabling checkpoints under gradients on all56 blocks may exceed memory.
        """
        hidden=self.model.backbone.embedding(input_ids)
        residual=None
        for index in range(len(self.model.backbone.layers)):
            hidden,residual=self.forward_block(index,hidden,residual,use_checkpoint)
        residual=hidden+residual if residual is not None else hidden
        return self.model.backbone.norm_f(residual.to(dtype=self.model.backbone.norm_f.weight.dtype))


def gradient_receipt(bank):
    per_table={};finite=True;nonzero=0
    for label,master in bank.masters.items():
        if master.grad is None:raise RuntimeError(f'Missing prototype gradient: {label}')
        grad=master.grad
        if grad.dtype!=torch.float32:raise RuntimeError(f'Prototype gradient is not FP32: {label}')
        valid=bool(torch.isfinite(grad).all());count=int(torch.count_nonzero(grad))
        per_table[label]={'finite':valid,'nonzero':count,'numel':grad.numel()}
        finite &= valid;nonzero+=count
    bank.assert_frozen()
    return {'finite':finite,'tensors':len(per_table),'parameters':sum(p.numel() for p in bank.parameters()),
            'nonzero':nonzero,'dtypes':['torch.float32'],'per_tensor':per_table}


@torch.no_grad()
def export_parity(bank,tables,ids):
    """Compare live rounded masters to native model with read-back table decode.

    Caller supplies tables loaded from serialized files. All previous projection
    parameter references are restored even when a decode or comparison fails.
    Temporary native installation uses additional full projection-weight memory.
    """
    if set(tables)!=set(bank.masters):raise ValueError('Export table coverage differs')
    for table in tables.values():validate_prototypes(table)
    expected=bank.forward(ids,use_checkpoint=False)
    expected_logits=bank.model.lm_head(expected[:,-8:])
    if not torch.isfinite(expected).all() or not torch.isfinite(expected_logits).all():
        raise FloatingPointError('Nonfinite functional export reference')
    previous={}
    try:
        for label,entry in bank.inventory.items():
            owner,leaf=entry['source_key'].rsplit('.',1)
            module=bank.model.get_submodule(owner)
            previous[entry['source_key']]=getattr(module,leaf)
            setattr(module,leaf,torch.nn.Parameter(bank.decode(label,tables[label]),requires_grad=False))
        actual=bank.model.backbone(ids)
        actual_logits=bank.model.lm_head(actual[:,-8:])
        if not torch.isfinite(actual).all() or not torch.isfinite(actual_logits).all():
            raise FloatingPointError('Nonfinite native export output')
        result={'bitwise_equal':bool(torch.equal(expected,actual) and torch.equal(expected_logits,actual_logits)),
                'hidden_bitwise_equal':bool(torch.equal(expected,actual)),
                'last8_logits_bitwise_equal':bool(torch.equal(expected_logits,actual_logits)),
                'maximum_absolute_difference':float((expected-actual).abs().max()),
                'functional_hidden_sha256':tensor_hash(expected),'reloaded_hidden_sha256':tensor_hash(actual),
                'functional_last8_logits_sha256':tensor_hash(expected_logits),
                'reloaded_last8_logits_sha256':tensor_hash(actual_logits),
                'input_tokens':ids.numel(),'tables':len(tables)}
        if not result['bitwise_equal']:raise RuntimeError(f'Prototype export/native forward mismatch: {result}')
    finally:
        for name,value in previous.items():
            owner,leaf=name.rsplit('.',1)
            setattr(bank.model.get_submodule(owner),leaf,value)
    bank.assert_frozen()
    return result
