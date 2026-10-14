"""Engineering candidate only: non-reentrant activation recomputation with BN buffers isolated."""
from contextlib import contextmanager
import functools
import torch
from torch.utils.checkpoint import checkpoint

class BNContexts:
    """Each checkpoint call owns a pre-forward BN snapshot, not shared module state."""
    def __init__(self,module):
        self.module=module;self.before=[];self.forward_entered=False
    @contextmanager
    def forward(self):
        if self.forward_entered: raise RuntimeError('Context is per checkpoint call')
        self.forward_entered=True
        for m in self.module.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
                self.before.append((m,m.training,m.momentum,{k:v.detach().clone() for k,v in m._buffers.items() if k in ('running_mean','running_var','num_batches_tracked') and v is not None}))
        yield
    @contextmanager
    def recompute(self):
        if not self.forward_entered:raise RuntimeError('Missing original BN snapshot')
        restore=[]
        try:
            for m,training,momentum,values in self.before:
                if m.training!=training or m.momentum!=momentum:raise RuntimeError('BN training policy changed between forward and backward')
                restore.append((m,{k:m._buffers[k] for k in values}))
                # Swap scratch buffers instead of copy_ into originals: preserve object identities and version counters.
                for k,v in values.items():m._buffers[k]=v.detach().clone()
            yield
        finally:
            for m,values in reversed(restore):
                for k,v in values.items():m._buffers[k]=v
    def pair(self):return self.forward(),self.recompute()

def invoke(module,fn,*args,**kwargs):
    if not torch.is_grad_enabled():return fn(*args,**kwargs)
    tensors=[]
    def collect(x):
        if isinstance(x,torch.Tensor):tensors.append(x)
        elif isinstance(x,(list,tuple)):
            for a in x:collect(a)
        elif isinstance(x,dict):
            for a in x.values():collect(a)
    collect(args);collect(kwargs)
    if any(t.device.type not in ('cpu','meta') for t in tensors):
        raise RuntimeError('Candidate is CPU/meta only; no device qualification')
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in tensors):
        raise RuntimeError('FP32 candidate only')
    ctx=BNContexts(module)
    return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=True,context_fn=ctx.pair,**kwargs)

def install(model,names):
    modules=dict(model.named_modules());restore=[]
    for name in names:
        module=modules[name];original=module.forward
        if hasattr(module,'_candidate79_wrapped'):raise RuntimeError('Duplicate checkpoint wrapper')
        def wrapped(*args,_m=module,_fn=original,**kwargs):return invoke(_m,_fn,*args,**kwargs)
        module.forward=wrapped;module._candidate79_wrapped=True;restore.append((module,original))
    return restore

def uninstall(restore):
    for module,original in reversed(restore):
        module.forward=original;del module._candidate79_wrapped

def pgs_policy(model,nested):
    outer=['conformer1','conformer2','conformer3','early_fusion_aolp','early_fusion_dolp','fusion_t']
    if not nested:return outer
    # Outer boundaries suppress persistent branch internals; inner boundaries bound recomputation storage.
    inner=[]
    for prefix in ['conformer1','conformer2','conformer3']:
        inner.extend(prefix+'.'+name for name in ['conv_1','trans_1',*[f'conv_trans_{i}' for i in range(2,13)]])
    return inner+outer
