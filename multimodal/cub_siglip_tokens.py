"""Dimension-general constructor; inherit the identical token/forward logic."""
import torch
from torch import nn
from .cub_tokens import TokenFusion


class SiglipTokenFusion(TokenFusion):
    def __init__(self,bank,probe_state,mean_probability,variant,config):
        nn.Module.__init__(self);self.variant=variant;self.max_residual=config['max_residual'];d=bank['class_text'].shape[1]
        self.register_buffer('class_text',bank['class_text'].clone());self.register_buffer('attribute_text',bank['attribute_text'].clone());self.register_buffer('mean_probability',mean_probability.clone())
        self.register_buffer('permutation',torch.randperm(len(self.attribute_text),generator=torch.Generator().manual_seed(1729)))
        self.probe=nn.Linear(d,len(self.attribute_text));self.probe.load_state_dict(probe_state);self.probe.requires_grad_(False)
        h=config['hidden_dim'];self.token_mlp=nn.Sequential(nn.Linear(2*d+1,h),nn.GELU(),nn.Linear(h,h),nn.LayerNorm(h));self.query=nn.Linear(d,h)
        self.attention=nn.MultiheadAttention(h,4,dropout=0,batch_first=True);self.output=nn.Linear(h,d);self.gate=nn.Parameter(torch.zeros(()))
