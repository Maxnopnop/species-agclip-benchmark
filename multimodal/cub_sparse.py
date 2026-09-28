"""Test confidence selection separately from the symmetric contrastive loss."""
import torch
from .cub_tokens import TokenFusion


class SparseTokens(TokenFusion):
    def __init__(self,bank,probe_state,mean_probability,variant,config):
        super().__init__(bank,probe_state,mean_probability,variant,config)
        self.top_k=config['attributes_per_view']

    def make_tokens(self,views,valid,intervention=None):
        if self.variant.startswith('dense_'):return super().make_tokens(views,valid,intervention)
        b,v,_=views.shape;a=len(self.attribute_text)
        probability=self.probe(views).sigmoid()
        if self.variant=='constant' or intervention=='constant':probability=self.mean_probability[None].expand(b,-1,-1)
        selected=torch.zeros_like(probability,dtype=torch.bool).scatter_(-1,probability.topk(self.top_k,dim=-1).indices,True)
        text=self.attribute_text
        if self.variant=='permuted' or intervention=='permuted':text=text[self.permutation]
        text=text[None,None].expand(b,v,-1,-1);confidence=2*probability[...,None]-1
        if self.variant=='pooled':
            weights=probability*selected;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            text=(text*weights[...,None]).sum(2,keepdim=True).expand(-1,-1,a,-1)
            confidence=(confidence*selected[...,None]).sum(2,keepdim=True).div(self.top_k).expand(-1,-1,a,-1)
        if self.variant=='region_only' or intervention=='zero_text':text=torch.zeros_like(text);confidence=torch.zeros_like(confidence)
        visual=views[:,:,None].expand(-1,-1,a,-1)
        tokens=self.token_mlp(torch.cat([visual,text,confidence],-1)).flatten(1,2)
        ok=torch.cat([torch.ones(b,1,dtype=torch.bool,device=valid.device),valid],1)
        padding=~(ok[:,:,None]&selected).flatten(1,2)
        return tokens,padding,probability
