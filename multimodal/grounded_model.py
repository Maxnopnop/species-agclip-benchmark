"""Raw-image, partially trainable two-tower CLIP with attribute-region CAF."""
import copy,math,random
import torch
from torch import nn
from torch.nn import functional as F
from .models import load_clip

def activate_tail(visual,blocks=2):
    visual.requires_grad_(False)
    for block in visual.transformer.resblocks[-blocks:]:block.requires_grad_(True)
    visual.ln_post.requires_grad_(True);visual.proj.requires_grad_(True)
    visual.set_grad_checkpointing(True)
    return visual

class GroundedCLIP(nn.Module):
    def __init__(self,visual,bank,variant,blocks=2,seed=42):
        super().__init__();self.variant=variant
        self.global_visual=activate_tail(visual,blocks)
        self.register_buffer('class_text',bank['class_text'].float().clone());self.register_buffer('attribute_text',bank['attribute_text'].float().clone())
        self.logit_scale=nn.Parameter(torch.tensor(math.log(20.)))
        if variant!='finetune':
            self.attribute_visual=copy.deepcopy(self.global_visual)
            self.attribute_projection=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Linear(512,512),nn.LayerNorm(512))
            self.caf=nn.TransformerEncoderLayer(512,8,dim_feedforward=1024,dropout=0.,batch_first=True,norm_first=True)
            self.fusion_gate=nn.Parameter(torch.tensor(.02))
        rng=random.Random(seed+781);order=list(range(len(self.attribute_text)))
        while True:
            rng.shuffle(order)
            if all(i!=j for i,j in enumerate(order)):break
        self.register_buffer('wrong_attributes',torch.tensor(order))

    def attribute_values(self,ids,intervention=None):
        if self.variant=='shuffled' or intervention=='permuted':ids=self.wrong_attributes[ids]
        values=self.attribute_text[ids]
        if self.variant=='region_only' or intervention=='zero_text':values=torch.zeros_like(values)
        return values

    def forward(self,images,attribute_ids,valid,intervention=None):
        whole=F.normalize(self.global_visual(images[:,0]).float(),dim=-1);embedding=whole
        regions=torch.zeros(len(images),2,512,device=whole.device,dtype=whole.dtype)
        if self.variant!='finetune':
            regions=F.normalize(self.attribute_visual(images[:,1:].flatten(0,1)).float(),dim=-1).reshape(len(images),2,512)
            text=self.attribute_values(attribute_ids,intervention)
            tokens=self.attribute_projection(torch.cat([regions,text],dim=-1))
            sequence=torch.cat([whole[:,None],tokens],dim=1)
            padding=torch.cat([torch.zeros(len(images),1,dtype=torch.bool,device=whole.device),~valid],dim=1)
            context=self.caf(sequence,src_key_padding_mask=padding)[:,0].float()
            fused=F.normalize(whole+self.fusion_gate.tanh()*F.normalize(context,dim=-1),dim=-1)
            # Missing detections must fall back exactly to the whole-image path.
            embedding=torch.where(valid.any(-1,keepdim=True),fused,whole)
        return dict(embedding=embedding,global_embedding=whole,regions=regions)

def build_model(bank,variant,protocol,seed=42):
    clip,transform,_=load_clip();visual=clip.visual;del clip
    model=GroundedCLIP(visual,bank,variant,protocol['trainable_visual_blocks'],seed)
    return model,transform

def multi_positive_contrastive(embedding,text,labels,scale):
    logits=scale*embedding@text.T;positive=labels[:,None]==labels[None,:]
    # Multiple photos of the same class share the same text and are all positives.
    def direction(x,p):return (torch.logsumexp(x,dim=-1)-torch.logsumexp(x.masked_fill(~p,-torch.inf),dim=-1)).mean()
    return .5*(direction(logits,positive)+direction(logits.T,positive.T))

def objective(outputs,model,labels,ids,valid,confidence,teacher,protocol):
    seen=torch.tensor(protocol['seen_classes'],device=labels.device);matches=labels[:,None]==seen[None,:]
    if not matches.any(-1).all():raise ValueError('Unseen labels cannot enter training objective')
    scale=model.logit_scale.exp().clamp(max=100)
    e=outputs['embedding'];g=outputs['global_embedding']
    ce=F.cross_entropy(scale*e@model.class_text[seen].T,matches.long().argmax(-1))
    contrast=multi_positive_contrastive(e,model.class_text[labels],labels,scale)
    preserve=.5*((1-(g*teacher).sum(-1)).mean()+(1-(e*teacher).sum(-1)).mean())
    region=torch.zeros((),device=e.device)
    if model.variant in ('ag','shuffled'):
        text=model.attribute_values(ids);weights=confidence*valid
        region=((1-(outputs['regions']*text).sum(-1))*weights).sum()/weights.sum().clamp_min(1e-8)
    loss=protocol['class_ce_weight']*ce+protocol['contrastive_weight']*contrast+protocol['preserve_weight']*preserve+protocol['region_alignment_weight']*region
    return loss,dict(ce=float(ce.detach()),contrastive=float(contrast.detach()),preserve=float(preserve.detach()),region=float(region.detach()))

def gradient_cached_step(model,batch,optimizer,protocol):
    """Exact two-pass feature-gradient caching, not independent tiny-batch losses."""
    optimizer.zero_grad(set_to_none=True);n=len(batch['labels']);micro=protocol['micro_batch'];chunks=[]
    def forward(start):
        sl=slice(start,start+micro)
        with torch.autocast('cuda',dtype=torch.bfloat16):return model(batch['images'][sl].cuda(),batch['attribute_ids'][sl].cuda(),batch['valid'][sl].cuda())
    with torch.no_grad():
        for start in range(0,n,micro):chunks.append({k:v.detach() for k,v in forward(start).items()})
    leaves={k:torch.cat([c[k] for c in chunks]).requires_grad_(True) for k in chunks[0]}
    loss,values=objective(leaves,model,*[batch[k].cuda() for k in ['labels','attribute_ids','valid','confidence','teacher']],protocol)
    if not torch.isfinite(loss):raise ValueError('Nonfinite grounded loss')
    loss.backward();grads={k:v.grad.detach() if v.grad is not None else torch.zeros_like(v) for k,v in leaves.items()}
    # Same inputs and deterministic modules in both passes; text/logit-scale gradients
    # were obtained above. Recompute only visual/CAF graphs, one microbatch at a time.
    replay_error=0.
    for start in range(0,n,micro):
        outputs=forward(start);sl=slice(start,start+micro)
        replay_error=max(replay_error,max(float((outputs[k].detach()-leaves[k][sl].detach()).abs().max()) for k in outputs))
        surrogate=sum((value*grads[k][sl]).sum() for k,value in outputs.items() if value.requires_grad)
        surrogate.backward()
    if replay_error>1e-5:raise ValueError(f'Gradient-cache replay is stochastic: {replay_error}')
    gradient_norms={key:sum(float(p.grad.detach().float().square().sum()) for name,p in model.named_parameters() if name.startswith(key) and p.grad is not None)**.5 for key in ['global_visual','attribute_visual','caf','attribute_projection']}
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    optimizer.step()
    return dict(loss=float(loss.detach()),**values,replay_max_error=replay_error,gradient_norms=gradient_norms)
