"""Matched predicted-attribute models; annotations are loss targets, never inputs."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .models import load_clip
from .grounded_model import activate_tail,multi_positive_contrastive

class CubModel(nn.Module):
    def __init__(self,visual,bank,variant,p):
        super().__init__();self.variant=variant;self.global_visual=activate_tail(visual,p['trainable_visual_blocks'])
        self.register_buffer('class_text',bank['class_text'].clone());self.register_buffer('attribute_text',bank['attribute_text'].clone())
        self.attr_head=nn.Linear(512,len(self.attribute_text));self.logit_scale=nn.Parameter(torch.tensor(math.log(20.)))
        with torch.no_grad():self.attr_head.weight.copy_(10*(bank['attribute_text']-bank['negative_text']));self.attr_head.bias.zero_()
        if variant in ['finetune','region_only']:self.attr_head.requires_grad_(False)
        if variant!='finetune':
            self.attribute_visual=copy.deepcopy(self.global_visual)
            self.projection=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Linear(512,512),nn.LayerNorm(512))
            self.caf=nn.TransformerEncoderLayer(512,8,1024,dropout=0.,batch_first=True,norm_first=True);self.gate=nn.Parameter(torch.tensor(.02))

    def forward(self,images,valid,intervention=None):
        whole=F.normalize(self.global_visual(images[:,0]).float(),dim=-1);regions=torch.zeros(len(images),2,512,device=whole.device);e=whole
        if self.variant!='finetune':regions=F.normalize(self.attribute_visual(images[:,1:].flatten(0,1)).float(),dim=-1).reshape(len(images),2,512)
        views=torch.cat([whole[:,None],regions],dim=1);attr=self.attr_head(views).float()
        if self.variant!='finetune':
            weights=attr[:,1:].sigmoid()
            if intervention=='permuted':weights=weights.roll(1,dims=-1)
            text=(weights/weights.sum(-1,keepdim=True).clamp_min(1e-8))@self.attribute_text
            if self.variant=='region_only' or intervention=='zero_text':text=torch.zeros_like(text)
            token=self.projection(torch.cat([regions,text],dim=-1));seq=torch.cat([whole[:,None],token],dim=1)
            padding=torch.cat([torch.zeros(len(images),1,dtype=torch.bool,device=whole.device),~valid],dim=1)
            context=self.caf(seq,src_key_padding_mask=padding)[:,0].float()
            fused=F.normalize(whole+self.gate.tanh()*F.normalize(context,dim=-1),dim=-1);e=torch.where(valid.any(-1,keepdim=True),fused,whole)
        return dict(embedding=e,global_embedding=whole,view_logits=attr)

def build(bank,variant,p):
    clip,transform,_=load_clip();visual=clip.visual;del clip
    return CubModel(visual,bank,variant,p),transform

def masked_bce(logits,target):
    mask=target>=0;values=F.binary_cross_entropy_with_logits(logits,target.clamp(0,1),reduction='none')
    return (values*mask).sum()/mask.sum().clamp_min(1)

def objective(o,model,b,p):
    labels=b['labels'];seen=torch.tensor(p['seen_classes'],device=labels.device);matched=labels[:,None]==seen[None,:]
    if not matched.any(-1).all():raise ValueError('Held-out labels in training')
    e=o['embedding'];g=o['global_embedding'];scale=model.logit_scale.exp().clamp(max=100)
    ce=F.cross_entropy(scale*e@model.class_text[seen].T,matched.long().argmax(-1));contrast=multi_positive_contrastive(e,model.class_text[labels],labels,scale)
    preserve=.5*((1-(g*b['teacher']).sum(-1)).mean()+(1-(e*b['teacher']).sum(-1)).mean())
    attr=torch.zeros((),device=e.device);regional=attr
    if model.variant=='automatic':attr=F.binary_cross_entropy_with_logits(o['view_logits'][:,0],b['pseudo'][:,0])
    if model.variant in ['gold','gold_region','shuffled']:
        target=b['shuffled_targets'] if model.variant=='shuffled' else b['targets']
        attr=masked_bce(o['view_logits'][:,0],target[:,0])
        if model.variant=='gold_region':regional=masked_bce(o['view_logits'][:,1:],target[:,1:])
    loss=p['class_ce_weight']*ce+p['contrastive_weight']*contrast+p['preserve_weight']*preserve+p['attribute_loss_weight']*attr+p['regional_attribute_weight']*regional
    return loss,{k:float(v.detach()) for k,v in dict(ce=ce,contrast=contrast,preserve=preserve,attribute=attr,regional=regional).items()}

def cached_step(model,b,optimizer,p):
    optimizer.zero_grad(set_to_none=True);n=len(b['labels']);micro=p['micro_batch']
    def forward(i):
        with torch.autocast('cuda',dtype=torch.bfloat16):return model(b['images'][i:i+micro].cuda(),b['valid'][i:i+micro].cuda())
    with torch.no_grad():parts=[forward(i) for i in range(0,n,micro)]
    leaves={k:torch.cat([x[k] for x in parts]).detach().requires_grad_(True) for k in parts[0]}
    gpu={k:v.cuda() for k,v in b.items() if k not in ['images','valid']};loss,terms=objective(leaves,model,gpu,p)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();grads={k:x.grad.detach() if x.grad is not None else torch.zeros_like(x) for k,x in leaves.items()};error=0.
    for i in range(0,n,micro):
        out=forward(i);error=max(error,max(float((v.detach()-leaves[k][i:i+micro].detach()).abs().max()) for k,v in out.items()))
        sum((v*grads[k][i:i+micro]).sum() for k,v in out.items() if v.requires_grad).backward()
    if error>1e-5:raise ValueError(f'Non-deterministic gradient replay: {error}')
    norms={prefix:sum(float(t.grad.detach().square().sum()) for name,t in model.named_parameters() if name.startswith(prefix) and t.grad is not None)**.5 for prefix in ['global_visual','attribute_visual','attr_head','caf']}
    nn.utils.clip_grad_norm_([t for t in model.parameters() if t.requires_grad],1.);optimizer.step()
    return dict(loss=float(loss.detach()),**terms,replay_error=error,gradient_norms=norms)
