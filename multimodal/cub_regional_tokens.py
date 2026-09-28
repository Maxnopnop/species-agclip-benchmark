"""Frozen native global features, trainable regional visual tail, separate tokens."""
import torch
from torch import nn
from torch.nn import functional as F
from .models import load_clip
from .grounded_model import activate_tail,multi_positive_contrastive
from .cub_tokens import TokenFusion
from .cub_model import masked_bce


class RegionalTokens(nn.Module):
    def __init__(self,visual,bank,probe,variant,c):
        super().__init__();self.variant=variant
        self.attribute_visual=activate_tail(visual,c['trainable_visual_blocks'])
        self.fusion=TokenFusion(bank,probe['state'],probe['mean_probability'],'region_only' if variant=='no_attribute_loss' else variant,c)

    def encode_regions(self,images):
        with torch.autocast('cuda',dtype=torch.bfloat16):features=self.attribute_visual(images[:,1:].flatten(0,1))
        return F.normalize(features.float(),dim=-1).reshape(len(images),2,512)

    def from_regions(self,regions,global_features,valid,intervention=None):
        # Global image features are immutable original-CLIP cache values. There
        # is no trainable global image encoder or trainable class/temperature.
        views=torch.cat([global_features[:,None],regions],1)
        output=self.fusion(views,valid,intervention)
        output['view_logits']=self.fusion.probe(views)
        output['regions']=regions
        return output


def build(bank,probe,variant,c):
    clip,transform,_=load_clip();visual=clip.visual;del clip
    return RegionalTokens(visual,bank,probe,variant,c),transform


def objective(o,model,b,p,c):
    seen=torch.tensor(p['seen_classes'],device=b['labels'].device);matches=b['labels'][:,None]==seen
    if not matches.any(-1).all():raise ValueError('Held-out training labels')
    ce=F.cross_entropy(o['logits'][:,seen],matches.long().argmax(-1))
    contrast=multi_positive_contrastive(o['embedding'],model.fusion.class_text[b['labels']],b['labels'],20.)
    preserve=(1-(o['embedding']*b['native_views'][:,0]).sum(-1)).mean()
    region_preserve=((1-(o['regions']*b['native_views'][:,1:]).sum(-1))*b['valid']).sum()/b['valid'].sum().clamp_min(1)
    attributes=masked_bce(o['view_logits'][:,1:],b['targets'][:,1:]) if model.variant!='no_attribute_loss' else ce*0
    loss=.5*ce+.5*contrast+c['preserve_weight']*preserve+c['regional_preserve_weight']*region_preserve+c['regional_attribute_weight']*attributes
    return loss,dict(ce=float(ce.detach()),contrast=float(contrast.detach()),attribute=float(attributes.detach()),preserve=float(preserve.detach()),region_preserve=float(region_preserve.detach()))


def cached_step(model,b,optimizer,p,c):
    optimizer.zero_grad(set_to_none=True);n=len(b['labels']);micro=c['micro_batch']
    with torch.no_grad():cached=torch.cat([model.encode_regions(b['images'][i:i+micro].cuda()) for i in range(0,n,micro)])
    regions=cached.detach().requires_grad_(True);gpu={k:v.cuda() for k,v in b.items() if k!='images'}
    output=model.from_regions(regions,gpu['native_views'][:,0],gpu['valid']);loss,terms=objective(output,model,gpu,p,c)
    if not torch.isfinite(loss):raise ValueError('Nonfinite regional loss')
    loss.backward();gradient=regions.grad.detach();error=0.
    for i in range(0,n,micro):
        encoded=model.encode_regions(b['images'][i:i+micro].cuda());error=max(error,float((encoded.detach()-cached[i:i+micro]).abs().max()))
        (encoded*gradient[i:i+micro]).sum().backward()
    if error>1e-5:raise ValueError('Regional gradient replay mismatch')
    norms={prefix:sum(float(t.grad.detach().square().sum()) for name,t in model.named_parameters() if name.startswith(prefix) and t.grad is not None)**.5 for prefix in ['attribute_visual','fusion']}
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    optimizer.step();return dict(loss=float(loss.detach()),**terms,replay_error=error,gradient_norms=norms)
