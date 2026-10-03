"""Native CoCa prefix caching and differentiable visual/text tails."""
import copy
import open_clip
import torch
from torch import nn
from torch.nn import functional as F
from coca_download import WEIGHT

def load_full():
    model,_,transform=open_clip.create_model_and_transforms('coca_ViT-L-14',pretrained=str(WEIGHT),device='cpu')
    # Caption generation is unused in the AG paper's symmetric contrastive objective.
    del model.text_decoder
    model.requires_grad_(False).eval()
    return model,transform,open_clip.get_tokenizer('coca_ViT-L-14')

def image_prefix(model,x):
    v=model.visual;assert v.transformer.batch_first
    x=v._embeds(x)
    for block in v.transformer.resblocks[:-1]:x=block(x)
    return x

def text_prefix(model,tokens):
    t=model.text;assert t.transformer.batch_first
    x,mask=t._embeds(tokens)
    for block in t.transformer.resblocks[:-1]:x=block(x,attn_mask=mask)
    return x,mask

class Tails(nn.Module):
    def __init__(self,model):
        super().__init__();v=model.visual;t=model.text
        assert v.attn_pool is not None and v.attn_pool_contrastive is None and v.pool_type=='tok'
        assert t.cls_emb is not None and not isinstance(t.text_projection,nn.Linear)
        self.visual_block=copy.deepcopy(v.transformer.resblocks[-1])
        self.visual_pool=copy.deepcopy(v.attn_pool)
        self.visual_ln=copy.deepcopy(v.ln_post)
        self.visual_projection=nn.Parameter(v.proj.detach().clone())
        self.text_block=copy.deepcopy(t.transformer.resblocks[-1])
        self.text_ln=copy.deepcopy(t.ln_final)
        self.text_projection=nn.Parameter(t.text_projection.detach().clone())
        self.logit_scale=nn.Parameter(model.logit_scale.detach().clone())
        self.requires_grad_(True)

    def image(self,x):
        x=self.visual_block(x)
        x=self.visual_ln(self.visual_pool(x))[:,0]@self.visual_projection
        return F.normalize(x.float(),dim=-1)

    def text(self,x,mask):
        x=self.text_block(x,attn_mask=mask)
        x=self.text_ln(x[:,-1])@self.text_projection
        return F.normalize(x.float(),dim=-1)

class Fusion(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        if variant=='baseline':return
        self.attribute_mlp=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.Linear(256,768))
        # Native global embedding is preserved at initialization for every AG arm.
        nn.init.zeros_(self.attribute_mlp[-1].weight);nn.init.zeros_(self.attribute_mlp[-1].bias)
        if variant!='attributes':
            self.input=nn.Linear(768,256)
            self.caf=nn.TransformerEncoderLayer(256,4,512,dropout=0.,batch_first=True,norm_first=True)
            self.output=nn.Linear(256,768)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,g,r,a,valid,confidence):
        if self.variant=='baseline':return g
        if self.variant=='regions':a=torch.zeros_like(a)
        weights=valid.float()
        if self.variant=='confidence':weights=weights*confidence.clamp_min(0)
        denom=weights.sum(-1,keepdim=True)
        uniform=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
        weights=torch.where(denom>0,weights/denom.clamp_min(1e-12),uniform)
        pooled=(torch.cat([r,a],dim=-1)*weights[...,None]).sum(1)
        attr=self.attribute_mlp(pooled)
        z=g+attr
        if self.variant!='attributes':
            seq=torch.stack([self.input(g),self.input(attr)],dim=1)
            z=z+self.output(self.caf(seq)[:,0])
        return torch.where(valid.any(-1,keepdim=True),F.normalize(z.float(),dim=-1),g)

def symmetric_loss(images,text,logit_scale):
    logits=logit_scale.exp().clamp(max=100)*images@text.T
    target=torch.arange(len(images),device=images.device)
    return (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2

def select_text(prefix,mask,indices):
    # CoCa mask shape is (N * heads,L,L); slicing must preserve all heads.
    n=len(prefix);h=mask.shape[0]//n
    return prefix[indices],mask.reshape(n,h,*mask.shape[1:])[indices].flatten(0,1)
