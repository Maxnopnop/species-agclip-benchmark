"""Matched detector-confidence pooling pilot; preserve historical experiments."""
import argparse, json, random, copy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT, digest, write_json
from run import seed_all, metrics_from_confusion
from multimodal.models import DIMS, load_backbone
from multimodal.expanded import is_better
from multimodal.experiment import split_indices

CONFIG=ROOT/'configs/confidence_v1.json'
OUT=ROOT/'reports/confidence_v1'
CACHE=ROOT/'cache/confidence_v1'
RUN=ROOT/'runs/confidence_v1'

def pool_weights(confidence,valid,mode):
    valid=valid.bool();uniform=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
    if mode in ('ag_uniform','region_only'):return uniform
    values=confidence.clamp_min(0)*valid
    if mode=='ag_shuffled':
        # Two valid boxes: swap weights. One/zero valid boxes: unchanged.
        values=torch.where(valid.all(-1,keepdim=True),values.flip(-1),values)
    total=values.sum(-1,keepdim=True)
    return torch.where(total>0,values/total.clamp_min(1e-12),uniform)

class Head(nn.Module):
    def __init__(self,name,variant,bank,c):
        super().__init__();self.variant=variant;self.native=name=='clip_vit_b32'
        classes=len(bank['class_text']);self.ag=variant in ('region_only','ag_uniform','ag_confidence','ag_shuffled')
        self.projection=nn.Linear(DIMS[name],classes if variant=='visual' else 512)
        if variant=='visual':return
        if self.native:nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.logit_scale=nn.Parameter(torch.tensor(np.log(14.2857),dtype=torch.float32))
        text=bank['class_text'].float().clone()
        if variant=='random_codes':
            generator=torch.Generator().manual_seed(c['random_code_seed'])
            text=F.normalize(torch.randn(classes,512,generator=generator),dim=-1)
        self.register_buffer('class_text',text)
        if self.ag:
            self.register_buffer('attribute_text',bank['attribute_text'].float().clone())
            self.token_encoder=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128))
            self.global_token=nn.Linear(512,128)
            self.caf=nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True,norm_first=True)
            self.output=nn.Linear(128,512);self.gate=nn.Parameter(torch.tensor(.1))
    def project(self,x):
        return F.normalize(x+self.projection(x) if self.native else self.projection(x),dim=-1)
    def forward(self,g,r,ids,valid,confidence):
        if self.variant=='visual':return self.projection(g)
        whole=self.project(g);embedding=whole
        if self.ag:
            text=self.attribute_text[ids]
            if self.variant=='region_only':text=torch.zeros_like(text)
            tokens=self.token_encoder(torch.cat([self.project(r),text],dim=-1))
            weights=pool_weights(confidence,valid,self.variant)
            pooled=(tokens*weights[...,None]).sum(1)
            sequence=torch.stack([self.global_token(whole),pooled],dim=1)
            context=self.caf(sequence)[:,0]
            fused=F.normalize(whole+self.gate.tanh()*F.normalize(self.output(context),dim=-1),dim=-1)
            embedding=torch.where(valid.any(-1,keepdim=True),fused,whole)
        return self.logit_scale.exp().clamp(max=100)*embedding@self.class_text.T

def inputs(cache,ix):
    return [cache[k][ix] for k in ('g','r','ids','valid','confidence')]

def definition():
    c=json.loads(CONFIG.read_text());old=json.loads((ROOT/'configs/grounded_v1.json').read_text())
    original=json.loads((ROOT/'data/expanded20/manifest.json').read_text())
    classes=old['seen_classes'];mapping={v:i for i,v in enumerate(classes)}
    rows=[dict(row,split=split,label=mapping[row['label']],original_label=row['label'])
          for split,items in original['splits'].items() for row in items if row['label'] in mapping]
    assert len(rows)==500 and len({r['sha256'] for r in rows})==500
    manifest=dict(rows=rows,classes=[original['classes'][v] for v in classes],image_root=original['image_root'])
    boxes={}
    for stage in ('development','evaluation'):
        record=json.loads((ROOT/f'cache/grounded_v1/regions_{stage}.json').read_text())
        assert record['metadata']['protocol_sha256']==digest(ROOT/'configs/grounded_v1.json')
        assert record['metadata']['implementation_sha256']==digest(ROOT/'multimodal/grounded_data.py')
        boxes.update(record['images'])
    assert all(r['path'] in boxes for r in rows)
    bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    bank=dict(class_text=bank['class_text'][classes],attribute_text=bank['attribute_text'])
    return c,manifest,boxes,bank

def source():
    paths=[CONFIG,Path(__file__),ROOT/'multimodal/models.py',ROOT/'multimodal/expanded.py',ROOT/'multimodal/experiment.py',ROOT/'run.py',ROOT/'multimodal/grounded_data.py',ROOT/'configs/grounded_v1.json',ROOT/'data/expanded20/manifest.json',ROOT/'cache/expanded20/text_bank.pt']
    paths += [ROOT/f'cache/grounded_v1/regions_{s}.json' for s in ('development','evaluation')]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def lock(c):
    s=source();path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['source']==s,'Protocol changed; use a new version.'
    else:write_json(path,dict(config=c,source=s,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return s

def prepare(c,m,boxes,s):
    CACHE.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'classes.json',m['classes'])
    for name in c['backbones']:
        target=CACHE/f'{name}.pt'
        if target.exists():assert torch.load(target,weights_only=True)['source']==s;continue
        model,transform,dim=load_backbone(name);model=model.cuda().eval()
        parts=[];ids=[];confidence=[];valid=[]
        with torch.inference_mode():
            for start in range(0,len(m['rows']),8):
                views=[]
                for row in m['rows'][start:start+8]:
                    path=Path(m['image_root'])/row['path'];assert digest(path)==row['sha256']
                    with Image.open(path) as im:im=im.convert('RGB')
                    rec=boxes[row['path']];n=len(rec['scores']);assert 0<=n<=2
                    views.append(transform(im))
                    views.extend(transform(im.crop(tuple(b))) for b in rec['boxes'])
                    views.extend(transform(im) for _ in range(2-n))
                    ids.append(rec['attribute_ids']+[0]*(2-n));confidence.append(rec['scores']+[0.]*(2-n));valid.append([True]*n+[False]*(2-n))
                encoded=F.normalize(model(torch.stack(views).cuda()).float(),dim=-1).cpu()
                parts.append(encoded.reshape(-1,3,dim))
                if start%80==0:print('features',name,start+len(views)//3,'/500',flush=True)
        features=torch.cat(parts)
        torch.save(dict(source=s,rows=m['rows'],classes=m['classes'],g=features[:,0],r=features[:,1:],ids=torch.tensor(ids),confidence=torch.tensor(confidence),valid=torch.tensor(valid),labels=torch.tensor([r['label'] for r in m['rows']])),target)
        del model;torch.cuda.empty_cache()

def load_cache(name,training=False):
    d=torch.load(CACHE/f'{name}.pt',weights_only=True)
    if training:
        ix=[i for i,r in enumerate(d['rows']) if r['split']!='test']
        d={k:(v[ix] if isinstance(v,torch.Tensor) else [v[i] for i in ix] if k=='rows' else v) for k,v in d.items()}
        assert len(d['rows'])==400
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in d.items()}

@torch.inference_mode()
def evaluate(model,d,ix):
    model.eval();probs=[];loss=0.
    for start in range(0,len(ix),64):
        batch=ix[start:start+64];logits=model(*inputs(d,batch));probs.append(logits.softmax(-1).cpu())
        loss+=float(F.cross_entropy(logits,d['labels'][batch],reduction='sum'))
    prob=torch.cat(probs);y=d['labels'][ix].cpu();cm=np.zeros((10,10),dtype=np.int64)
    np.add.at(cm,(y.numpy(),prob.argmax(-1).numpy()),1)
    return dict(**metrics_from_confusion(cm),loss=loss/len(ix)),prob,y

def train(model,d,ti,vi,c,seed):
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c['stage_steps'])
    rng=random.Random(seed);order=[];best=None;history=[]
    for step in range(1,c['stage_steps']+1):
        if not order:order=ti.copy();rng.shuffle(order)
        batch=order[:c['batch_size']];order=order[c['batch_size']:];model.train()
        loss=F.cross_entropy(model(*inputs(d,batch)),d['labels'][batch]);assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if step%c['validation_interval']==0:
            metrics,_,_=evaluate(model,d,vi);history.append(dict(step=step,**metrics))
            if is_better(metrics,best):best=metrics;best_step=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return dict(state_dict=state,validation=best,best_step=best_step,history=history,training=evaluate(model,d,ti)[0])

def train_all(c,bank,s):
    for name in c['backbones']:
        d=load_cache(name,True);vi=split_indices(d,'val')
        for seed in c['seeds']:
            ti=split_indices(d,'train',c['shots'],seed);assert len(ti)==200
            for variant in c['variants']:
                folder=RUN/f'{name}_{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{variant}.pt'
                if path.exists():assert torch.load(path,weights_only=True)['source']==s;continue
                initial=variant if variant in ('visual','random_codes') else 'text';initpath=folder/f'initial_{initial}.pt'
                if not initpath.exists():
                    seed_all(seed);model=Head(name,initial,bank,c).cuda();trained=train(model,d,ti,vi,c,seed)
                    torch.save(dict(source=s,**trained),initpath)
                initial_state=torch.load(initpath,weights_only=True);assert initial_state['source']==s
                seed_all(seed);model=Head(name,variant,bank,c).cuda()
                state=initial_state['state_dict']
                if model.ag:state={k:v for k,v in state.items() if k.startswith(('projection.','logit_scale','class_text'))}
                model.load_state_dict(state,strict=not model.ag)
                trained=train(model,d,ti,vi,c,seed)
                record=dict(source=s,backbone=name,seed=seed,variant=variant,train_paths=[d['rows'][i]['path'] for i in ti],parameters=sum(p.numel() for p in model.parameters()),**trained)
                torch.save(record,path)
                print('trained',name,seed,variant,'validation',trained['validation']['top1_accuracy'],flush=True)
        del d;torch.cuda.empty_cache()

def seal(c,s):
    records=[]
    for name in c['backbones']:
        for seed in c['seeds']:
            for variant in c['variants']:
                path=RUN/f'{name}_{seed}/{variant}.pt';ck=torch.load(path,weights_only=True);assert ck['source']==s
                records.append(dict(backbone=name,seed=seed,variant=variant,checkpoint_sha256=digest(path),validation=ck['validation'],best_step=ck['best_step']))
    assert len(records)==42
    obj=dict(source=s,checkpoints=records,feature_hashes={n:digest(CACHE/f'{n}.pt') for n in c['backbones']})
    path=OUT/'selection_locked.json'
    if path.exists():assert json.loads(path.read_text())==obj
    else:write_json(path,obj)

def score(c,bank,s):
    seal(c,s);rows=[];predictions={}
    for name in c['backbones']:
        d=load_cache(name);ix=split_indices(d,'test');assert len(ix)==100
        for seed in c['seeds']:
            for variant in c['variants']:
                ck=torch.load(RUN/f'{name}_{seed}/{variant}.pt',weights_only=True)
                assert ck['train_paths']==[d['rows'][i]['path'] for i in split_indices(d,'train',c['shots'],seed)]
                model=Head(name,variant,bank,c).cuda();model.load_state_dict(ck['state_dict'])
                metrics,p,y=evaluate(model,d,ix)
                rows.append(dict(backbone=name,seed=seed,variant=variant,parameters=ck['parameters'],training_accuracy=ck['training']['top1_accuracy'],**metrics))
                predictions[f'{name}/{seed}/{variant}']=dict(probabilities=p,labels=y,paths=[d['rows'][i]['path'] for i in ix])
        print('evaluated',name,flush=True)
    torch.save(predictions,RUN/'predictions.pt');write_json(OUT/'results.json',dict(source=s,rows=rows))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train','score','all']);args=parser.parse_args()
    torch.set_num_threads(4);c,m,boxes,bank=definition();s=lock(c)
    if args.action in ('prepare','all'):prepare(c,m,boxes,s)
    if args.action in ('train','all'):train_all(c,bank,s)
    if args.action in ('score','all'):score(c,bank,s)

if __name__=='__main__':main()
