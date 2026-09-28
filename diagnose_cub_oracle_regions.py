"""Oracle-location development diagnostic, explicitly excluded from deployment."""
import gc,json
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal.cub_data import BASE,lines,prepare,pixels
from multimodal.cub_rich_data import load_data,text_bank,PARTS,names
from multimodal.models import load_clip
from multimodal.grounded_data import image_tensors
from multimodal.cub_model import masked_bce
from multimodal.visible_train import attribute_metrics

OUT=ROOT/'runs/cub_oracle_region_diagnosis';OUT.mkdir(parents=True,exist_ok=True)


def build_cache(data,bank):
    path=OUT/'development.pt';metadata=dict(source_sha256=digest(__file__),manifest_sha256=digest(ROOT/'data/cub_attributes_v1/manifest.json'))
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==metadata;return saved
    index={r['image_id']:i for i,r in enumerate(data['rows'])};parts=defaultdict(dict);observations={};columns=bank['columns'];selected={a+1 for a in columns};naming=names()
    for r in lines(BASE/'parts/part_locs.txt'):
        if int(r[0]) in index:parts[int(r[0])][int(r[1])]=(float(r[2]),float(r[3]),int(r[4]))
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,pos,certainty=map(int,r[:4])
        if iid in index and aid in selected and certainty>=3:observations[iid,aid-1]=pos
    clip,transform,_=load_clip();clip=clip.cuda().eval();features=[];targets=[];valid=[];allboxes=[]
    groups=[[2,5,6,7,10,11,15],[1,3,4,9,13,14]]
    with torch.no_grad():
        for i,row in enumerate(data['rows']):
            iid=row['image_id'];w,h=row['size'];boxes=[]
            for group in groups:
                points=[parts[iid][pid][:2] for pid in group if parts[iid].get(pid,(0,0,0))[2]]
                if not points:continue
                xs,ys=zip(*points);cx,cy=(max(xs)+min(xs))/2,(max(ys)+min(ys))/2;side=max(32.,1.4*max(max(xs)-min(xs),max(ys)-min(ys)))
                boxes.append([max(0,cx-side/2),max(0,cy-side/2),min(w,cx+side/2),min(h,cy+side/2)])
            record=dict(boxes=boxes,attribute_ids=list(range(len(boxes))),scores=[1.]*len(boxes))
            image=BASE/'images'/row['path'];x,_,_,ok=image_tensors(image,record,transform)
            with torch.autocast('cuda',dtype=torch.bfloat16):f=F.normalize(clip.encode_image(x.cuda()).float(),dim=-1).cpu()
            y=torch.full_like(data['targets'][i],-1);y[0]=data['targets'][i,0]
            for v,(x1,y1,x2,y2) in enumerate(boxes,1):
                cx,cy=(x1+x2)/2,(y1+y2)/2;side=min(x2-x1,y2-y1);visible={pid for pid,(px,py,shown) in parts[iid].items() if shown and cx-side/2<=px<=cx+side/2 and cy-side/2<=py<=cy+side/2}
                for a,column in enumerate(columns):
                    field=naming[column].split('::')[0][4:];key=next((k for k in sorted(PARTS,key=len,reverse=True) if field==k or field.startswith(k+'_')),None)
                    if key is not None and visible&set(PARTS[key]) and (iid,column) in observations:y[v,a]=observations[iid,column]
            torch.testing.assert_close(f[0],data['native_views'][i,0],rtol=0,atol=0)
            features.append(f);targets.append(y);valid.append(ok);allboxes.append(boxes)
            if i%80==0:print(f'Oracle development crops {i}/{len(data["rows"])}',flush=True)
    result=dict(metadata=metadata,features=torch.stack(features),targets=torch.stack(targets),valid=torch.stack(valid),boxes=allboxes)
    torch.save(result,path);del clip;gc.collect();torch.cuda.empty_cache();return result


def main():
    seed_all(42);torch.backends.mha.set_fastpath_enabled(False);data=load_data('development');bank=text_bank();oracle=build_cache(data,bank)
    ti=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen'];vi=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen']
    common=data['targets'].clone();common[(oracle['targets']<0)|(data['targets']<0)]=-1
    assert torch.equal(oracle['targets'][common>=0],data['targets'][common>=0]);records=[]
    for variant,x,y in [('detected',data['native_views'],data['targets']),('oracle_location',oracle['features'],oracle['targets'])]:
        best=None;history=[]
        for lr in [.01,.001]:
            seed_all(42);model=nn.Linear(512,len(bank['attribute_text']))
            with torch.no_grad():model.weight.copy_(10*(bank['attribute_text']-bank['negative_text']));model.bias.zero_()
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
            for step in range(601):
                if step in [0,100,300,600]:
                    with torch.no_grad():prediction=model(x[vi]).sigmoid()
                    metrics=attribute_metrics(common[vi,1:].flatten(0,1),prediction[:,1:].flatten(0,1));global_map=attribute_metrics(common[vi,0],prediction[:,0])['attribute_map'];row=dict(lr=lr,step=step,common_regional_map=metrics['attribute_map'],global_map=global_map);history.append(row)
                    if best is None or row['common_regional_map']>best['common_regional_map']:best=row
                if step==600:break
                opt.zero_grad();logits=model(x[ti]);loss=masked_bce(logits[:,0],y[ti,0])+.5*masked_bce(logits[:,1:],y[ti,1:]);loss.backward();opt.step()
        result=dict(variant=variant,selected=best,history=history);records.append(result);print(json.dumps(dict(variant=variant,selected=best)),flush=True)
    write_json(ROOT/'reports/cub_followup_v1/oracle_region_diagnosis.json',dict(records=records,common_development_regional_labels=int((common[vi,1:]>=0).sum()),source_sha256=digest(__file__),scope='Development-only localization diagnostic. True annotated part points define two square crops, so oracle_location is NOT deployable, NOT an AG classification result and NOT a theoretical upper bound. Both feature sets use identical common evaluation label masks and identical probe grids; training crops/supervision differ. No final images were accessed.'))


if __name__=='__main__':main()
