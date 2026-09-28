"""Development-only check: does stricter crowd-label certainty improve readout?"""
import json
import torch
from torch import nn
from data_tools import ROOT,write_json,digest
from run import seed_all
from multimodal.cub_data import BASE,lines,prepare,text_bank
from multimodal.cub_token_experiment import load_data
from multimodal.cub_rich_data import load_data as rich_data,text_bank as rich_bank
from multimodal.cub_model import masked_bce
from multimodal.visible_train import attribute_metrics


def main():
    seed_all(42);_,manifest=prepare();original=load_data('development');index={r['image_id']:i for i,r in enumerate(original['rows'])};certainty=torch.zeros(len(index),312,dtype=torch.long)
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,_,c=map(int,r[:4])
        if iid in index:certainty[index[iid],aid-1]=c
    ti=[i for i,r in enumerate(original['rows']) if r['role']=='train_seen'];vi=[i for i,r in enumerate(original['rows']) if r['role']!='train_seen'];records=[]
    for version in ['24','158']:
        data=original if version=='24' else rich_data('development');bank=text_bank() if version=='24' else rich_bank();columns=[a['source_id']-1 for a in manifest['attributes']] if version=='24' else bank['columns'];x=data['native_views'];gold=data['targets']
        strict=gold.clone();strict[~(certainty[:,columns]>=4)[:,None].expand_as(strict)]=-1
        for threshold in [3,4]:
            y=gold if threshold==3 else strict;best=None;history=[]
            for lr in [.01,.001]:
                seed_all(42);model=nn.Linear(512,len(columns))
                with torch.no_grad():model.weight.copy_(10*(bank['attribute_text']-bank['negative_text']));model.bias.zero_()
                opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
                for step in range(601):
                    if step in [0,100,300,600]:
                        with torch.no_grad():pred=model(x[vi]).sigmoid()
                        gm=attribute_metrics(strict[vi,0],pred[:,0])['attribute_map'];rm=attribute_metrics(strict[vi,1:].flatten(0,1),pred[:,1:].flatten(0,1))['attribute_map'];row=dict(lr=lr,step=step,strict_global_map=gm,strict_regional_map=rm,selection_map=(gm+rm)/2);history.append(row)
                        if best is None or row['selection_map']>best['selection_map']:best=row
                    if step==600:break
                    opt.zero_grad();logit=model(x[ti]);loss=masked_bce(logit[:,0],y[ti,0])+.5*masked_bce(logit[:,1:],y[ti,1:]);loss.backward();opt.step()
            record=dict(attributes=len(columns),training_minimum_certainty=threshold,known_training_targets=int((y[ti]>=0).sum()),selection=best,history=history);records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)
    write_json(ROOT/'reports/cub_followup_v1/certainty_diagnosis.json',dict(records=records,source_sha256=digest(__file__),scope='Development-only label-quality diagnostic. Both training thresholds evaluated and selected on exactly the same certainty-4 development targets. Unknown/visibility masks retained. Selected development performance is not a final test or proof that certainty-4 labels are true.'))


if __name__=='__main__':main()
