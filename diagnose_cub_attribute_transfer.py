"""Do learned attribute probes transfer from seen to unseen species?"""
import argparse,json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score
from data_tools import ROOT,write_json,digest
from predict_cub_tokens import configure
from multimodal import cub_token_experiment as exp


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',default='cub_rich_v1');args=ap.parse_args();configure(args.version);torch.set_num_threads(4)
    bank=exp.text_bank();saved=torch.load(exp.OUT/'probe.pt',weights_only=True);a=bank['attribute_text'].shape[0];probe=nn.Linear(bank['attribute_text'].shape[-1],a);probe.load_state_dict(saved['state']);probe.eval()
    groups={}
    for stage in ['development','evaluation']:
        data=exp.load_data(stage)
        for kind in ['seen','unseen']:
            role=('dev_' if stage=='development' else 'eval_')+kind
            ix=[i for i,r in enumerate(data['rows']) if r['role']==role];assert ix,role
            image=data['native_views'][ix,0];target=data['targets'][ix,0]
            groups[role]=dict(target=target,trained=probe(image).sigmoid(),native=(10*image@(bank['attribute_text']-bank['negative_text']).T).sigmoid())
    common=[j for j in range(a) if all(int((g['target'][:,j]==1).sum())>=3 and int((g['target'][:,j]==0).sum())>=3 for g in groups.values())]
    assert common;rows=[]
    for name,g in groups.items():
        for model in ['trained','native']:
            aps=[];aucs=[];prevalence=[];brier=[];baseline=[]
            for j in common:
                mask=g['target'][:,j]>=0;y=g['target'][mask,j].numpy();q=g[model][mask,j].numpy()
                aps.append(average_precision_score(y,q));aucs.append(roc_auc_score(y,q));prevalence.append(y.mean());brier.append(np.mean((q-y)**2));baseline.append(np.mean((float(saved['mean_probability'][0,j])-y)**2))
            rows.append(dict(group=name,model=model,images=len(g['target']),common_attributes=len(common),mAP=float(np.mean(aps))*100,AUROC=float(np.mean(aucs))*100,mean_positive_prevalence=float(np.mean(prevalence))*100,Brier=float(np.mean(brier)),training_mean_Brier=float(np.mean(baseline))))
    result=dict(version=args.version,source_sha256=digest(__file__),probe_sha256=digest(exp.OUT/'probe.pt'),common_columns=[bank['columns'][j] for j in common] if 'columns' in bank else common,rows=rows,scope='Posthoc diagnosis only. Same attributes evaluated across all four groups, requiring >=3 known positives and negatives in each. This evaluation eligibility filter uses final labels but never changes model selection or inference. AP depends on prevalence; AUROC provides a complementary ranking measure. Native sigmoid similarity is not probability calibrated; Brier comparison is diagnostic, not a fair probability calibration test.')
    write_json(exp.REPORT/'attribute_transfer.json',result);print(json.dumps(rows))


if __name__=='__main__':main()
