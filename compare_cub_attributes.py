"""Paired exploratory contrasts for separately locked attribute-best checkpoints."""
import json,math
import numpy as np
import torch
from data_tools import ROOT,write_json
from multimodal.cub_data import OUT,REPORT,prepare
from multimodal.visible_train import average_precision

def metrics(a,b,p):
    assert a['paths']==b['paths'];labels=a['labels'].numpy();unseen=p['eval_unseen_classes'];mask=np.isin(labels,unseen)
    ac=(np.array(unseen)[a['logits'].numpy()[mask][:,unseen].argmax(1)]==labels[mask]);bc=(np.array(unseen)[b['logits'].numpy()[mask][:,unseen].argmax(1)]==labels[mask]);wins=int((ac&~bc).sum());losses=int((~ac&bc).sum());n=wins+losses
    exact=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    target=a['targets'][:,0].numpy();aa=a['attributes'][:,0].numpy();bb=b['attributes'][:,0].numpy();rng=np.random.default_rng(20260928);groups=[np.where(labels==c)[0] for c in np.unique(labels)];deltas=[]
    def map_(scores,ids):
        values=[]
        for j in range(target.shape[1]):
            valid=ids[target[ids,j]>=0];y=target[valid,j]
            if (y==1).any() and (y==0).any():values.append(average_precision(y,scores[valid,j]))
        return np.mean(values)
    ids=np.arange(len(labels));delta=100*(map_(aa,ids)-map_(bb,ids))
    for _ in range(500):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);deltas.append(100*(map_(aa,ix)-map_(bb,ix)))
    return dict(ZSL_difference_pp=100*float(ac.mean()-bc.mean()),unseen_photos=len(ac),A_correct_B_wrong=wins,A_wrong_B_correct=losses,exact_mcnemar_p=exact,attribute_mAP_difference_pp=float(delta),attribute_mAP_bootstrap_95pct=np.quantile(deltas,[.025,.975]).tolist())

def main():
    p,_=prepare();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));pred={'native':torch.load(OUT/'native_predictions.pt',weights_only=True)}
    for r in lock['attribute_selected']:pred[r['variant']]=torch.load((ROOT/r['checkpoint']).parent/'attribute_final_predictions.pt',weights_only=True)
    result=[dict(comparison='gold minus '+b,**metrics(pred['gold'],pred[b],p)) for b in ['native','finetune','region_only','automatic','shuffled']]
    result.append(dict(comparison='gold_region minus gold',**metrics(pred['gold_region'],pred['gold'],p)))
    write_json(REPORT/'attribute_selected_contrasts.json',dict(comparisons=result,scope='Attribute-selected checkpoints, exploratory post-hoc contrasts. No multiplicity adjustment; one seed. Attribute bootstrap uses 500 paired within-class image resamples; ZSL McNemar uses the fixed 120 unseen photos. No retuning.'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
