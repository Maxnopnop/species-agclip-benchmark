"""Is fusion mostly learning a class-wise offset rather than image evidence?"""
import argparse,json
import torch
from data_tools import ROOT,write_json,digest
from predict_cub_tokens import configure
from multimodal import cub_token_experiment as exp
from multimodal.grounded_experiment import measures,rank


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='cub_tokens_v1');a=parser.parse_args();configure(a.version)
    exp.setup();p,_=exp.prepare();data=exp.load_data('development');lock=json.loads((exp.OUT/'selection_locked.json').read_text());records=[]
    ti=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen']
    for chosen in lock['selected']:
        model,_=exp.restore(ROOT/chosen['checkpoint']);deltas=[]
        for start in range(0,len(ti),32):
            ix=ti[start:start+32];o=model(data['native_views'][ix].cuda(),data['valid'][ix].cuda());deltas.append((o['logits']-o['native_logits']).cpu())
        delta=torch.cat(deltas);offset=delta.mean(0);ratio=float(offset.square().sum()/delta.square().sum(-1).mean().clamp_min(1e-12))
        # De-mean each logit vector first to ignore a uniform offset that cannot
        # change any softmax or argmax classification decision.
        relevant=delta-delta.mean(-1,keepdim=True);mean=relevant.mean(0)
        relevant_ratio=float(mean.square().sum()/relevant.square().sum(-1).mean().clamp_min(1e-12))
        _,dev=exp.evaluate(model,data,p,'development');options=[]
        for weight in [0.,.5,1.]:
            logits=dev['logits']-weight*offset
            metrics,_=measures(logits,dev['labels'],p,'development');options.append(dict(weight=weight,metrics=metrics))
        selected=max(options,key=lambda r:rank(r['metrics']))
        row=dict(**chosen,common_offset_energy_fraction=ratio,class_contrast_common_offset_energy_fraction=relevant_ratio,selected=selected,development_grid=options,offset=offset.tolist())
        records.append(row)
    selection=dict(records=records,source_sha256=digest(__file__),scope='Posthoc mechanism test: subtract only a class-wise correction estimated on training images. Removal weight selected on development; final scores evaluated only afterward. This is bias correction, not an attribute-specific method.')
    write_json(exp.OUT/'offset_selection_locked.json',selection);final=[]
    for r in records:
        pred=torch.load((ROOT/r['checkpoint']).parent/'final_predictions.pt',weights_only=True);logits=pred['logits']-r['selected']['weight']*torch.tensor(r['offset']);metrics,classes=measures(logits,pred['labels'],p,'evaluation')
        final.append(dict(variant=r['variant'],seed=r['seed'],weight=r['selected']['weight'],common_offset_fraction=r['class_contrast_common_offset_energy_fraction'],metrics=metrics,changed_predictions=int((classes!=pred['predictions']).sum())))
    write_json(exp.REPORT/'offset_diagnosis.json',dict(selection=selection,final=final))
    print(json.dumps([(r['variant'],r['seed'],round(r['common_offset_fraction'],4),r['weight'],round(r['metrics']['H'],2)) for r in final]))


if __name__=='__main__':main()
