"""Apply identical development-only calibration selection to every control."""
import argparse,json
import torch
from data_tools import ROOT,write_json,digest
from predict_cub_tokens import configure
from multimodal import cub_token_experiment as exp
from multimodal.grounded_experiment import measures,rank


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='cub_tokens_v1');args=parser.parse_args();configure(args.version)
    exp.setup();p,_=exp.prepare();data=exp.load_data('development');lock=json.loads((exp.OUT/'selection_locked.json').read_text());rows=[]
    def score(logits,labels,stage,gamma):
        x=logits.clone();x[:,p['seen_classes']]-=gamma
        metrics,pred=measures(x,labels,p,stage);return metrics,pred,x
    ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen']
    for chosen in [dict(variant='native',seed=0),*lock['selected']]:
        if chosen['variant']=='native':logits=20*data['native_views'][ids,0]@exp.text_bank()['class_text'].T;labels=data['labels'][ids]
        else:
            model,_=exp.restore(ROOT/chosen['checkpoint']);_,pred=exp.evaluate(model,data,p,'development');logits=pred['logits'];labels=pred['labels']
        grid=[dict(gamma=g,development=score(logits,labels,'development',g)[0]) for g in [0.,.25,.5,1.,2.,4.]]
        rows.append(dict(**chosen,selected=max(grid,key=lambda r:rank(r['development'])),grid=grid))
    selection=dict(rows=rows,source_sha256=digest(__file__),scope='Exploratory follow-up: equal development calibration grid for every trained variant, not an AG-only optimization.')
    write_json(exp.OUT/'calibration_locked.json',selection)
    final=[]
    for row in rows:
        path=exp.OUT/'native_predictions.pt' if row['variant']=='native' else (ROOT/row['checkpoint']).parent/'final_predictions.pt'
        pred=torch.load(path,weights_only=True);metrics,classes,logits=score(pred['logits'],pred['labels'],'evaluation',row['selected']['gamma'])
        torch.save(dict(logits=logits,labels=pred['labels'],predictions=classes),path.with_name('calibrated_'+path.name))
        final.append(dict(variant=row['variant'],seed=row['seed'],gamma=row['selected']['gamma'],metrics=metrics))
    write_json(exp.REPORT/'calibration.json',dict(selection=selection,final=final))
    print(json.dumps([(r['variant'],r['seed'],r['gamma'],round(r['metrics']['H'],3)) for r in final]))


if __name__=='__main__':main()
