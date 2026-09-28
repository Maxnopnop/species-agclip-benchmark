"""Cheap CPU linear attribute probe: representation learnability, not species AG."""
import argparse,json
import torch
from torch import nn
from data_tools import ROOT,write_json,digest
from multimodal.cub_data import prepare,VERSION,OUT,REPORT,text_bank,pixels
from multimodal.cub_model import masked_bce
from multimodal.visible_train import attribute_metrics

def run(stage):
    torch.set_num_threads(4);torch.manual_seed(42);p,m=prepare();bank=text_bank()
    if stage=='train':
        data=torch.load(ROOT/f'cache/{VERSION}/pixels_development.pt',weights_only=True);ti=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen'];vi=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen'];x=data['native_views'][:,0];y=data['targets'][:,0];results=[];best=None
        for lr in [.01,.001]:
            model=nn.Linear(512,y.shape[1]);model.weight.data.copy_(10*(bank['attribute_text']-bank['negative_text']));model.bias.data.zero_();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
            for step in range(601):
                if step in [0,100,300,600]:
                    with torch.no_grad():metrics=attribute_metrics(y[vi],model(x[vi]).sigmoid())
                    row=dict(lr=lr,step=step,development_map=metrics['attribute_map']);results.append(row)
                    if best is None or row['development_map']>best['development_map']:
                        best=row;torch.save(dict(state=model.state_dict(),selection=row,attributes=len(m['attributes']),source_sha256=digest(__file__)),OUT/'attribute_probe.pt')
                if step==600:break
                opt.zero_grad();loss=masked_bce(model(x[ti]),y[ti]);loss.backward();opt.step()
        write_json(OUT/'probe_locked.json',dict(best=best,source_sha256=digest(__file__),selection='Development attribute mAP only; no species classifier and no held-out image supervision.'))
        write_json(REPORT/'attribute_probe_development.json',dict(best=best,history=results));print(json.dumps(results,indent=2))
    else:
        lock=json.loads((OUT/'probe_locked.json').read_text(encoding='utf-8'));assert lock['source_sha256']==digest(__file__);data=pixels('evaluation');saved=torch.load(OUT/'attribute_probe.pt',weights_only=True);model=nn.Linear(512,saved['attributes']);model.load_state_dict(saved['state']);x=data['native_views'][:,0];y=data['targets'][:,0]
        with torch.no_grad():baseline=(10*x@(bank['attribute_text']-bank['negative_text']).T).sigmoid();pred=model(x).sigmoid()
        report=dict(selection=lock['best'],native=attribute_metrics(y,baseline),linear_probe=attribute_metrics(y,pred),scope='Frozen original CLIP features; image-level attribute labels only from 200 seen-class training photos. Evaluated after locked selections. Not an AG species classification result.')
        write_json(REPORT/'attribute_probe_final.json',report);print(json.dumps({k:v.get('attribute_map') for k,v in report.items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate']);run(parser.parse_args().stage)
