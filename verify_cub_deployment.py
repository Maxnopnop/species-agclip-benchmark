import json,argparse
from pathlib import Path
import torch
from data_tools import ROOT,write_json
from multimodal.cub_data import prepare,OUT,REPORT
from multimodal.cub_experiment import provenance
from predict_cub import predict

def main(attributes_only=False):
    p,m=prepare();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));records=[]
    if attributes_only:
        previous=json.loads((REPORT/'deployment_verification.json').read_text(encoding='utf-8'));assert previous['status']=='passed' and previous['provenance']==provenance()
        records=[dict(r,selection='classification') for r in previous['checks'] if r.get('selection','classification')=='classification'];assert len(records)==6
    selected=[] if attributes_only else [(r,'classification') for r in lock['selected']]
    selected += [(r,'attribute') for r in lock['attribute_selected'] if r['variant'] in ['gold','gold_region']]
    for chosen,selection in selected:
        path=ROOT/chosen['checkpoint'];filename='final_predictions.pt' if selection=='classification' else 'attribute_final_predictions.pt';cached=torch.load(path.parent/filename,weights_only=True)
        result,logits,attr=predict(path,Path(m['image_root'])/cached['paths'][0],p['seen_classes']+p['eval_unseen_classes'])
        torch.testing.assert_close(logits,cached['logits'][0],rtol=1e-4,atol=1e-4);torch.testing.assert_close(attr,cached['attributes'][0,0],rtol=1e-4,atol=1e-4)
        assert result['predictions'][0]['label']==int(cached['predictions'][0])
        records.append(dict(variant=chosen['variant'],selection=selection,step=chosen['step'],max_logit_error=float((logits-cached['logits'][0]).abs().max()),max_attribute_error=float((attr-cached['attributes'][0,0]).abs().max()),prediction_matches=True))
    write_json(REPORT/'deployment_verification.json',dict(status='passed',provenance=provenance(),checks=records));print(json.dumps(records,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attributes-only',action='store_true');main(parser.parse_args().attributes_only)
