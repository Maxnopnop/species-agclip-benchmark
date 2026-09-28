import argparse,json
from pathlib import Path
import torch
from data_tools import ROOT,write_json
from predict_cub_tokens import configure,predict
from multimodal import cub_token_experiment as exp


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='cub_tokens_v1');parser.add_argument('--seed',type=int,default=42);a=parser.parse_args();configure(a.version)
    p,m=exp.prepare();lock=json.loads((exp.OUT/'selection_locked.json').read_text());chosen=next(r for r in lock['selected'] if r['variant']=='tokens' and r['seed']==a.seed)
    checkpoint=ROOT/chosen['checkpoint'];cached=torch.load(checkpoint.parent/'final_predictions.pt',weights_only=True);checks=[]
    for i in [0,len(cached['labels'])-1]:
        result,logits,probabilities=predict(checkpoint,Path(m['image_root'])/cached['paths'][i],p['seen_classes']+p['eval_unseen_classes'])
        torch.testing.assert_close(logits,cached['logits'][i],rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(probabilities,cached['probability'][i],rtol=1e-4,atol=1e-4)
        assert result['predictions'][0]['label']==int(cached['predictions'][i])
        checks.append(dict(index=i,max_logit_error=float((logits-cached['logits'][i]).abs().max()),max_probability_error=float((probabilities-cached['probability'][i]).abs().max()),prediction_matches=True))
    report=dict(status='passed',version=a.version,seed=a.seed,step=chosen['step'],checks=checks)
    write_json(exp.REPORT/'deployment_verification.json',report);print(json.dumps(report))


if __name__=='__main__':main()
