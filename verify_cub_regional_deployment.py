import json
from pathlib import Path
import torch
from data_tools import ROOT,write_json
from cub_regional_experiment import OUT,REPORT,prepare
from predict_cub_regional import predict


def main():
    p,m=prepare();lock=json.loads((OUT/'selection_locked.json').read_text());chosen=next(r for r in lock['selected'] if r['variant']=='tokens' and r['seed']==42)
    checkpoint=ROOT/chosen['checkpoint'];cached=torch.load(checkpoint.parent/'final_predictions.pt',weights_only=True);checks=[]
    for i in [0,len(cached['labels'])-1]:
        result,logits,prob=predict(checkpoint,Path(m['image_root'])/cached['paths'][i],p['seen_classes']+p['eval_unseen_classes'])
        torch.testing.assert_close(logits,cached['logits'][i],rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(prob,cached['probability'][i],rtol=1e-4,atol=1e-4)
        assert result['predictions'][0]['label']==int(cached['predictions'][i])
        checks.append(dict(index=i,max_logit_error=float((logits-cached['logits'][i]).abs().max()),max_probability_error=float((prob-cached['probability'][i]).abs().max()),prediction_matches=True))
    write_json(REPORT/'deployment_verification.json',dict(status='passed',seed=42,step=chosen['step'],checks=checks));print(json.dumps(checks))


if __name__=='__main__':main()
