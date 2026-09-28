"""Post-hoc sensitivity of trained, attribute-selected models; no further tuning."""
import gc,json
import torch
from torch.nn import functional as F
from data_tools import ROOT,write_json,digest
from multimodal.cub_data import prepare,pixels,OUT,REPORT
from multimodal.cub_experiment import setup,restore,evaluate,provenance

def main():
    setup();p,_=prepare();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));data=pixels('evaluation');records=[]
    for chosen in lock['attribute_selected']:
        if chosen['variant'] not in ['gold','gold_region']:continue
        model,_,_=restore(ROOT/chosen['checkpoint']);base,pred=evaluate(model,data,p,'evaluation');interventions={}
        for mode in ['zero_text','permuted']:
            metrics,other=evaluate(model,data,p,'evaluation',mode)
            interventions[mode]=dict(H=metrics['H'],ZSL=metrics['ZSL'],changed_gzsl_predictions=int((other['predictions']!=pred['predictions']).sum()),max_logit_difference=float((other['logits']-pred['logits']).abs().max()))
        probabilities=pred['attributes'][:,1:][data['valid']];weights=probabilities/probabilities.sum(-1,keepdim=True).clamp_min(1e-8);mixtures=F.normalize(weights@model.attribute_text.cpu(),dim=-1);n=len(mixtures)
        mean_cos=float((mixtures.sum(0).square().sum()-n)/(n*(n-1))) if n>1 else None
        records.append(dict(variant=chosen['variant'],selected_step=chosen['step'],checkpoint=chosen['checkpoint'],H=base['H'],attribute_map=base['attribute_map'],fusion_gate=float(model.gate.detach().tanh()),valid_regions=n,mean_pairwise_attribute_text_cosine=mean_cos,mean_attribute_probability_std=float(probabilities.std(0).mean()),interventions=interventions))
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(REPORT/'posthoc_diagnosis.json',dict(provenance=provenance(),diagnostic_source_sha256=digest(__file__),records=records,scope='Attribute-selected trained checkpoints; evaluation-only, no tuning. Text interventions do not erase learned visual features. High text-mixture similarity is a candidate bottleneck, not a causal proof.'))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
