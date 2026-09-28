"""GPU correctness and supervision isolation checks for the CUB pilot."""
import json,gc,time
import torch
from data_tools import ROOT,write_json
from multimodal.cub_data import prepare,pixels,text_bank,REPORT
from multimodal.cub_experiment import setup,shuffle_targets,batch,provenance
from multimodal.cub_model import build,objective,cached_step,masked_bce
from multimodal.grounded_experiment import optimizer_for

def main():
    setup();p,m=prepare();data=pixels('development');shuffle_targets(data,p);model,_=build(text_bank(),'gold_region',p);model=model.cuda().train()
    assert len(m['rows'])==len({r['sha256'] for r in m['rows']})==700
    s,d,e=[set(p[k]) for k in ['seen_classes','dev_unseen_classes','eval_unseen_classes']];assert not (s&d or s&e or d&e)
    ids=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen' and data['valid'][i].any() and (data['targets'][i,1:]>=0).any()][:4];assert len(ids)==4;b=batch(data,ids)
    started=time.time();torch.cuda.reset_peak_memory_stats();out=[]
    for start in range(0,4,2):
        with torch.autocast('cuda',dtype=torch.bfloat16):out.append(model(b['images'][start:start+2].cuda(),b['valid'][start:start+2].cuda()))
    combined={k:torch.cat([o[k] for o in out]) for k in out[0]};loss,_=objective(combined,model,{k:v.cuda() for k,v in b.items() if k not in ['images','valid']},p);loss.backward();torch.nn.utils.clip_grad_norm_([t for t in model.parameters() if t.requires_grad],1.)
    reference={n:t.grad.detach().cpu().clone() for n,t in model.named_parameters() if t.grad is not None};del out,combined,loss;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    values=cached_step(model,b,optimizer_for(model,0.,p),p);errors=[]
    for n,t in model.named_parameters():
        if t.grad is not None:torch.testing.assert_close(t.grad.cpu(),reference[n],rtol=1e-4,atol=1e-6);errors.append(float((t.grad.cpu()-reference[n]).abs().max()))
    assert all(v>0 for v in values['gradient_norms'].values())
    # No annotations appear in forward; changing loss labels cannot alter it.
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model(b['images'][:2].cuda(),b['valid'][:2].cuda());z=model(b['images'][:2].cuda(),torch.zeros_like(b['valid'][:2]).cuda())
    assert torch.equal(z['embedding'],z['global_embedding'])
    unknown=torch.randn(2,5,requires_grad=True);loss=masked_bce(unknown,torch.full_like(unknown,-1));loss.backward();assert float(loss.detach())==0 and torch.equal(unknown.grad,torch.zeros_like(unknown))
    bad={k:v[:2].cuda() for k,v in b.items() if k not in ['images','valid']};bad['labels']=torch.tensor(p['eval_unseen_classes'][:2],device='cuda');rejected=False
    try:objective(a,model,bad,p)
    except ValueError:rejected=True
    assert rejected
    ti=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen'];gold=data['targets'][ti];shuffled=data['shuffled_targets'][ti]
    assert torch.equal(gold>=0,shuffled>=0) and torch.equal((gold==1).sum(0),(shuffled==1).sum(0)) and (gold!=shuffled).any()
    before={n:t.detach().cpu().clone() for n,t in model.named_parameters() if n in ['global_visual.proj','attribute_visual.proj','attr_head.weight','caf.linear1.weight']};frozen=model.global_visual.conv1.weight.detach().cpu().clone();model.train()
    update=cached_step(model,batch(data,ti[:16]),optimizer_for(model,1e-5,p),p);changes={n:float((dict(model.named_parameters())[n].detach().cpu()-x).abs().max()) for n,x in before.items()}
    assert all(x>0 for x in changes.values()) and torch.equal(frozen,model.global_visual.conv1.weight.detach().cpu())
    result=dict(status='passed',provenance=provenance(),gradient_cache_max_error=max(errors),gradient_norms=values['gradient_norms'],parameter_changes=changes,unknown_targets_zero_gradient=True,unseen_training_rejected=True,shuffled_mask_and_marginals_preserved=True,no_region_exact_fallback=True,class_and_image_isolation=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.time()-started,counts={role:sum(r['role']==role for r in m['rows']) for role in ['train_seen','dev_seen','dev_unseen','eval_seen','eval_unseen']},attributes=len(m['attributes']))
    write_json(REPORT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
