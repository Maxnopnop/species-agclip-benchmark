import gc,json
import torch
from data_tools import ROOT,write_json
from cub_regional_experiment import setup,load_data,batch,prepare,REPORT,text_bank
from multimodal.cub_regional_tokens import build,objective,cached_step


def main():
    c=setup();p,_=prepare();d=load_data('development');probe=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True)
    model,_=build(text_bank(),probe,'tokens',c);model=model.cuda();model.fusion.gate.data.fill_(.1)
    ids=[i for i,r in enumerate(d['rows']) if r['role']=='train_seen'][:4];b=batch(d,ids);gpu={k:v.cuda() for k,v in b.items() if k!='images'}
    regions=torch.cat([model.encode_regions(b['images'][i:i+2].cuda()) for i in range(0,4,2)])
    out=model.from_regions(regions,gpu['native_views'][:,0],gpu['valid']);loss,_=objective(out,model,gpu,p,c);loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    reference={n:t.grad.detach().cpu().clone() for n,t in model.named_parameters() if t.grad is not None}
    del out,regions,loss;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    opt=torch.optim.AdamW([t for t in model.parameters() if t.requires_grad],lr=0.)
    result=cached_step(model,b,opt,p,c);errors=[]
    for n,t in model.named_parameters():
        if t.grad is not None:
            torch.testing.assert_close(t.grad.cpu(),reference[n],rtol=1e-4,atol=1e-6);errors.append(float((t.grad.cpu()-reference[n]).abs().max()))
    assert result['gradient_norms']['attribute_visual']>0 and result['gradient_norms']['fusion']>0
    assert all(t.grad is None for t in model.fusion.probe.parameters())
    with torch.no_grad():
        regions=model.encode_regions(b['images'][:2].cuda());out=model.from_regions(regions,gpu['native_views'][:2,0],gpu['valid'][:2]);expected=20*gpu['native_views'][:2,0]@model.fusion.class_text.T
        assert torch.equal(out['native_logits'],expected)
    bad={**gpu,'labels':torch.tensor([p['eval_unseen_classes'][0]]*4,device='cuda')};rejected=False
    try: objective(model.from_regions(torch.zeros(4,2,512,device='cuda'),gpu['native_views'][:,0],gpu['valid']),model,bad,p,c)
    except ValueError: rejected=True
    assert rejected
    write_json(REPORT/'verification.json',dict(status='passed',exact_cached_gradient_max_error=max(errors),global_logits_bitwise_unchanged=True,frozen_probe=True,heldout_labels_rejected=True,gradient_norms=result['gradient_norms']))
    print('PASS: exact gradient cache; frozen global logits and probe; nonzero regional/fusion gradients; held-out train labels rejected.')


if __name__=='__main__':main()
