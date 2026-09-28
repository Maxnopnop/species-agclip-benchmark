"""Real GPU smoke, gradient-cache equivalence, provenance and deployment tests."""
import argparse,gc,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from data_tools import ROOT,write_json
from run import seed_all
from multimodal.grounded_data import prepare,VERSION
from multimodal.grounded_model import build_model,objective,gradient_cached_step,multi_positive_contrastive
from multimodal.grounded_experiment import setup,bank,dataset,batch,optimizer_for,provenance,restore,OUT

def smoke():
    setup();p,m=prepare();seed_all(42);model,transform=build_model(bank(),'ag',p);data=dataset('development',transform)
    model=model.cuda().train();torch.cuda.reset_peak_memory_stats();started=time.time()
    candidates=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen' and data['valid'][i].any()]
    ids=[next(i for i in candidates if data['rows'][i]['label']==c) for c in p['seen_classes'][:4]];b=batch(data,ids)
    # Cache-gradient equivalence to a graph over exactly the same microbatches.
    outputs=[]
    for start in range(0,4,2):
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs.append(model(b['images'][start:start+2].cuda(),b['attribute_ids'][start:start+2].cuda(),b['valid'][start:start+2].cuda()))
    merged={k:torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    loss,_=objective(merged,model,*[b[k].cuda() for k in ['labels','attribute_ids','valid','confidence','teacher']],p);loss.backward()
    torch.nn.utils.clip_grad_norm_([t for t in model.parameters() if t.requires_grad],1.)
    reference={n:t.grad.detach().cpu().clone() for n,t in model.named_parameters() if t.grad is not None}
    del outputs,merged,loss;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    optimizer=optimizer_for(model,0.,p);cached=gradient_cached_step(model,b,optimizer,p)
    differences={n:float((t.grad.cpu()-reference[n]).abs().max()) for n,t in model.named_parameters() if t.grad is not None}
    for n,t in model.named_parameters():
        if t.grad is not None:torch.testing.assert_close(t.grad.cpu(),reference[n],rtol=1e-4,atol=1e-6)
    assert all(cached['gradient_norms'][key]>0 for key in ['global_visual','attribute_visual','caf','attribute_projection'])
    assert model.global_visual.conv1.weight.grad is None and model.attribute_visual.conv1.weight.grad is None
    # Original feature cache is only a teacher; validate against fresh raw-image output.
    model.eval()
    with torch.no_grad():
        fresh=F.normalize(model.global_visual(b['images'][:,0].cuda()).float(),dim=-1).cpu()
        teacher_error=float((fresh-b['teacher']).abs().max())
        # The old cache used batches of 16; batch-shaped GPU kernels need not be
        # bitwise identical. Bound both coordinate error and angular difference.
        teacher_cosine_min=float((fresh*b['teacher']).sum(-1).min())
        assert teacher_error<.001 and teacher_cosine_min>.99999
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fallback=model(b['images'][:2].cuda(),b['attribute_ids'][:2].cuda(),torch.zeros_like(b['valid'][:2]).cuda())
        assert torch.equal(fallback['embedding'],fallback['global_embedding'])
    # Seen-only objective must reject held-out classes.
    rejected=False
    try:objective(fallback,model,torch.tensor(p['eval_unseen_classes'][:2],device='cuda'),b['attribute_ids'][:2].cuda(),b['valid'][:2].cuda(),b['confidence'][:2].cuda(),b['teacher'][:2].cuda(),p)
    except ValueError:rejected=True
    assert rejected
    x=F.normalize(torch.randn(3,512),dim=-1);same=torch.zeros(3,dtype=torch.long)
    assert abs(float(multi_positive_contrastive(x,x,same,torch.tensor(10.))))<1e-6
    before={k:dict(model.named_parameters())[k].detach().cpu().clone() for k in ['global_visual.proj','attribute_visual.proj','caf.linear1.weight','attribute_projection.0.weight']}
    frozen=model.global_visual.conv1.weight.detach().cpu().clone();optimizer=optimizer_for(model,1e-5,p);model.train()
    real=gradient_cached_step(model,b,optimizer,p)
    changes={k:float((dict(model.named_parameters())[k].detach().cpu()-v).abs().max()) for k,v in before.items()}
    assert all(v>0 for v in changes.values()) and torch.equal(frozen,model.global_visual.conv1.weight.detach().cpu())
    # Exercise the deployed effective batch of 16, not just the four-image parity test.
    full=gradient_cached_step(model,batch(data,candidates[:16]),optimizer,p)
    result=dict(status='passed',provenance=provenance(),device=torch.cuda.get_device_name(),micro_batch=2,effective_batch=16,train_images=300,development_images=180,unique_total_images=len(m['rows']),gradient_cache_max_error=max(differences.values()),teacher_feature_max_error=teacher_error,teacher_cosine_min=teacher_cosine_min,actual_parameter_changes=changes,gradient_norms=real['gradient_norms'],full_batch_loss=full['loss'],replay_max_error=full['replay_max_error'],missing_region_exact_fallback=True,unseen_training_rejected=True,duplicate_classes_handled=True,frozen_early_layer_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-started)
    write_json(ROOT/f'reports/{VERSION}/smoke.json',result);print(json.dumps(result,indent=2))

def deployment():
    from predict_grounded import predict
    setup();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));p,m=prepare();records=[]
    for selected in lock['selected']:
        checkpoint=ROOT/selected['checkpoint'];cached=torch.load(checkpoint.parent/'final_predictions.pt',weights_only=True)
        result,logits=predict(checkpoint,Path(m['image_root'])/cached['paths'][0],candidates=p['seen_classes']+p['eval_unseen_classes'])
        error=float((logits-cached['logits'][0]).abs().max());torch.testing.assert_close(logits,cached['logits'][0],rtol=1e-3,atol=.003)
        assert result['predictions'][0]['label']==int(cached['predictions'][0])
        records.append(dict(variant=selected['variant'],max_logit_error=error,prediction_matches=True))
    write_json(ROOT/f'reports/{VERSION}/deployment_verification.json',dict(status='passed',provenance=provenance(),checks=records))
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','deployment'],default='smoke',nargs='?');args=parser.parse_args()
    smoke() if args.stage=='smoke' else deployment()
