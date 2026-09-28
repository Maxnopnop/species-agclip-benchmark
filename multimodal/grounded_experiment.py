"""Locked, image-based starter experiment. No task-trained feature cache is used."""
import gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all
from .grounded_data import VERSION,prepare,image_tensors,ground
from .grounded_model import build_model,gradient_cached_step

OUT=ROOT/f'runs/{VERSION}'
REPORT=ROOT/f'reports/{VERSION}'

def setup():
    seed_all(42)
    # Use the same attention path under training/no-grad/BF16 inference.
    torch.backends.mha.set_fastpath_enabled(False)

def provenance():
    files=['configs/grounded_v1.json','multimodal/grounded_data.py','multimodal/grounded_model.py','multimodal/grounded_experiment.py','multimodal/models.py']
    return {f:digest(ROOT/f) for f in files}

def bank():
    result=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    assert result['attributes_sha256']==digest(ROOT/'configs/expanded_attributes.json')
    return result

def dataset(stage,transform):
    p,m=prepare();regions_path=ROOT/f'cache/{VERSION}/regions_{stage}.json'
    regions=json.loads(regions_path.read_text(encoding='utf-8'))
    assert regions['metadata']['implementation_sha256']==digest(ROOT/'multimodal/grounded_data.py')
    roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen']
    rows=[r for r in m['rows'] if r['role'] in roles]
    metadata=dict(regions_sha256=digest(regions_path),manifest_sha256=digest(ROOT/f'data/{VERSION}/manifest.json'),preprocess='native CLIP RGB bicubic shortest-side resize/center-crop/normalize 224; deterministic')
    cache_path=ROOT/f'cache/{VERSION}/pixels_{stage}.pt'
    if cache_path.exists():
        result=torch.load(cache_path,weights_only=True)
        assert result['metadata']==metadata
        return result
    original=torch.load(ROOT/'cache/expanded20/clip_vit_b32_features.pt',weights_only=True)
    assert original['metadata']['pretrained'] and original['metadata']['frozen']
    assert original['metadata']['manifest_sha256']==p['manifest_sha256']
    index={r['path']:i for i,r in enumerate(original['rows'])}
    views=[];ids=[];confidence=[];valid=[];teacher=[]
    for r in rows:
        assert digest(Path(m['image_root'])/r['path'])==r['sha256']
        v,a,c,ok=image_tensors(Path(m['image_root'])/r['path'],regions['images'][r['path']],transform)
        views.append(v);ids.append(a);confidence.append(c);valid.append(ok)
        i=index[r['path']];assert original['rows'][i]['sha256']==r['sha256']
        teacher.append(original['global_features'][i])
    result=dict(metadata=metadata,rows=rows,images=torch.stack(views),attribute_ids=torch.stack(ids),confidence=torch.stack(confidence),valid=torch.stack(valid),teacher=torch.stack(teacher),labels=torch.tensor([r['label'] for r in rows]))
    torch.save(result,cache_path);return result

def batch(data,indices):
    return {k:data[k][indices] for k in ['images','attribute_ids','confidence','valid','teacher','labels']}

def measures(logits,labels,p,stage):
    unseen=p['dev_unseen_classes'] if stage=='development' else p['eval_unseen_classes']
    seen=p['seen_classes'];candidates=seen+unseen
    candidate_tensor=torch.tensor(candidates);pred=candidate_tensor[logits[:,candidates].argmax(-1)]
    per={str(c):float((pred[labels==c]==c).float().mean())*100 for c in candidates}
    s=sum(per[str(c)] for c in seen)/len(seen);u=sum(per[str(c)] for c in unseen)/len(unseen);h=2*s*u/(s+u) if s+u else 0.
    unseen_mask=torch.isin(labels,torch.tensor(unseen));z=torch.tensor(unseen)[logits[unseen_mask][:,unseen].argmax(-1)]
    zsl=sum(float((z[labels[unseen_mask]==c]==c).float().mean())*100 for c in unseen)/len(unseen)
    target=(labels[:,None]==candidate_tensor).long().argmax(-1)
    return dict(S=s,U=u,H=h,ZSL=zsl,ce=float(F.cross_entropy(logits[:,candidates],target)),per_class=per,images=len(labels)),pred

@torch.no_grad()
def evaluate(model,data,p,stage,intervention=None):
    model.eval();indices=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen'];embeddings=[]
    for start in range(0,len(indices),p['micro_batch']):
        b=batch(data,indices[start:start+p['micro_batch']])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            e=model(b['images'].cuda(),b['attribute_ids'].cuda(),b['valid'].cuda(),intervention)['embedding']
        embeddings.append(e.cpu())
    logits=model.logit_scale.exp().clamp(max=100).cpu()*torch.cat(embeddings)@model.class_text.cpu().T
    labels=data['labels'][indices];metrics,pred=measures(logits,labels,p,stage)
    return metrics,dict(logits=logits,labels=labels,predictions=pred,paths=[data['rows'][i]['path'] for i in indices])

def rank(result):return (result['H'],(result['U']+result['S'])/2,-result['ce'])

def save_checkpoint(path,model,config,step,metrics):
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    buffers={n for n,_ in model.named_buffers()}
    state={n:t.detach().cpu().clone() for n,t in model.state_dict().items() if n in trainable or n in buffers}
    torch.save(dict(state=state,trainable_names=sorted(trainable),config=config,step=step,metrics=metrics),path)

def restore(path):
    saved=torch.load(path,weights_only=True)
    assert saved['config']['provenance']==provenance(),'Checkpoint/source provenance mismatch'
    c=saved['config'];seed_all(c['seed']);model,transform=build_model(bank(),c['variant'],c['protocol'],c['seed'])
    missing,unexpected=model.load_state_dict(saved['state'],strict=False)
    assert not unexpected and not set(missing)&set(saved['trainable_names'])
    assert set(saved['trainable_names'])=={n for n,p in model.named_parameters() if p.requires_grad}
    return model.cuda(),transform,saved

def optimizer_for(model,lr,p):
    visual=[];other=[]
    for n,t in model.named_parameters():
        if t.requires_grad:(visual if n.startswith(('global_visual','attribute_visual')) else other).append(t)
    return torch.optim.AdamW([dict(params=visual,lr=lr),dict(params=other,lr=lr*p['fusion_lr_multiplier'])],weight_decay=.01)

def cell_name(variant,lr,seed):return f'{variant}_seed{seed}_lr{lr:g}'

def train_cell(variant,lr,seed,p,data):
    folder=OUT/cell_name(variant,lr,seed);folder.mkdir(parents=True,exist_ok=True)
    config=dict(variant=variant,lr=lr,seed=seed,protocol=p,provenance=provenance())
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'));assert result['config']==config;return result
    seed_all(seed);model,_=build_model(bank(),variant,p,seed);model=model.cuda();optimizer=optimizer_for(model,lr,p)
    torch.cuda.reset_peak_memory_stats();start=time.time();history=[];best=None;best_step=None
    train_ids=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='train_seen']);gen=torch.Generator().manual_seed(seed+191)
    frozen=model.global_visual.conv1.weight.detach().cpu().clone()
    for step in range(p['updates']+1):
        if step in p['validation_steps']:
            metrics,_=evaluate(model,data,p,'development');history.append(dict(step=step,development=metrics))
            if best is None or rank(metrics)>rank(best):
                best,best_step=metrics,step;save_checkpoint(folder/'best.pt',model,config,step,metrics)
            print(f'{folder.name} step {step}: H={metrics["H"]:.2f} U={metrics["U"]:.2f} S={metrics["S"]:.2f}',flush=True)
            write_json(folder/'history.json',history)
        if step==p['updates']:break
        for i,g in enumerate(optimizer.param_groups):g['lr']=lr*(p['fusion_lr_multiplier'] if i else 1)*(.1+.9*.5*(1+math.cos(math.pi*step/p['updates'])))
        ids=train_ids[torch.randperm(len(train_ids),generator=gen)[:p['effective_batch']]]
        model.train();values=gradient_cached_step(model,batch(data,ids),optimizer,p)
        if step==0 or (step+1)%10==0:
            history.append(dict(step=step+1,training=values));print(f'{folder.name} update {step+1}/{p["updates"]}: loss={values["loss"]:.4f}',flush=True)
    assert torch.equal(frozen,model.global_visual.conv1.weight.detach().cpu())
    result=dict(config=config,best_step=best_step,development=best,selected_untrained=best_step==0,seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,trainable_parameters=sum(t.numel() for t in model.parameters() if t.requires_grad),frozen_early_layer_unchanged=True)
    write_json(folder/'result.json',result);del model,optimizer;gc.collect();torch.cuda.empty_cache();return result

def train_all():
    setup();p,_=prepare();_,transform=build_model(bank(),'finetune',p);data=dataset('development',transform)
    # Avoid holding an unused model while running the matrix.
    del _;gc.collect();torch.cuda.empty_cache()
    results=[]
    for variant in p['variants']:
        for seed in p['seeds']:
            for lr in p['learning_rates']:results.append(train_cell(variant,lr,seed,p,data))
    selected=[]
    for variant in p['variants']:
        for seed in p['seeds']:
            candidates=[r for r in results if r['config']['variant']==variant and r['config']['seed']==seed]
            r=max(candidates,key=lambda x:rank(x['development']));c=r['config']
            selected.append(dict(variant=variant,seed=seed,lr=c['lr'],best_step=r['best_step'],development=r['development'],checkpoint=str((OUT/cell_name(variant,c['lr'],seed)/'best.pt').relative_to(ROOT))))
    lock=dict(provenance=provenance(),selected=selected,created_unix=time.time(),final_images_not_used_for_selection=True)
    target=OUT/'selection_locked.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'));assert previous['provenance']==lock['provenance'] and previous['selected']==lock['selected']
    else:write_json(target,lock)
    print('All 8 development runs complete; selection locked.',flush=True)

def final_evaluation():
    setup();p,_=prepare();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));assert lock['provenance']==provenance()
    ground('evaluation');model,transform=build_model(bank(),'finetune',p);data=dataset('evaluation',transform);model=model.cuda()
    native,predictions=evaluate(model,data,p,'evaluation');torch.save(predictions,OUT/'native_predictions.pt');results=[dict(variant='native',best_step=0,metrics=native)]
    del model;gc.collect();torch.cuda.empty_cache()
    for chosen in lock['selected']:
        model,_,saved=restore(ROOT/chosen['checkpoint']);metrics,predictions=evaluate(model,data,p,'evaluation')
        folder=(ROOT/chosen['checkpoint']).parent;torch.save(predictions,folder/'final_predictions.pt')
        interventions={}
        if chosen['variant']=='ag':
            for intervention in ['zero_text','permuted']:interventions[intervention]=evaluate(model,data,p,'evaluation',intervention)[0]
        results.append(dict(**chosen,metrics=metrics,interventions=interventions))
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(REPORT/'final_results.json',dict(provenance=provenance(),selection_lock_sha256=digest(OUT/'selection_locked.json'),results=results))
    print(json.dumps(results,indent=2),flush=True)
