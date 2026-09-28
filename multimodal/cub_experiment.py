"""Controlled quality-of-supervision pilot with sealed class-disjoint evaluation."""
import gc,json,math,time
import torch
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all
from .cub_data import VERSION,OUT,REPORT,prepare,pixels,text_bank
from .cub_model import build,cached_step
from .grounded_experiment import measures,rank,optimizer_for
from .visible_train import attribute_metrics

def setup():seed_all(42);torch.backends.mha.set_fastpath_enabled(False)

def provenance():
    files=['multimodal/cub_data.py','multimodal/cub_model.py','multimodal/cub_experiment.py','multimodal/models.py','multimodal/grounded_model.py','multimodal/grounded_experiment.py','multimodal/visible_train.py',f'configs/{VERSION}.json']
    return {f:digest(ROOT/f) for f in files}

def shuffle_targets(data,p):
    result=data['targets'].clone();ids=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='train_seen']);gen=torch.Generator().manual_seed(1729)
    # Preserve known/unknown masks and per-attribute marginals, break image pairing.
    for v in range(3):
        for a in range(result.shape[-1]):
            known=ids[result[ids,v,a]>=0]
            result[known,v,a]=data['targets'][known[torch.randperm(len(known),generator=gen)],v,a]
    data['shuffled_targets']=result

def batch(data,ids):return {k:data[k][ids] for k in ['images','valid','targets','shuffled_targets','teacher','pseudo','labels']}

@torch.no_grad()
def evaluate(model,data,p,stage,intervention=None):
    model.eval();ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen'];embedding=[];attributes=[]
    for start in range(0,len(ids),p['micro_batch']):
        ix=ids[start:start+p['micro_batch']]
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(data['images'][ix].cuda(),data['valid'][ix].cuda(),intervention)
        embedding.append(o['embedding'].cpu());attributes.append(o['view_logits'].sigmoid().cpu())
    logits=model.logit_scale.exp().clamp(max=100).cpu()*torch.cat(embedding)@model.class_text.cpu().T;labels=data['labels'][ids];attr=torch.cat(attributes)
    metrics,pred=measures(logits,labels,p,stage);a=attribute_metrics(data['targets'][ids,0],attr[:,0]);regional=attribute_metrics(data['targets'][ids,1:].reshape(-1,attr.shape[-1]),attr[:,1:].reshape(-1,attr.shape[-1]))
    metrics.update({k:v for k,v in a.items() if k!='per_attribute'});metrics['regional_attribute_map']=regional['attribute_map'];metrics['regional_known_labels']=regional['known_attribute_labels']
    unseen=p['dev_unseen_classes'] if stage=='development' else p['eval_unseen_classes'];mask=torch.isin(labels,torch.tensor(unseen))
    metrics['unseen_attribute_map']=attribute_metrics(data['targets'][ids,0][mask],attr[:,0][mask])['attribute_map']
    return metrics,dict(logits=logits,labels=labels,predictions=pred,attributes=attr,targets=data['targets'][ids],paths=[data['rows'][i]['path'] for i in ids],per_attribute=a['per_attribute'])

def save(path,model,config,step,metrics):
    trainable={n for n,t in model.named_parameters() if t.requires_grad};buffers={n for n,_ in model.named_buffers()}
    # Frozen attribute readout is initialized deterministically from the text bank.
    state={n:t.detach().cpu().clone() for n,t in model.state_dict().items() if n in trainable or n in buffers}
    torch.save(dict(state=state,trainable_names=sorted(trainable),config=config,step=step,metrics=metrics),path)

def restore(path):
    saved=torch.load(path,weights_only=True);assert saved['config']['provenance']==provenance()
    c=saved['config'];seed_all(c['seed']);model,transform=build(text_bank(),c['variant'],c['protocol']);missing,unexpected=model.load_state_dict(saved['state'],strict=False)
    assert not unexpected and not set(missing)&set(saved['trainable_names'])
    assert set(saved['trainable_names'])=={n for n,t in model.named_parameters() if t.requires_grad}
    return model.cuda(),transform,saved

def cell_name(variant,lr):return f'{variant}_seed42_lr{lr:g}'

def train_cell(variant,lr,p,data):
    folder=OUT/cell_name(variant,lr);folder.mkdir(parents=True,exist_ok=True);config=dict(variant=variant,lr=lr,seed=42,protocol=p,provenance=provenance())
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'));assert result['config']==config;return result
    seed_all(42);model,_=build(text_bank(),variant,p);model=model.cuda();optimizer=optimizer_for(model,lr,p);gen=torch.Generator().manual_seed(233);ids=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='train_seen'])
    best=None;best_attr=None;history=[];start=time.time();torch.cuda.reset_peak_memory_stats();frozen=model.global_visual.conv1.weight.detach().cpu().clone()
    for step in range(p['updates']+1):
        if step in p['validation_steps']:
            metrics,_=evaluate(model,data,p,'development');history.append(dict(step=step,development=metrics))
            if best is None or rank(metrics)>rank(best):best=metrics;best_step=step;save(folder/'best.pt',model,config,step,metrics)
            if best_attr is None or metrics['attribute_map']>best_attr['attribute_map']:best_attr=metrics;best_attr_step=step;save(folder/'attribute_best.pt',model,config,step,metrics)
            print(f'{folder.name} step {step}: H={metrics["H"]:.2f}, attr mAP={metrics["attribute_map"]*100:.2f}',flush=True);write_json(folder/'history.json',history)
        if step==p['updates']:break
        for i,g in enumerate(optimizer.param_groups):g['lr']=lr*(p['fusion_lr_multiplier'] if i else 1)*(.1+.9*.5*(1+math.cos(math.pi*step/p['updates'])))
        selected=ids[torch.randperm(len(ids),generator=gen)[:p['effective_batch']]];model.train();values=cached_step(model,batch(data,selected),optimizer,p)
        if step==0 or (step+1)%20==0:history.append(dict(step=step+1,training=values));print(f'{folder.name} update {step+1}: loss={values["loss"]:.4f}',flush=True)
    assert torch.equal(frozen,model.global_visual.conv1.weight.detach().cpu())
    result=dict(config=config,best_step=best_step,development=best,attribute_best_step=best_attr_step,attribute_development=best_attr,seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,trainable_parameters=sum(t.numel() for t in model.parameters() if t.requires_grad),frozen_early_unchanged=True)
    write_json(folder/'result.json',result);del model,optimizer;gc.collect();torch.cuda.empty_cache();return result

def train_all():
    setup();p,_=prepare();data=pixels('development');shuffle_targets(data,p);results=[]
    for variant in p['variants']:
        for lr in p['learning_rates']:results.append(train_cell(variant,lr,p,data))
    selected=[];attr_selected=[]
    for variant in p['variants']:
        options=[r for r in results if r['config']['variant']==variant];chosen=max(options,key=lambda x:rank(x['development']));c=chosen['config']
        selected.append(dict(variant=variant,lr=c['lr'],step=chosen['best_step'],development=chosen['development'],checkpoint=str((OUT/cell_name(variant,c['lr'])/'best.pt').relative_to(ROOT))))
        chosen=max(options,key=lambda x:x['attribute_development']['attribute_map']);c=chosen['config']
        attr_selected.append(dict(variant=variant,lr=c['lr'],step=chosen['attribute_best_step'],development=chosen['attribute_development'],checkpoint=str((OUT/cell_name(variant,c['lr'])/'attribute_best.pt').relative_to(ROOT))))
    lock=dict(provenance=provenance(),selected=selected,attribute_selected=attr_selected,final_not_used=True)
    path=OUT/'selection_locked.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==lock
    else:write_json(path,lock)
    print('12 cells complete; classification and attribute selections locked.',flush=True)

def final_evaluation():
    setup();p,_=prepare();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));assert lock['provenance']==provenance();data=pixels('evaluation')
    seed_all(42);model,_=build(text_bank(),'finetune',p);model=model.cuda();metrics,pred=evaluate(model,data,p,'evaluation');torch.save(pred,OUT/'native_predictions.pt');results=[dict(variant='native',step=0,metrics=metrics)]
    del model;gc.collect();torch.cuda.empty_cache()
    for chosen in lock['selected']:
        model,_,saved=restore(ROOT/chosen['checkpoint']);metrics,pred=evaluate(model,data,p,'evaluation');folder=(ROOT/chosen['checkpoint']).parent;torch.save(pred,folder/'final_predictions.pt');interventions={}
        if chosen['variant'] in ['gold','gold_region']:
            for intervention in ['zero_text','permuted']:interventions[intervention]=evaluate(model,data,p,'evaluation',intervention)[0]
        results.append(dict(**chosen,metrics=metrics,interventions=interventions));del model;gc.collect();torch.cuda.empty_cache()
    attr_results=[]
    for chosen in lock['attribute_selected']:
        model,_,_=restore(ROOT/chosen['checkpoint']);metrics,pred=evaluate(model,data,p,'evaluation');torch.save(pred,(ROOT/chosen['checkpoint']).parent/'attribute_final_predictions.pt');attr_results.append(dict(**chosen,metrics=metrics));del model;gc.collect();torch.cuda.empty_cache()
    write_json(REPORT/'final_results.json',dict(provenance=provenance(),selection_sha256=digest(OUT/'selection_locked.json'),classification_selected=results,attribute_selected=attr_results))
    print(json.dumps([(r['variant'],r['metrics']['H'],r['metrics']['attribute_map']) for r in results]),flush=True)
