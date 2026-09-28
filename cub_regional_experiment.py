"""Partial regional visual fine-tuning with immutable global recognition branch."""
import argparse,gc,json,math,time
import torch
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal.cub_data import prepare,pixels
from multimodal.cub_rich_data import text_bank,load_data as rich_data
from multimodal.cub_regional_tokens import build,cached_step
from multimodal.grounded_experiment import measures,rank
from multimodal.cub_token_experiment import attribute_score

VERSION='cub_regional_tokens_v1';OUT=ROOT/'runs'/VERSION;REPORT=ROOT/'reports'/VERSION


def setup():
    seed_all(42);torch.backends.mha.set_fastpath_enabled(False);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    return json.loads((ROOT/f'configs/{VERSION}.json').read_text())


def provenance():
    paths=['multimodal/cub_regional_tokens.py','cub_regional_experiment.py','multimodal/cub_tokens.py','multimodal/cub_model.py','multimodal/grounded_model.py','multimodal/grounded_experiment.py','multimodal/models.py','multimodal/cub_rich_data.py','multimodal/cub_data.py','multimodal/visible_train.py',f'configs/{VERSION}.json','runs/cub_rich_v1/probe.pt','cache/cub_rich_v1/text.pt','data/cub_attributes_v1/manifest.json']
    return {path:digest(ROOT/path) for path in paths}


def load_data(stage):
    data=rich_data(stage);raw=pixels(stage);assert [r['path'] for r in data['rows']]==[r['path'] for r in raw['rows']]
    data['images']=raw['images'];return data


def batch(data,ids):return {k:data[k][ids] for k in ['images','native_views','valid','targets','labels']}


@torch.no_grad()
def evaluate(model,data,p,stage,c,intervention=None):
    model.eval();ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen'];out=[]
    for start in range(0,len(ids),c['micro_batch']):
        ix=ids[start:start+c['micro_batch']];regions=model.encode_regions(data['images'][ix].cuda())
        o=model.from_regions(regions,data['native_views'][ix,0].cuda(),data['valid'][ix].cuda(),intervention)
        out.append({k:o[k].cpu() for k in ['logits','native_logits','probability','regions','embedding']})
    outputs={k:torch.cat([o[k] for o in out]) for k in out[0]};metrics,pred=measures(outputs['logits'],data['labels'][ids],p,stage)
    metrics.update(attribute_score(data['targets'][ids],outputs['probability']))
    return metrics,dict(**outputs,labels=data['labels'][ids],predictions=pred,paths=[data['rows'][i]['path'] for i in ids])


def restore(path):
    c=setup();saved=torch.load(path,weights_only=True);assert saved['config']['provenance']==provenance()
    seed_all(saved['config']['seed']);probe=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True);model,transform=build(text_bank(),probe,saved['config']['variant'],c)
    missing,unexpected=model.load_state_dict(saved['state'],strict=False);assert not unexpected and not set(missing)&set(saved['trainable_names'])
    return model.cuda().eval(),transform,saved


def cell_name(v,s,lr):return f'{v}_seed{s}_lr{lr:g}'


def train_cell(v,s,lr,data,p,c):
    folder=OUT/cell_name(v,s,lr);folder.mkdir(exist_ok=True);config=dict(variant=v,seed=s,lr=lr,protocol=c,provenance=provenance())
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text());assert r['config']==config;return r
    seed_all(s);probe=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True);model,_=build(text_bank(),probe,v,c);model=model.cuda()
    visual=[t for n,t in model.named_parameters() if t.requires_grad and n.startswith('attribute_visual')];fusion=[t for n,t in model.named_parameters() if t.requires_grad and n.startswith('fusion')]
    opt=torch.optim.AdamW([dict(params=visual,lr=lr),dict(params=fusion,lr=lr*c['fusion_lr_multiplier'])],weight_decay=.01)
    ti=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='train_seen']);gen=torch.Generator().manual_seed(s+191);best=None;history=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    frozen=model.attribute_visual.conv1.weight.detach().clone();probe_before={k:v.clone() for k,v in model.fusion.probe.state_dict().items()}
    for step in range(c['updates']+1):
        if step in c['validation_steps']:
            metrics,_=evaluate(model,data,p,'development',c);history.append(dict(step=step,metrics=metrics))
            if best is None or rank(metrics)>rank(best):
                best,best_step=metrics,step;trainable={n for n,t in model.named_parameters() if t.requires_grad};keep=trainable|{n for n,_ in model.named_buffers()}|{n for n in model.state_dict() if n.startswith('fusion.probe.')}
                torch.save(dict(state={n:t.detach().cpu().clone() for n,t in model.state_dict().items() if n in keep},trainable_names=sorted(trainable),config=config,step=step),folder/'best.pt')
            print(f'{folder.name} step{step}: dev H={metrics["H"]:.2f}, region mAP={100*metrics["regional_map"]:.2f}',flush=True)
        if step==c['updates']:break
        ix=ti[torch.randperm(len(ti),generator=gen)[:c['batch_size']]];model.train();values=cached_step(model,batch(data,ix),opt,p,c)
        if step==0 or (step+1)%40==0:history.append(dict(step=step+1,training=values));write_json(folder/'history.json',history)
        for j,g in enumerate(opt.param_groups):g['lr']=lr*(c['fusion_lr_multiplier'] if j else 1)*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/c['updates'])))
    assert torch.equal(frozen,model.attribute_visual.conv1.weight)
    assert all(torch.equal(v,model.fusion.probe.state_dict()[k]) for k,v in probe_before.items())
    result=dict(config=config,best_step=best_step,development=best,seconds=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,frozen_early_visual_and_probe_unchanged=True)
    write_json(folder/'history.json',history);write_json(folder/'result.json',result);del model,opt;gc.collect();torch.cuda.empty_cache();return result


def train_all():
    c=setup();p,_=prepare();data=load_data('development');results=[train_cell(v,s,lr,data,p,c) for s in c['seeds'] for v in c['variants'] for lr in c['learning_rates']];selected=[]
    for s in c['seeds']:
        for v in c['variants']:
            r=max([r for r in results if r['config']['seed']==s and r['config']['variant']==v],key=lambda r:rank(r['development']));lr=r['config']['lr']
            selected.append(dict(variant=v,seed=s,lr=lr,step=r['best_step'],development=r['development'],checkpoint=str((OUT/cell_name(v,s,lr)/'best.pt').relative_to(ROOT))))
    lock=dict(selected=selected,provenance=provenance(),scope=c['scope']);path=OUT/'selection_locked.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:write_json(path,lock)
    print(f'{len(results)} cells complete, {len(selected)} selections locked.',flush=True)


def final_evaluation():
    c=setup();p,_=prepare();lock=json.loads((OUT/'selection_locked.json').read_text());assert lock['provenance']==provenance();data=load_data('evaluation')
    logits=20*data['native_views'][:,0]@text_bank()['class_text'].T;native,pred=measures(logits,data['labels'],p,'evaluation');torch.save(dict(logits=logits,labels=data['labels'],predictions=pred),OUT/'native_predictions.pt');results=[]
    for chosen in lock['selected']:
        model,_,_=restore(ROOT/chosen['checkpoint']);metrics,pred=evaluate(model,data,p,'evaluation',c);torch.save(pred,(ROOT/chosen['checkpoint']).parent/'final_predictions.pt');interventions={}
        if chosen['variant']=='tokens':
            for mode in ['zero_text','permuted','constant']:
                counter,counter_pred=evaluate(model,data,p,'evaluation',c,mode);interventions[mode]=dict(metrics=counter,changed_predictions=int((pred['predictions']!=counter_pred['predictions']).sum()))
        results.append(dict(**chosen,metrics=metrics,interventions=interventions));print(f'FINAL {chosen["variant"]} seed{chosen["seed"]}: H={metrics["H"]:.2f}, ZSL={metrics["ZSL"]:.2f}',flush=True);del model;gc.collect();torch.cuda.empty_cache()
    write_json(REPORT/'final_results.json',dict(native=native,results=results,provenance=provenance(),scope=c['scope'],selection_sha256=digest(OUT/'selection_locked.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','all']);a=parser.parse_args()
    if a.stage in ['train','all']:train_all()
    if a.stage in ['evaluate','all']:final_evaluation()
