"""Factorial sparse/dense tokens and CE/contrastive loss, plus matched controls."""
import argparse,json,math,time
import torch
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal import cub_token_experiment as runner
from multimodal.cub_rich_data import text_bank,load_data
from multimodal.cub_sparse import SparseTokens
from multimodal.grounded_model import multi_positive_contrastive

original_provenance=runner.provenance


def provenance():
    result=original_provenance()
    for path in ['configs/cub_sparse_v1.json','multimodal/cub_sparse.py','cub_sparse_experiment.py','multimodal/cub_rich_data.py','multimodal/grounded_model.py','runs/cub_coverage_v1/definition.json']:
        result[path]=digest(ROOT/path)
    return result


def build(variant,seed,bank,probe,c):
    seed_all(seed);return SparseTokens(bank,probe['state'],probe['mean_probability'],variant,c)


def fixed_probe(data,bank,c):
    saved=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True)
    import cub_rich_experiment
    assert saved['provenance']==cub_rich_experiment.provenance()
    saved['provenance']=provenance();torch.save(saved,runner.OUT/'probe.pt');return saved


def train_cell(variant,seed,lr,data,p,c,bank,probe):
    folder=runner.OUT/runner.name(variant,seed,lr);folder.mkdir(exist_ok=True)
    config=dict(variant=variant,seed=seed,lr=lr,protocol=c,provenance=provenance(),probe_sha256=digest(runner.OUT/'probe.pt'))
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text());assert result['config']==config;return result
    model=build(variant,seed,bank,probe,c).cuda();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
    ti=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='train_seen']);assert set(data['labels'][ti].tolist())==set(p['seen_classes'])
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith('probe.') or k in ['class_text','attribute_text']}
    gen=torch.Generator().manual_seed(seed+191);best=None;history=[];start=time.time()
    for step in range(c['updates']+1):
        if step in c['validation_steps']:
            metrics,_=runner.evaluate(model,data,p,'development');history.append(dict(step=step,metrics=metrics))
            if best is None or runner.rank(metrics)>runner.rank(best):
                best,best_step=metrics,step;torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},config=config,step=step),folder/'best.pt')
        if step==c['updates']:break
        model.train();b=runner.subset(data,ti[torch.randperm(len(ti),generator=gen)[:c['batch_size']]],'cuda');o=model(b['native_views'],b['valid'])
        seen=torch.tensor(p['seen_classes'],device='cuda');labels=(b['labels'][:,None]==seen).long().argmax(-1)
        ce=F.cross_entropy(o['logits'][:,seen],labels)
        if variant in ['dense_ce','sparse_ce']:classification=ce
        else:classification=.5*ce+.5*multi_positive_contrastive(o['embedding'],model.class_text[b['labels']],b['labels'],20.)
        loss=classification+c['preserve_weight']*(1-(o['embedding']*o['global_embedding']).sum(-1)).mean()
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
        for g in opt.param_groups:g['lr']=lr*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/c['updates'])))
    for k,v in frozen.items():assert torch.equal(v,model.state_dict()[k])
    result=dict(config=config,best_step=best_step,development=best,seconds=time.time()-start,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_assets_unchanged=True)
    write_json(folder/'result.json',result);write_json(folder/'history.json',history);print(f'{folder.name}: dev H={best["H"]:.2f}, step={best_step}',flush=True);return result


def configure():
    # Import before replacing runner.provenance, so the rich version's provenance
    # wrapper retains the immutable base function rather than this wrapper.
    import cub_rich_experiment
    runner.VERSION='cub_sparse_v1';runner.OUT=ROOT/'runs'/runner.VERSION;runner.REPORT=ROOT/'reports'/runner.VERSION;runner.CONFIG=ROOT/'configs'/f'{runner.VERSION}.json'
    runner.provenance=provenance;runner.build=build;runner.load_data=load_data;runner.text_bank=text_bank;runner.train_probe=fixed_probe;runner.train_cell=train_cell


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','all']);a=parser.parse_args();configure()
    if a.stage in ['train','all']:runner.train_all()
    if a.stage in ['evaluate','all']:runner.final_evaluation()
