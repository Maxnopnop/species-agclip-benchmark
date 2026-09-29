"""Frozen dense features, train-only part supervision, and explicit attribute readout.

No final-evaluation rows are loaded. The native SigLIP classifier is never updated.
"""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .grounded_experiment import measures
from .visible_train import attribute_metrics

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'attribute_path_v1'
CACHE = ROOT / 'cache' / VERSION
RUN = ROOT / 'runs' / VERSION
REPORT = ROOT / 'reports' / VERSION
MODEL = ROOT / 'cache/four_routes_v1/models/fgclip2_base'
os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
os.environ['HF_MODULES_CACHE'] = str(ROOT / 'cache/huggingface/modules')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def setup():
    torch.set_num_threads(4)
    for path in [CACHE, RUN, REPORT]:
        path.mkdir(parents=True, exist_ok=True)
    c = json.loads((ROOT / f'configs/{VERSION}.json').read_text())
    m = json.loads((ROOT / 'data/cub100_v1/manifest.json').read_text())
    p = json.loads((ROOT / 'configs/cub100_v1.json').read_text())
    return c, m, p


def source():
    return {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
        ROOT / f'configs/{VERSION}.json', ROOT / 'data/cub100_v1/manifest.json',
        MODEL / 'config.json', MODEL / 'modeling_fgclip2.py', MODEL / 'configuration_fgclip2.py']}


def build_cache(c, m):
    path = CACHE / 'dense.pt'
    if path.exists():
        d = torch.load(path, weights_only=True)
        assert d['source'] == source(), 'Cache source mismatch: use a new version.'
        return d
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
    from .cub100_data import prompt
    from prepare_four_routes import verified_model_file
    meta = json.loads((ROOT / 'reports/four_routes_v1/fgclip2_hf_metadata.json').read_text())
    weight = next(r for r in meta['siblings'] if r['rfilename'] == 'model.safetensors')
    assert verified_model_file(MODEL / 'model.safetensors', weight)
    rows = [r for r in m['rows'] if r['role'] in ['train_seen', 'dev_seen', 'dev_unseen']]
    assert len(rows) == 1000 and sum(r['role'] == 'train_seen' for r in rows) == 500
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    positive = [prompt(a['name']).lower() for a in m['attributes']]
    negative = [s.replace('a bird with', 'a bird without') for s in positive]
    texts = [f'the {name} of a bird' for name in c['part_groups']] + positive + negative
    encoded, patches, global_features, masks, shapes = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            tokens = tok(texts[i:i+16], padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
            encoded.append(F.normalize(model.get_text_features(**tokens, walk_type='box').float(), dim=-1).cpu())
        for i in range(0, len(rows), 4):
            images = []
            for r in rows[i:i+4]:
                file = Path(m['image_root']) / r['path']
                assert digest(file) == r['sha256']
                with Image.open(file) as im:
                    images.append(im.convert('RGB'))
            batch = processor(images=images, max_num_patches=c['max_patches'], return_tensors='pt').to('cuda')
            local = F.normalize(model.get_image_dense_feature(**batch).float(), dim=-1)
            whole = F.normalize(model.get_image_features(**batch).float(), dim=-1)
            assert torch.isfinite(local).all() and torch.isfinite(whole).all()
            patches.append(local.cpu().half()); global_features.append(whole.cpu())
            masks.append(batch['pixel_attention_mask'].bool().cpu()); shapes.append(batch['spatial_shapes'].cpu())
            if i % 100 == 0:
                print(f'dense features {i+len(images)}/{len(rows)}', flush=True)
    text = torch.cat(encoded); count = len(m['attributes'])
    raw = torch.load(ROOT / 'data/cub100_v1/raw_targets.pt', weights_only=True)['raw_targets']
    index = {r['image_id']: i for i, r in enumerate(m['rows'])}
    targets = raw[[index[r['image_id']] for r in rows]]
    profiles = np.loadtxt(ROOT / 'data/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt')
    profiles = torch.tensor(profiles[[x['source_id']-1 for x in m['classes']]][:,
                       [x['source_id']-1 for x in m['attributes']]], dtype=torch.float32) / 100
    assert profiles.shape == (100, count) and profiles.min() >= 0 and profiles.max() <= 1
    d = dict(source=source(), rows=rows, patches=torch.cat(patches), global_features=torch.cat(global_features),
             masks=torch.cat(masks), shapes=torch.cat(shapes), part_text=text[:4],
             positive_text=text[4:4+count], negative_text=text[4+count:],
             attribute_targets=targets, profiles=profiles,
             profile_sha256=digest(ROOT / 'data/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'),
             labels=torch.tensor([r['label'] for r in rows]))
    torch.save(d, path)
    del model; gc.collect(); torch.cuda.empty_cache()
    return d


def geometry(c, d):
    """Labels/coordinates live in evaluation and supervision data, never model forward."""
    n, r = d['masks'].shape
    xy = torch.zeros(n, r, 2)
    distance = torch.full((n, 4, r), float('inf'))
    visible = torch.zeros(n, 4, dtype=torch.bool)
    for i, row in enumerate(d['rows']):
        h, w = d['shapes'][i].tolist(); iw, ih = row['size']
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack([(xx.flatten()+.5)/w, (yy.flatten()+.5)/h], dim=-1)
        xy[i, :h*w] = grid
        assert int(d['masks'][i].sum()) == h*w
        for j, ids in enumerate(c['part_groups'].values()):
            points = [row['parts'][str(pid)][:2] for pid in ids if row['parts'][str(pid)][2]]
            if points:
                visible[i, j] = True
                distance[i, j, :h*w] = torch.cdist(grid*torch.tensor([iw, ih]),
                                       torch.tensor(points)).min(-1).values / (iw*iw+ih*ih)**.5
    target = (-.5*(distance/c['target_sigma_diagonal']).square()).exp()
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    return dict(xy=xy, distance=distance, visible=visible, target=target)


def train_position_prior(c, d, g, ti):
    """Fixed 16x16 mean training target, resized to each image; no visual input."""
    prior = torch.zeros(4, 16, 16); counts = torch.zeros(4, 1, 1)
    for i in ti.tolist():
        h, w = d['shapes'][i].tolist()
        target = g['target'][i, :, :h*w].reshape(4, 1, h, w)
        resized = F.interpolate(target, size=(16, 16), mode='bilinear', align_corners=False)[:, 0]
        resized /= resized.sum((1,2), keepdim=True).clamp_min(1e-12)
        prior += resized; counts += g['visible'][i, :, None, None]
    prior /= counts.clamp_min(1)
    maps = torch.zeros(len(d['rows']), 4, d['masks'].shape[1])
    for i, shape in enumerate(d['shapes'].tolist()):
        h, w = shape
        maps[i, :, :h*w] = F.interpolate(prior[:, None], size=(h, w), mode='bilinear', align_corners=False)[:, 0].flatten(1)
    return maps / maps.sum(-1, keepdim=True).clamp_min(1e-12)


class PartLocalizer(nn.Module):
    def __init__(self, text, scale=20.):
        super().__init__()
        self.query = nn.Parameter(text.clone())
        self.register_buffer('initial', text.clone())
        self.scale = scale

    def forward(self, patches, mask):
        logits = self.scale * patches @ F.normalize(self.query, dim=-1).T
        return logits.transpose(1,2).masked_fill(~mask[:, None], -1e4).softmax(-1)


def location_metrics(maps, g, ids, rows, radius):
    pred = maps[ids].argmax(-1)
    distances = g['distance'][ids].gather(-1, pred[..., None]).squeeze(-1)
    visible = g['visible'][ids]
    result = {}
    for subset in ['all', 'dev_seen', 'dev_unseen']:
        keep = torch.tensor([subset == 'all' or rows[i]['role'] == subset for i in ids.tolist()])
        per = []
        for j in range(4):
            ok = keep & visible[:, j]
            per.append(dict(n=int(ok.sum()), hit=float((distances[ok,j] <= radius).float().mean()),
                            distance=float(distances[ok,j].mean())))
        result[subset] = dict(parts=per, macro_hit=sum(x['hit'] for x in per)/4)
    result['wing_tail_same_peak'] = float((pred[:,1] == pred[:,3]).float().mean())
    return result


def localize(c, d, g, ti, vi):
    run_path = RUN / 'localizers.pt'
    if run_path.exists():
        saved = torch.load(run_path, weights_only=True); assert saved['source'] == source()
        return saved
    patch = F.normalize(d['patches'].float(), dim=-1).cuda(); mask = d['masks'].cuda()
    target = g['target'].cuda(); visible = g['visible'].cuda()
    prior = train_position_prior(c, d, g, ti)
    base = PartLocalizer(d['part_text'], c['localization_scale']).cuda()
    with torch.no_grad():
        native = torch.cat([base(patch[i:i+64], mask[i:i+64]).cpu() for i in range(0,len(patch),64)])
    maps = dict(native_part=native, position_prior=prior)
    metrics = {name: location_metrics(value, g, vi, d['rows'], c['pointing_radius_diagonal']) for name,value in maps.items()}
    states, learned, history = {}, {}, []
    for seed in c['seeds']:
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        model = PartLocalizer(d['part_text'], c['localization_scale']).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=c['localization_lr'], weight_decay=.01)
        for step in range(1, c['localization_steps']+1):
            ix = ti[torch.randint(len(ti),(c['localization_batch'],),generator=rng)].cuda()
            attention = model(patch[ix], mask[ix])
            ce = -(target[ix]*attention.clamp_min(1e-12).log()).sum(-1)
            loss = (ce*visible[ix]).sum()/visible[ix].sum().clamp_min(1)
            loss = loss + c['localization_anchor']*(F.normalize(model.query,dim=-1)-model.initial).square().mean()
            opt.zero_grad(); loss.backward()
            assert model.query.grad is not None and torch.isfinite(model.query.grad).all()
            opt.step()
            if step % 100 == 0:
                history.append(dict(seed=seed,step=step,train_loss=float(loss.detach())))
        with torch.no_grad():
            attention = torch.cat([model(patch[i:i+64],mask[i:i+64]).cpu() for i in range(0,len(patch),64)])
        learned[str(seed)] = attention
        states[str(seed)] = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        metrics[f'learned_{seed}'] = location_metrics(attention,g,vi,d['rows'],c['pointing_radius_diagonal'])
        print('localization',seed,{key: round(metrics[f'learned_{seed}'][key]['macro_hit'],4)
                                  for key in ['all','dev_seen','dev_unseen']},flush=True)
    saved = dict(source=source(),maps=maps,learned=learned,states=states,metrics=metrics,history=history)
    torch.save(saved,run_path)
    save_json(REPORT/'localization.json',dict(metrics=metrics,history=history,
        training_images=len(ti),development_images=len(vi),selection='fixed final step',
        annotation_scope='Only train_seen coordinates optimize queries. Development coordinates score predictions.'))
    del patch,mask,target,visible,model,base; gc.collect(); torch.cuda.empty_cache()
    return saved


def attribute_view_indices(m):
    indices = []
    for a in m['attributes']:
        field = a['name'].split('::')[0][4:]
        if any(field.startswith(k) for k in ['bill','head','forehead','crown','nape','eye','throat']): j=1
        elif field.startswith('wing'): j=2
        elif field.startswith('breast'): j=3
        elif 'tail' in field: j=4
        else: j=0
        indices.append(j)
    return torch.tensor(indices)


def pooled_views(d, attention):
    patch = F.normalize(d['patches'].float(),dim=-1)
    local = torch.einsum('npr,nrd->npd',attention,patch)
    return torch.cat([d['global_features'][:,None],F.normalize(local,dim=-1)],1)


class AttributeReadout(nn.Module):
    """There is no direct visual-to-class skip, trainable class head, or class bias."""
    def __init__(self, d, view_ids, scale):
        super().__init__()
        self.weight = nn.Parameter(20*(d['positive_text']-d['negative_text']))
        self.bias = nn.Parameter(torch.zeros(len(view_ids)))
        self.register_buffer('view_ids',view_ids.clone())
        self.register_buffer('profiles',F.normalize(d['profiles'],dim=-1))
        self.scale = scale

    def forward(self, views):
        evidence = torch.einsum('bad,ad->ba',views[:,self.view_ids],self.weight)+self.bias
        scores = self.scale*F.normalize(evidence,dim=-1)@self.profiles.T
        return scores,evidence


def classification(c,d,m,p,loc,ti,vi):
    output_path = REPORT/'classification.json'
    if output_path.exists():
        saved = json.loads(output_path.read_text()); assert saved['source'] == source()
        return saved
    idx = attribute_view_indices(m)
    uniform = d['masks'].float()[:,None].expand(-1,4,-1)
    uniform = uniform/uniform.sum(-1,keepdim=True)
    common = {name:pooled_views(d,value) for name,value in {**loc['maps'],'uniform':uniform}.items()}
    global_views = d['global_features'][:,None].expand(-1,5,-1).clone()
    common['global'] = global_views
    y = d['attribute_targets'].cuda(); labels=d['labels']; seen=torch.tensor(p['seen_classes']).cuda()
    positive = (y[ti]==1).sum(0); negative = (y[ti]==0).sum(0)
    pos_weight=((negative+.5)/(positive+.5)).clamp(1,20)
    results = []
    for seed in c['seeds']:
        learned = pooled_views(d,loc['learned'][str(seed)])
        wrong = learned[:,[0,2,3,4,1]].clone()
        variants={**common,'learned_part':learned,'wrong_part':wrong}
        for variant in c['variants']:
            checkpoint=RUN/f'{variant}_{seed}.pt'
            if checkpoint.exists():
                saved=torch.load(checkpoint,weights_only=True);assert saved['source']==source()
                results.append(saved['metrics']);continue
            torch.manual_seed(seed); rng=torch.Generator().manual_seed(seed)
            model=AttributeReadout(d,idx,c['readout_scale']).cuda()
            x=variants[variant].cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=c['readout_lr'],weight_decay=1e-4)
            for step in range(1,c['readout_steps']+1):
                ix=ti[torch.randint(len(ti),(c['readout_batch'],),generator=rng)]
                scores,q=model(x[ix]); target=y[ix]; known=target>=0
                bce=F.binary_cross_entropy_with_logits(q,target.clamp(0,1),pos_weight=pos_weight,reduction='none')
                bce=(bce*known).sum()/known.sum().clamp_min(1)
                ce=F.cross_entropy(scores[:,seen],labels[ix].cuda())
                loss=c['ce_weight']*ce+c['attribute_bce_weight']*bce
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                scores,q=model(x[vi]); scores=scores.cpu();q=q.cpu()
                met,pred=measures(scores,labels[vi],p,'development')
                attr=attribute_metrics(y[vi].cpu(),q.sigmoid())
                # Renaming both sides is invariant; breaking only one side measures reliance.
                perm=torch.randperm(q.shape[1],generator=torch.Generator().manual_seed(982))
                z=model.profiles.cpu(); permuted=c['readout_scale']*F.normalize(q[:,perm],dim=-1)@z.T
                renamed=c['readout_scale']*F.normalize(q[:,perm],dim=-1)@z[:,perm].T
                error=float((renamed-scores).abs().max());assert error<2e-5
                disrupted,_=measures(permuted,labels[vi],p,'development')
                record=dict(seed=seed,variant=variant,classification=met,
                    attribute_map=attr['attribute_map'],attribute_evaluable_count=attr['attribute_evaluable_count'],
                    train_loss=float(loss.detach()),train_ce=float(ce.detach()),train_bce=float(bce.detach()),
                    column_rename_max_error=error,wrong_column_classification=disrupted)
            torch.save(dict(source=source(),state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            predictions=pred,evidence=q,metrics=record),checkpoint)
            results.append(record)
            print('readout',variant,seed,'H',round(met['H'],3),'attr_mAP',round(attr['attribute_map'],4),flush=True)
            del model,x,opt
    sig=torch.load(ROOT/'cache/cub_siglip_v1/features_development.pt',weights_only=True)
    text=torch.load(ROOT/'cache/cub_siglip_v2/text.pt',weights_only=True)['class_text']
    lookup={r['image_id']:i for i,r in enumerate(sig['rows'])}; ix=[lookup[d['rows'][i]['image_id']] for i in vi]
    baseline,_=measures(20*sig['native_views'][ix,0]@text.T,labels[vi],p,'development')
    report=dict(source=source(),results=results,siglip2_native=baseline,
        attribute_mapping=[dict(name=a['name'],view=int(idx[i])) for i,a in enumerate(m['attributes'])],
        profile_sha256=d['profile_sha256'],fixed_steps=c['readout_steps'],
        caution='Extra official class attribute profiles are semantic side information. These ablations do not isolate all differences against native SigLIP2. No final test or calibration.')
    save_json(output_path,report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','localize','all'],default='all')
    args=parser.parse_args(); c,m,p=setup()
    save_json(REPORT/'protocol.json',dict(config=c,source=source(),time_started=time.strftime('%Y-%m-%d %H:%M:%S')))
    d=build_cache(c,m)
    if args.stage=='prepare':return
    ti=torch.tensor([i for i,r in enumerate(d['rows']) if r['role']=='train_seen'])
    vi=torch.tensor([i for i,r in enumerate(d['rows']) if r['role']!='train_seen'])
    assert len(ti)==len(vi)==500 and set(d['labels'][ti].tolist())==set(p['seen_classes'])
    g=geometry(c,d);loc=localize(c,d,g,ti,vi)
    if args.stage=='localize':return
    classification(c,d,m,p,loc,ti,vi)
    save_json(REPORT/'status.json',dict(status='completed',source=source(),training_images=500,
              development_images=500,final_test_images_used=0,seeds=c['seeds']))


if __name__=='__main__':main()
