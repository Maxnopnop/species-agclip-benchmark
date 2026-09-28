"""Attribute coverage on existing photos; no new labels or final-image tuning."""
import json
from collections import defaultdict
import torch
from torch import nn
from data_tools import ROOT, write_json, digest
from run import seed_all
from multimodal.cub_data import prepare, BASE, lines
from multimodal.cub_token_experiment import load_data
from multimodal.cub_model import masked_bce
from multimodal.visible_train import attribute_metrics

OUT = ROOT / 'runs/cub_coverage_v1'
REPORT = ROOT / 'reports/cub_coverage_v1'
PARTS = {
    'bill': [2], 'wing': [9, 13], 'upperparts': [1, 9, 13],
    'underparts': [3, 4], 'back': [1], 'upper_tail': [14],
    'under_tail': [14], 'breast': [4], 'throat': [15], 'forehead': [6],
    'nape': [10], 'belly': [3], 'leg': [8, 12], 'crown': [5],
    'eye': [7, 11], 'head': [2, 5, 6, 7, 10, 11, 15], 'tail': [14],
    'primary': list(range(1, 16)), 'shape': list(range(1, 16))}


def targets(rows):
    names = {int(r[0]): r[1] for r in lines(ROOT / 'data/cub/attributes.txt')}
    index = {r['image_id']: i for i, r in enumerate(rows)}
    part = defaultdict(dict)
    for r in lines(BASE / 'parts/part_locs.txt'):
        if int(r[0]) in index: part[int(r[0])][int(r[1])] = [float(r[2]), float(r[3]), int(r[4])]
    usable = {}
    for row in rows:
        width, height = row['size']; side = min(width, height)
        visible = {pid for pid, (x, y, ok) in part[row['image_id']].items() if ok and (width-side)/2 <= x <= (width+side)/2 and (height-side)/2 <= y <= (height+side)/2}
        for aid, name in names.items():
            group = name.split('::')[0][4:]
            key = next((k for k in sorted(PARTS, key=len, reverse=True) if group == k or group.startswith(k + '_')), None)
            usable[row['image_id'], aid] = key is not None and bool(visible & set(PARTS[key]))
    y = torch.full((len(rows), len(names)), -1.)
    for r in lines(BASE / 'attributes/image_attribute_labels.txt'):
        iid, aid, positive, certainty = map(int, r[:4])
        if iid in index and certainty >= 3 and usable[iid, aid]: y[index[iid], aid-1] = positive
    return y, names


def prototype_scores(train_y, train_labels, prediction, mask, classes):
    prototypes = []
    for c in classes:
        y = train_y[train_labels == c]; known = y >= 0
        prototypes.append(((y.clamp_min(0).sum(0)+1) / (known.sum(0)+2)).clamp(.01,.99))
    proto = torch.stack(prototypes)
    return ((prediction[:,None] * proto.log()[None] + (1-prediction[:,None]) * (1-proto).log()[None]) * mask[:,None]).sum(-1)


def accuracy(scores, labels, classes):
    return float((torch.tensor(classes)[scores.argmax(-1)] == labels).float().mean() * 100)


def main():
    seed_all(42); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    p, manifest = prepare(); data = load_data('development')
    cache = OUT / 'development_targets.pt'
    saved = torch.load(cache, weights_only=True) if cache.exists() else None
    if saved is not None and saved['source_sha256'] == digest(__file__):
        all_y, names = saved['targets'], saved['names']
    else:
        all_y, names = targets(data['rows'])
        torch.save(dict(targets=all_y, names=names, source_sha256=digest(__file__)), cache)
    ti = torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='train_seen'])
    vi = torch.tensor([i for i,r in enumerate(data['rows']) if r['role']!='train_seen'])
    si = torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='dev_seen'])
    original = [int(a['source_id'])-1 for a in manifest['attributes']]
    # Same labels/masks for shared dimensions in the coverage comparison.
    torch.testing.assert_close(all_y[:,original], data['targets'][:,0], rtol=0, atol=0)
    supported = ((all_y[ti]==1).sum(0)>=8) & ((all_y[ti]==0).sum(0)>=20)
    extended = sorted(set(original) | set(supported.nonzero().flatten().tolist()))
    definition = dict(original=original, extended=extended, support='At least 8 positive and 20 negative training observations; original 24 retained.', scope='Fixed CUB photos/split; one deterministic linear probe per attribute set. No final images used. Absolute body-size attributes excluded because image scale is not physical scale. Visible-part point inside native center crop and certainty >=3; not a guarantee that the full part is visible.', source_sha256=digest(__file__))
    write_json(OUT / 'definition.json', definition)
    x = data['native_views'][:,0]; results=[]
    for label, columns in [('original24',original), ('extended',extended)]:
        y=all_y[:,columns]; history=[]; best=None
        for lr in [.01,.001]:
            seed_all(42); model=nn.Linear(512,len(columns))
            with torch.no_grad(): model.weight.zero_(); model.bias.zero_()
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
            for step in range(601):
                if step in [0,100,300,600]:
                    with torch.no_grad(): pred=model(x).sigmoid()
                    metrics=attribute_metrics(y[vi],pred[vi]); row=dict(lr=lr,step=step,development_map=metrics['attribute_map']); history.append(row)
                    if best is None or row['development_map']>best['development_map']:
                        best=row; state={k:v.detach().clone() for k,v in model.state_dict().items()}
                if step==600: break
                opt.zero_grad(); loss=masked_bce(model(x[ti]),y[ti]); loss.backward(); opt.step()
        model.load_state_dict(state)
        with torch.no_grad(): pred=model(x).sigmoid()
        normal=prototype_scores(y[ti],data['labels'][ti],pred[si],torch.ones_like(pred[si]),p['seen_classes'])
        matched=prototype_scores(y[ti],data['labels'][ti],pred[si],y[si]>=0,p['seen_classes'])
        oracle=prototype_scores(y[ti],data['labels'][ti],y[si].clamp_min(0),y[si]>=0,p['seen_classes'])
        common=[columns.index(i) for i in original]
        result=dict(variant=label,attributes=len(columns),selection=best,history=history,common24_map=attribute_metrics(y[vi][:,common],pred[vi][:,common])['attribute_map'], predicted_attribute_seen_accuracy=accuracy(normal,data['labels'][si],p['seen_classes']), oracle_visibility_matched_predicted_seen_accuracy=accuracy(matched,data['labels'][si],p['seen_classes']), oracle_attribute_seen_accuracy=accuracy(oracle,data['labels'][si],p['seen_classes']))
        results.append(result); torch.save(dict(state=state,columns=columns,selection=best),OUT/f'{label}_probe.pt')
        print(json.dumps(result),flush=True)
    write_json(REPORT/'development_results.json',dict(definition=definition,results=results,warning='Coverage changes also change dimension and available supervision. These are development diagnostics, not final performance. True-attribute/visibility scores are oracle diagnostics, never deployable results; a naive prototype classifier is not a theoretical upper bound.'))


if __name__=='__main__': main()
