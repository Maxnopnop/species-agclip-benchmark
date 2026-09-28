import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data_tools import digest, write_json
from run import metrics_from_confusion, seed_all
from .models import AlignedClassifier


def split_indices(cache, split, shots=None, seed=42):
    indices=[i for i,r in enumerate(cache['rows']) if r['split']==split]
    if shots is not None:
        selected=[]
        for label in range(len(cache['classes'])):
            group=sorted((i for i in indices if cache['rows'][i]['label']==label),
                         key=lambda i:cache['rows'][i]['path'])
            random.Random(seed+label).shuffle(group)
            if shots>len(group) or shots<1:
                raise ValueError(f'Need 1..{len(group)} shots for class {label}')
            selected.extend(group[:shots])
        indices=selected
    return indices


def forward_batch(model, cache, indices, device):
    return model(cache['global'][indices].to(device),cache['regions'][indices].to(device),
                 cache['attribute_ids'][indices].to(device),cache['region_mask'][indices].to(device))


@torch.inference_mode()
def measure(model,cache,indices,device):
    model.eval()
    count=len(cache['classes'])
    confusion=np.zeros((count,count),dtype=np.int64)
    predictions=[]
    loss,total_top5=0.0,0
    for start in range(0,len(indices),64):
        batch=indices[start:start+64]
        logits,_=forward_batch(model,cache,batch,device)
        labels=cache['labels'][batch].to(device)
        loss+=float(F.cross_entropy(logits,labels,reduction='sum'))
        scores=logits.softmax(-1)
        pred=scores.argmax(-1)
        total_top5+=int((scores.topk(min(5,count),dim=-1).indices==labels[:,None]).any(1).sum())
        np.add.at(confusion,(labels.cpu().numpy(),pred.cpu().numpy()),1)
        for i,y,p,s in zip(batch,labels.cpu().tolist(),pred.cpu().tolist(),scores.max(1).values.cpu().tolist()):
            predictions.append({'path':cache['rows'][i]['path'],'true_label':y,'predicted_label':p,'score':s})
    metrics=dict(metrics_from_confusion(confusion),loss=loss/len(indices),
                 top5_accuracy=total_top5/len(indices),images=len(indices))
    return metrics,confusion,predictions


def fit(model,cache,train_indices,val_indices,epochs,seed,device,output):
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    best=-1.0
    history=[]
    generator=random.Random(seed)
    for epoch in range(1,epochs+1):
        model.train()
        order=train_indices.copy()
        generator.shuffle(order)
        train_loss=0.0
        for start in range(0,len(order),32):
            batch=order[start:start+32]
            logits,_=forward_batch(model,cache,batch,device)
            loss=F.cross_entropy(logits,cache['labels'][batch].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            train_loss+=float(loss.detach())*len(batch)
        scheduler.step()
        val,_,_=measure(model,cache,val_indices,device)
        history.append(dict(epoch=epoch,train_loss=train_loss/len(order),**{f'val_{k}':v for k,v in val.items()}))
        if val['macro_f1']>best:
            best=val['macro_f1']
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch
        if epoch==1 or epoch%5==0 or epoch==epochs:
            print(f'  {model.variant} epoch {epoch}/{epochs}: val F1={val["macro_f1"]:.3f}',flush=True)
    model.load_state_dict(state)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys())
        writer.writeheader();writer.writerows(history)
    return model,state,best_epoch


def run_experiment(backbone,cache_dir,output,shots,seed,alignment_epochs,comparison_epochs,device):
    seed_all(seed)
    cache_dir,output=Path(cache_dir),Path(output)
    cache=torch.load(cache_dir/f'{backbone}_features.pt',weights_only=True)
    bank=torch.load(cache_dir/'text_bank.pt',weights_only=True)
    if any(cache['metadata'][k]!=bank[k] for k in ('manifest_sha256','attributes_sha256')):
        raise ValueError('Text and image feature caches differ.')
    output.mkdir(parents=True,exist_ok=True)
    config=dict(backbone=backbone,shots=shots,seed=seed,alignment_epochs=alignment_epochs,
                comparison_epochs=comparison_epochs,manifest_sha256=bank['manifest_sha256'],
                attributes_sha256=bank['attributes_sha256'],pilot=cache['pilot'],
                protocol='frozen visual/text encoders; trained adapters and attribute fusion',
                cache_sha256=digest(cache_dir/f'{backbone}_features.pt'))
    if (output/'config.json').exists():
        previous=json.loads((output/'config.json').read_text(encoding='utf-8'))
        if previous!=config:
            raise ValueError('Run settings differ from existing run. Use a new output directory.')
        if (output/'comparison.json').exists():
            print('Completed experiment retained:',output,flush=True)
            return json.loads((output/'comparison.json').read_text(encoding='utf-8'))
    write_json(output/'config.json',config)
    train_idx=split_indices(cache,'train',shots,seed)
    val_idx=split_indices(cache,'val')
    test_idx=split_indices(cache,'test')
    write_json(output/'training_paths.json',[cache['rows'][i]['path'] for i in train_idx])
    baseline=AlignedClassifier(backbone,bank['class_text'],bank['attribute_text'],'baseline')
    print(f'{backbone}: stage 1 alignment, {len(train_idx)} training images',flush=True)
    stage1=output/'alignment'
    checkpoint=stage1/'best.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=True)
        initial_state=saved['state_dict']
        baseline.load_state_dict(initial_state)
    else:
        baseline,initial_state,best_epoch=fit(baseline,cache,train_idx,val_idx,alignment_epochs,seed,device,stage1)
        torch.save({'state_dict':initial_state,'config':config,'best_epoch':best_epoch},checkpoint)
    # Stage-one test set is not inspected for method selection. All variants
    # receive equal additional epochs and are chosen with validation only.
    results=[]
    for variant in ('baseline','average','agclip'):
        seed_all(seed)
        dest=output/variant
        model=AlignedClassifier(backbone,bank['class_text'],bank['attribute_text'],variant)
        model.load_alignment(initial_state)
        start=time.monotonic()
        model,state,best_epoch=fit(model,cache,train_idx,val_idx,comparison_epochs,seed,device,dest)
        seconds=time.monotonic()-start
        metrics,confusion,predictions=measure(model,cache,test_idx,device)
        metrics.update(backbone=backbone,variant=variant,shots=shots,seed=seed,pilot=cache['pilot'],
                       comparison_training_seconds=seconds,best_epoch=best_epoch,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        torch.save({'state_dict':state,'config':config,'variant':variant,
                    'classes':cache['classes'],'best_epoch':best_epoch},dest/'best.pt')
        write_json(dest/'metrics.json',metrics)
        write_json(dest/'predictions.json',predictions)
        np.savetxt(dest/'confusion_matrix.csv',confusion,delimiter=',',fmt='%d')
        results.append(metrics)
        print(f'{backbone}/{variant}: held-out pilot/test accuracy={metrics["top1_accuracy"]:.3f}',flush=True)
    write_json(output/'comparison.json',results)
    return results


def aggregate(result_root):
    rows=[]
    for file in sorted(Path(result_root).glob('*/comparison.json')):
        rows.extend(json.loads(file.read_text(encoding='utf-8')))
    if not rows:
        return []
    path=Path(result_root)/'comparison.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    write_json(Path(result_root)/'comparison.json',rows)
    return rows
