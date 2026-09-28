"""Post-hoc reference: original CLIP with no task-adaptation weights or tuning."""
import torch
from multimodal.zsl import prepare_protocol,load_features,evaluate,ZSLHead
from data_tools import ROOT,write_json

def main():
    torch.set_num_threads(4);p,d=prepare_protocol()
    cache=load_features('clip_vit_b32',d,['eval_unseen','eval_seen']);bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    # CLIP residual projection initializes to exactly zero, preserving native features.
    head=ZSLHead('clip_vit_b32',bank,'baseline',p['seen_classes'],42).cuda()
    assert not bool(head.projection.weight.any()) and not bool(head.projection.bias.any())
    ui=[i for i,r in enumerate(cache['rows']) if r['role']=='eval_unseen'];si=[i for i,r in enumerate(cache['rows']) if r['role']=='eval_seen']
    z,pred=evaluate(head,cache,ui,p['eval_unseen_classes']);candidates=sorted(p['seen_classes']+p['eval_unseen_classes'])
    u,_=evaluate(head,cache,ui,candidates);s,_=evaluate(head,cache,si,candidates)
    h=2*u['per_class_accuracy']*s['per_class_accuracy']/max(1e-12,u['per_class_accuracy']+s['per_class_accuracy'])
    record=dict(timing='Post-hoc descriptive reference after final results, no retraining or selection changes',
        model='Original OpenAI CLIP ViT-B/32, same class-name prompts and global-image preprocessing, no task adaptation',
        zsl=z,gzsl_unseen=u['per_class_accuracy'],gzsl_seen=s['per_class_accuracy'],gzsl_h=h)
    write_json(ROOT/'reports/zsl_v1/native_clip_diagnostic.json',record)
    torch.save(pred,ROOT/'runs/zsl_v1/native_clip_predictions.pt');print(record)

if __name__=='__main__':main()
