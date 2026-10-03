"""Check native equivalence, gradients and peak memory before pilot lock."""
import gc,time,json
from pathlib import Path
import torch
from PIL import Image
from data_tools import ROOT,write_json
from coca_components import load_full,Tails,Fusion,image_prefix,text_prefix,symmetric_loss

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    model,transform,tokenize=load_full();tail=Tails(model)
    m=json.loads((ROOT/'data/confidence_fresh_v1/manifest.json').read_text())
    rows=[next(r for r in m['rows'] if r['label']==i and r['split']=='train') for i in range(2)]
    x=torch.stack([transform(Image.open(Path(m['image_root'])/r['path']).convert('RGB')) for r in rows]).cuda()
    tokens=tokenize(['a photo of a bird.','a bird with black wings.']).cuda()
    model=model.cuda();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        native_i=model.encode_image(x);native_t=model.encode_text(tokens)
        ip=image_prefix(model,x).cpu();tp,mask=text_prefix(model,tokens);tp=tp.cpu();mask=mask.cpu()
    del model,x,tokens;gc.collect();torch.cuda.empty_cache()
    tail=tail.cuda();fusion=Fusion('caf').cuda()
    ip=ip.cuda();tp=tp.cuda();mask=mask.cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        im=tail.image(ip);tx=tail.text(tp,mask)
    ie=float((im-native_i).abs().max());te=float((tx-native_t).abs().max())
    torch.testing.assert_close(im,native_i.float(),atol=.005,rtol=.03)
    torch.testing.assert_close(tx,native_t.float(),atol=.005,rtol=.03)
    start=time.monotonic()
    opt=torch.optim.AdamW(list(tail.parameters())+list(fusion.parameters()),lr=1e-5)
    for step in range(2):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            im=tail.image(ip);tx=tail.text(tp,mask)
            out=fusion(im,im[:,None].expand(-1,2,-1),tx[:,None].expand(-1,2,-1),torch.ones(2,2,device='cuda',dtype=torch.bool),torch.tensor([[.2,.8],[.4,.6]],device='cuda'))
            loss=symmetric_loss(out,tx,tail.logit_scale)
        opt.zero_grad(set_to_none=True);loss.backward()
        grad={name:sum(float(p.grad.float().square().sum()) for p in mod.parameters() if p.grad is not None)**.5 for name,mod in [('visual_tail',tail.visual_block),('text_tail',tail.text_block),('attribute_mlp',fusion.attribute_mlp),('caf',fusion.caf)]}
        if step==1:assert all(v>0 for v in grad.values()),grad
        torch.nn.utils.clip_grad_norm_(list(tail.parameters())+list(fusion.parameters()),1.);opt.step()
    result=dict(native_image_max_error=ie,native_text_max_error=te,gradient_norms=grad,loss=float(loss),two_updates_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    write_json(ROOT/'reports/coca_pilot_v1/smoke.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
