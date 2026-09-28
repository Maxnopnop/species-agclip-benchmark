"""Exercise dimension-general tokens before starting the new GPU experiment."""
import torch
from data_tools import ROOT,write_json
from multimodal.cub_siglip_tokens import SiglipTokenFusion


def main():
    torch.manual_seed(42);torch.set_num_threads(4);d=768;a=12;h=32;b=2
    bank={k:torch.nn.functional.normalize(torch.randn(n,d),dim=-1) for k,n in [('class_text',7),('attribute_text',a)]}
    probe=torch.nn.Linear(d,a);mean=torch.rand(3,a);views=torch.nn.functional.normalize(torch.randn(b,3,d),dim=-1);valid=torch.tensor([[True,False],[False,False]])
    c=dict(hidden_dim=h,max_residual=.2);counts=[]
    for variant in ['region_only','pooled','tokens','permuted','constant']:
        model=SiglipTokenFusion(bank,probe.state_dict(),mean,variant,c);counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad));out=model(views,valid)
        torch.testing.assert_close(out['logits'],out['native_logits'],atol=1e-5,rtol=1e-5)
        with torch.no_grad():model.gate.fill_(.3)
        modified=views.clone();modified[0,2]=torch.randn(d);modified[1,1:]=torch.randn(2,d)
        torch.testing.assert_close(model(views,valid)['logits'],model(modified,valid)['logits'],atol=1e-5,rtol=1e-5)
        model(views,valid)['logits'].square().mean().backward();assert model.token_mlp[0].weight.grad.abs().sum()>0;assert model.probe.weight.grad is None
    assert len(set(counts))==1
    write_json(ROOT/'reports/cub_siglip_v1/verification.json',dict(status='passed',dimension=d,matched_trainable_parameters=counts[0],checks=['step-zero native logits','invalid views masked','token gradients nonzero after gate activation','probe frozen','equal variant capacity']))
    print('SigLIP token verification passed')


if __name__=='__main__':main()
