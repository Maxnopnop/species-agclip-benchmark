import unittest
import torch
from torch.nn import functional as F
from multimodal.diagnostics import ControlledHead,derangement,Tail
from multimodal.expanded import ExpandedHead
from multimodal.models import load_backbone


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(19)
        self.bank=dict(class_text=F.normalize(torch.randn(20,512),dim=-1),
                       attribute_text=F.normalize(torch.randn(58,512),dim=-1),
                       class_attributes=F.normalize(torch.randn(20,512),dim=-1))

    def test_controls_change_values_not_keys(self):
        h=ControlledHead('efficientnet_b0',self.bank,'matched').eval()
        r=F.normalize(torch.randn(2,5,512),dim=-1)
        original=h.retrieve(r);keys=h.attribute_text.clone()
        h.mode='permuted';shuffled=h.retrieve(r)
        self.assertTrue(torch.equal(keys,h.attribute_text));self.assertFalse(torch.allclose(original,shuffled))
        self.assertTrue(bool((h.permutation!=torch.arange(58)).all()))
        h.mode='zero';self.assertEqual(float(h.retrieve(r).abs().sum()),0.)

    def test_matched_exactly_reproduces_original_head(self):
        g=torch.randn(3,1280);r=torch.randn(3,5,1280)
        for variant in ['ag_mean','ag_attention','ag_aux']:
            old=ExpandedHead('efficientnet_b0',self.bank,variant).eval()
            new=ControlledHead('efficientnet_b0',self.bank,'matched',variant=variant).eval()
            missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
            self.assertEqual(missing,['permutation']);self.assertFalse(unexpected)
            with torch.no_grad():self.assertTrue(torch.allclose(old(g,r)[0],new(g,r)[0],atol=1e-6))

    def test_real_architecture_tail_reconstruction_and_gradient(self):
        for name in ['efficientnet_b0','clip_vit_b32']:
            encoder,_,_=load_backbone(name,pretrained=False);encoder.eval();tail=Tail(name,encoder).eval()
            boundary=encoder.features[-2] if name=='efficientnet_b0' else encoder.transformer.resblocks[-1]
            captured=[];hook=boundary.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
            with torch.no_grad():expected=F.normalize(encoder(torch.randn(2,3,224,224)).float(),dim=-1)
            hook.remove();tail.requires_grad_(True);actual=tail(captured[0]);self.assertTrue(torch.allclose(expected,actual,atol=1e-5))
            actual[:,0].sum().backward();self.assertGreater(sum(float(p.grad.abs().sum()) for p in tail.parameters() if p.grad is not None),0.)


if __name__=='__main__':unittest.main()
