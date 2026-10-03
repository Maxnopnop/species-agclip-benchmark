import unittest
import torch
from confidence_experiment import pool_weights,Head

class ConfidenceTests(unittest.TestCase):
    def test_weights_and_padding(self):
        c=torch.tensor([[.1,.3],[.2,9.],[0.,0.],[.1,.2]])
        v=torch.tensor([[1,1],[1,0],[1,1],[0,0]],dtype=torch.bool)
        w=pool_weights(c,v,'ag_confidence')
        torch.testing.assert_close(w,torch.tensor([[.25,.75],[1.,0.],[.5,.5],[0.,0.]]))
        torch.testing.assert_close(pool_weights(c,v,'ag_shuffled')[0],w[0].flip(0))
    def test_uniform_equivalence_and_missing_fallback(self):
        torch.manual_seed(42);bank=dict(class_text=torch.nn.functional.normalize(torch.randn(10,512),dim=-1),attribute_text=torch.randn(58,512))
        a=Head('clip_vit_b32','ag_uniform',bank,{}).eval();b=Head('clip_vit_b32','ag_confidence',bank,{}).eval();b.load_state_dict(a.state_dict())
        g=torch.randn(3,512);r=torch.randn(3,2,512);ids=torch.zeros(3,2,dtype=torch.long);v=torch.ones(3,2,dtype=torch.bool);c=torch.ones(3,2)
        torch.testing.assert_close(a(g,r,ids,v,c),b(g,r,ids,v,c))
        t=Head('clip_vit_b32','text',bank,{}).eval();t.load_state_dict({k:z for k,z in a.state_dict().items() if k in t.state_dict()})
        torch.testing.assert_close(a(g,r,ids,v&False,c),t(g,r,ids,v,c))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        unequal=torch.tensor([[.05,.9]]).expand(3,-1)
        self.assertGreater(float((a(g,r,ids,v,c)-b(g,r,ids,v,unequal)).detach().abs().max()),1e-6)
        b.train();torch.nn.functional.cross_entropy(b(g,r,ids,v,unequal),torch.tensor([0,1,2])).backward()
        self.assertGreater(float(b.token_encoder[0].weight.grad.abs().sum()),0.)
    def test_visual_ignores_semantic_inputs(self):
        model=Head('efficientnet_b0','visual',dict(class_text=torch.randn(10,512)),{})
        g=torch.randn(2,1280)
        torch.testing.assert_close(model(g,None,None,None,None),model(g,torch.randn(2),None,None,None))
        self.assertFalse(any('text' in k for k in model.state_dict()))

if __name__=='__main__':unittest.main()
