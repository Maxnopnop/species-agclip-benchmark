"""Mechanism and leakage checks for the new attribute decision path."""
import unittest
import torch
from torch.nn import functional as F
from multimodal.attribute_path_v1 import PartLocalizer, AttributeReadout, geometry, train_position_prior


class AttributePathTests(unittest.TestCase):
    def test_padding_is_not_localization_evidence(self):
        torch.manual_seed(5)
        model=PartLocalizer(F.normalize(torch.randn(4,8),dim=-1))
        x=F.normalize(torch.randn(2,6,8),dim=-1);mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,0,0]]).bool()
        original=model(x,mask)
        altered=x.clone();altered[~mask]=999
        torch.testing.assert_close(model(altered,mask),original)
        self.assertEqual(float(original.masked_select(~mask[:,None]).sum().detach()),0.)
        (-original[0,0,0].log()).backward()
        self.assertTrue(torch.isfinite(model.query.grad).all())
        self.assertGreater(float(model.query.grad.abs().sum()),0.)

    def test_profiles_are_fixed_and_attribute_path_is_mandatory(self):
        torch.manual_seed(6)
        d=dict(positive_text=torch.randn(7,8),negative_text=torch.randn(7,8),profiles=torch.rand(3,7))
        model=AttributeReadout(d,torch.tensor([0,1,2,3,4,0,1]),20)
        x=torch.randn(2,5,8);scores,q=model(x)
        self.assertNotIn('profiles',dict(model.named_parameters()))
        expected=20*F.normalize(q,dim=-1)@model.profiles.T
        torch.testing.assert_close(scores,expected)
        permutation=torch.tensor([6,5,4,3,2,1,0])
        renamed=20*F.normalize(q[:,permutation],dim=-1)@model.profiles[:,permutation].T
        torch.testing.assert_close(scores,renamed)
        with torch.no_grad():model.weight.zero_();model.bias.zero_()
        self.assertEqual(float(model(x)[0].abs().sum().detach()),0.)
        self.assertEqual(float(model(x*99)[0].abs().sum().detach()),0.)

    def test_development_annotations_do_not_fit_spatial_prior(self):
        c=dict(part_groups={str(i):[i+1] for i in range(4)},target_sigma_diagonal=.05)
        rows=[dict(size=[100,100],parts={str(i):[25.,25.,1] for i in range(1,5)}),
              dict(size=[100,100],parts={str(i):[75.,75.,1] for i in range(1,5)})]
        d=dict(rows=rows,masks=torch.ones(2,4).bool(),shapes=torch.tensor([[2,2],[2,2]]))
        before=train_position_prior(c,d,geometry(c,d),torch.tensor([0]))
        d['rows'][1]['parts']={str(i):[1.,99.,1] for i in range(1,5)}
        after=train_position_prior(c,d,geometry(c,d),torch.tensor([0]))
        torch.testing.assert_close(before,after)


if __name__=='__main__':unittest.main()
