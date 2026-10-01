import unittest
import torch
from local_attribute_experiment import subset_data, subset_scores, predict_fused, select_candidate
from multimodal.attribute_path_v1 import AttributeReadout


class LocalAttributeTests(unittest.TestCase):
    def test_no_global_path_and_profile_renaming(self):
        torch.manual_seed(20)
        d=dict(positive_text=torch.randn(6,8),negative_text=torch.randn(6,8),profiles=torch.rand(75,6))
        keep=torch.tensor([False,True,True,True,False,True])
        data=subset_data(d,keep)
        model=AttributeReadout(data,torch.tensor([1,2,3,4]),20.)
        views=torch.randn(3,5,8,requires_grad=True)
        scores,q=model(views)
        scores.sum().backward()
        self.assertEqual(float(views.grad[:,0].abs().sum()),0.)
        changed=views.detach().clone();changed[:,0]=100*torch.randn(3,8)
        torch.testing.assert_close(scores,model(changed)[0],atol=0,rtol=0)
        perm=torch.tensor([2,0,3,1])
        torch.testing.assert_close(subset_scores(q[:,perm],data['profiles'][:,perm]),scores,atol=1e-5,rtol=1e-5)
        self.assertGreater(float((subset_scores(q[:,perm],data['profiles'])-scores).detach().abs().max()),.01)

    def test_native_fallback_confident_guard(self):
        native=torch.zeros(2,75);native[:,0]=10
        attr=torch.zeros(2,75);attr[:,60]=100
        self.assertTrue(torch.equal(predict_fused(native,attr,0),native.argmax(-1)))
        self.assertTrue(torch.equal(predict_fused(native,attr,100,0,True,.5),native.argmax(-1)))
        self.assertTrue((predict_fused(native,attr,100)==60).all())

    def test_unseen_error_constraint_and_fit_boundary(self):
        labels=torch.arange(75)
        native=labels.clone();native[:10]=74
        better=native.clone();better[:10]=labels[:10];better[50]=0
        candidates=[dict(weight=0.,penalty=0.),dict(weight=.5,penalty=0.)]
        preds=torch.stack([native,better])[:,None].expand(-1,3,-1)
        self.assertEqual(select_candidate(preds,labels,native,candidates,False,.01)[0],1)
        self.assertEqual(select_candidate(preds,labels,native,candidates,True,.01)[0],0)
        # Append an arbitrarily favorable holdout; passing fit-only rows is invariant.
        doubled=torch.cat([preds,preds.flip(0)],-1)
        self.assertEqual(select_candidate(doubled[:,:,:75],labels,native,candidates,True,.01)[0],0)


if __name__=='__main__':unittest.main()
