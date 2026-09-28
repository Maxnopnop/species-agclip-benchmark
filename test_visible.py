import unittest,inspect
import torch
from PIL import Image
from torch.nn import functional as F
from multimodal.visible_data import letterbox,patch_box,prepare_manifest
from multimodal.visible_train import VisibleHead,masked_attribute_loss,region_loss,shuffled_indices,attribute_metrics


class VisibleTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(1);torch.set_num_threads(4)

    def test_unknown_attributes_produce_no_gradient(self):
        logits=torch.randn(2,3,requires_grad=True);targets=torch.tensor([[1.,-1,0],[-1,-1,-1]])
        masked_attribute_loss(logits,targets).backward()
        self.assertEqual(float(logits.grad[targets<0].abs().sum()),0.)
        self.assertGreater(float(logits.grad[targets>=0].abs().sum()),0.)

    def test_region_loss_rewards_evidence_and_ignores_unknown(self):
        evidence=torch.tensor([[[True,False,False,False],[False,True,False,False]]]);valid=torch.ones(1,4,dtype=torch.bool)
        target=torch.tensor([[1.,-1.]])
        good=torch.tensor([[[.7,.1,.1,.1],[.1,.1,.4,.4]]]);bad=torch.tensor([[[.1,.7,.1,.1],[.1,.8,.05,.05]]])
        self.assertLess(float(region_loss(good,target,evidence,valid)),float(region_loss(bad,target,evidence,valid)))

    def test_letterbox_coordinates_and_tiny_region(self):
        im,geo=letterbox(Image.new('RGB',(400,200)));self.assertEqual(im.size,(224,224))
        self.assertEqual(geo,(1.,.5,0.,.25));self.assertEqual(int(patch_box([0,0,1,1],geo).sum()),112)
        self.assertEqual(int(patch_box([.499,.499,.501,.501],geo).sum()),1)

    def test_forward_never_uses_attribute_labels_or_boxes(self):
        cache=dict(class_text=F.normalize(torch.randn(20,32),dim=-1),attribute_text=F.normalize(torch.randn(16,32),dim=-1))
        head=VisibleHead(cache,'attribute_region');g=torch.randn(2,32);p=torch.randn(2,196,32);valid=torch.ones(2,196,dtype=torch.bool)
        self.assertEqual(list(inspect.signature(head.forward).parameters),['whole','patches','valid','intervention'])
        a=head(g,p,valid);b=head(g,p,valid,'zero')
        self.assertTrue(torch.allclose(b['logits'],b['global_logits']))
        self.assertFalse(torch.allclose(a['logits'],b['logits']))
        masked_attribute_loss(a['attribute_logits'],torch.randint(0,2,(2,16)).float()).backward()
        self.assertGreater(float(head.patch_adapter.up.weight.grad.abs().sum()),0.)

    def test_actual_split_and_shuffled_supervision(self):
        m=prepare_manifest();self.assertEqual({r['split'] for r in m['rows']},{'train','val'})
        self.assertEqual(sum('review_id' in r and r['split']=='train' for r in m['rows']),40)
        ti=[i for i,r in enumerate(m['rows']) if r['split']=='train'];mapping=shuffled_indices(m,ti,42)
        for i,r in enumerate(m['rows']):
            if r['split']=='val':self.assertEqual(i,int(mapping[i]))
            elif 'review_id' in r:
                donor=m['rows'][int(mapping[i])];self.assertNotEqual(r['label'],donor['label']);self.assertEqual(r['label']<10,donor['label']<10)
        metric=attribute_metrics(torch.tensor([[1.,-1],[0,1],[1,0]]),torch.tensor([[.9,.5],[.1,.9],[.8,.2]]))
        self.assertEqual(metric['attribute_map'],1.)


if __name__=='__main__':unittest.main()
