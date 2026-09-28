import unittest
import torch
from torch.nn import functional as F
from multimodal.zsl import prepare_protocol,ZSLHead,remap_labels,objective

class ZSLTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(4)
    def test_class_and_image_disjointness(self):
        p,d=prepare_protocol();seen=set(p['seen_classes']);dev=set(p['dev_unseen_classes']);unseen=set(p['eval_unseen_classes'])
        self.assertEqual((len(seen),len(dev),len(unseen)),(10,4,6));self.assertFalse(seen&dev or seen&unseen or dev&unseen)
        self.assertEqual(len({r['sha256'] for r in d['rows']}),len(d['rows']))
        self.assertEqual({r['label'] for r in d['rows'] if r['role']=='train_seen'},seen)
        self.assertEqual(sum(r['role']=='train_seen' for r in d['rows']),200)
    def test_label_remapping_rejects_unseen_training(self):
        self.assertEqual(remap_labels(torch.tensor([3,1]),[1,3,5]).tolist(),[1,0])
        with self.assertRaises(ValueError):remap_labels(torch.tensor([2]),[1,3,5])
    def test_semantic_classifier_supports_untrained_candidates(self):
        p,_=prepare_protocol();bank={k:F.normalize(torch.randn(n,512),dim=-1) for k,n in [('class_text',20),('attribute_text',58),('class_attributes',20)]}
        head=ZSLHead('resnet18',bank,'ag_aux',p['seen_classes'],42);g=torch.randn(2,512);r=torch.randn(2,5,512)
        scores,_=head(g,r);self.assertEqual(scores[:,p['eval_unseen_classes']].shape,(2,6))
        self.assertFalse(any(k.startswith('classifier.') for k,_ in head.named_parameters()))
        cache=dict(global_features=g,region_features=r,labels=torch.tensor(p['seen_classes'][:2]));loss=objective(head,cache,[0,1],p);loss.backward()
        self.assertGreater(float(head.token_encoder[0].weight.grad.abs().sum()),0.)
        cache['labels']=torch.tensor(p['eval_unseen_classes'][:2])
        with self.assertRaises(ValueError):objective(head,cache,[0,1],p)
    def test_shuffled_targets_remain_in_seen_group(self):
        p,_=prepare_protocol();bank={k:F.normalize(torch.randn(n,512),dim=-1) for k,n in [('class_text',20),('attribute_text',58),('class_attributes',20)]}
        head=ZSLHead('resnet18',bank,'shuffled_attributes',p['seen_classes'],42)
        for c in p['seen_classes']:
            wrong=int(head.wrong_class_order[c]);self.assertNotEqual(c,wrong);self.assertIn(wrong,p['seen_classes']);self.assertEqual(c<10,wrong<10)

if __name__=='__main__':unittest.main()
