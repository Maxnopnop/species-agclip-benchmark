"""Small deterministic tests of selection/metrics and sample isolation."""
import unittest
import torch
from multimodal.grounded_data import prepare
from multimodal.grounded_experiment import measures,rank

class GroundedProtocolTests(unittest.TestCase):
    def test_class_and_image_isolation(self):
        p,m=prepare();sets=[set(p[k]) for k in ['seen_classes','dev_unseen_classes','eval_unseen_classes']]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len({r['sha256'] for r in m['rows']}),700)
        roles={role:[r for r in m['rows'] if r['role']==role] for role in ['train_seen','dev_seen','dev_unseen','eval_seen','eval_unseen']}
        self.assertEqual([len(x) for x in roles.values()],[300,100,80,100,120])
        self.assertEqual({r['label'] for r in roles['train_seen']},sets[0])

    def test_gzsl_distinct_from_zsl(self):
        p=dict(seen_classes=[0,1],dev_unseen_classes=[2,3],eval_unseen_classes=[2,3])
        # Every image favors seen class zero globally, while the unseen-only
        # restricted candidate task correctly identifies both unseen examples.
        logits=torch.tensor([[5.,0.,0.,0.],[5.,0.,0.,0.],[5.,0.,4.,0.],[5.,0.,0.,4.]])
        m,pred=measures(logits,torch.arange(4),p,'evaluation')
        self.assertEqual(m['S'],50.);self.assertEqual(m['U'],0.);self.assertEqual(m['H'],0.);self.assertEqual(m['ZSL'],100.)
        self.assertEqual(pred.tolist(),[0,0,0,0])

    def test_selection_uses_both_seen_and_unseen(self):
        balanced=dict(H=60.,S=60.,U=60.,ce=1.)
        biased=dict(H=0.,S=100.,U=0.,ce=.1)
        self.assertGreater(rank(balanced),rank(biased))
        self.assertGreater(rank(dict(balanced,ce=.5)),rank(balanced))

if __name__=='__main__':unittest.main()
