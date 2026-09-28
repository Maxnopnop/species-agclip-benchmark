import gc
import unittest
import torch
from torch.nn import functional as F
from multimodal.models import BACKBONES,DIMS,AlignedClassifier,load_backbone
from multimodal.experiment import split_indices


class MultimodalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def banks(self):
        torch.manual_seed(42)
        return F.normalize(torch.randn(7,512),dim=-1),F.normalize(torch.randn(12,512),dim=-1)

    def test_five_backbone_feature_shapes(self):
        # No download: verifies real architectures and feature extraction hooks.
        for name in BACKBONES:
            with self.subTest(backbone=name):
                model,_,dim=load_backbone(name,pretrained=False)
                with torch.inference_mode():
                    result=model(torch.randn(1,3,224,224))
                self.assertEqual(result.shape,(1,dim))
                self.assertTrue(torch.isfinite(result).all())
                del model,result
                gc.collect()

    def test_all_variants_backward_and_reload(self):
        classes,attrs=self.banks()
        for name in BACKBONES:
            for variant in ('baseline','average','agclip'):
                with self.subTest(backbone=name,variant=variant):
                    torch.manual_seed(42)
                    model=AlignedClassifier(name,classes,attrs,variant)
                    x=torch.randn(3,DIMS[name]);regions=torch.randn(3,3,DIMS[name])
                    ids=torch.tensor([[1,2,3],[4,5,0],[0,0,0]])
                    mask=torch.tensor([[1,1,1],[1,1,0],[0,0,0]],dtype=torch.bool)
                    logits,_=model(x,regions,ids,mask)
                    loss=F.cross_entropy(logits,torch.tensor([0,1,2]))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertIsNotNone(model.projection.weight.grad)
                    clone=AlignedClassifier(name,classes,attrs,variant)
                    clone.load_state_dict(model.state_dict())
                    model.eval();clone.eval()
                    torch.testing.assert_close(model(x,regions,ids,mask)[0],clone(x,regions,ids,mask)[0])

    def test_no_regions_preserves_baseline(self):
        classes,attrs=self.banks()
        for name in BACKBONES:
            baseline=AlignedClassifier(name,classes,attrs).eval()
            ag=AlignedClassifier(name,classes,attrs,'agclip').eval()
            ag.load_alignment(baseline.state_dict())
            x=torch.randn(2,DIMS[name]);regions=torch.randn(2,3,DIMS[name])
            ids=torch.zeros(2,3,dtype=torch.long);mask=torch.zeros(2,3,dtype=torch.bool)
            torch.testing.assert_close(baseline(x)[0],ag(x,regions,ids,mask)[0])
            # Zero-initialized fusion gate also starts at the alignment baseline.
            torch.testing.assert_close(baseline(x)[0],ag(x,regions,ids,torch.ones_like(mask))[0])

    def test_nested_shots_do_not_include_validation(self):
        rows=[{'path':f'{split}/{label}_{i}.jpg','label':label,'split':split}
              for split in ['train','val','test'] for label in range(3) for i in range(10)]
        cache={'rows':rows,'classes':[{}, {}, {}]}
        a=set(split_indices(cache,'train',2,42));b=set(split_indices(cache,'train',5,42))
        self.assertTrue(a.issubset(b))
        self.assertTrue(b.isdisjoint(split_indices(cache,'val')))
        self.assertTrue(b.isdisjoint(split_indices(cache,'test')))


if __name__=='__main__':
    unittest.main()
