import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
import coca_experiment as e
from coca_components import Fusion,select_text,symmetric_loss
from coca_data import validate,prepare

class CoCaTests(unittest.TestCase):
    def test_class_split_and_leak_guard(self):
        m=prepare();validate(m)
        bad=copy.deepcopy(m);next(r for r in bad['rows'] if r['role']=='train')['label']=15
        with self.assertRaises(AssertionError):validate(bad)

    def test_test_requires_seal(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'OUT',Path(folder)),patch.object(e,'detector_parts') as loader:
            with self.assertRaises(FileNotFoundError):e.ground({},dict(rows=[]),{},'test')
            loader.assert_not_called()

    def test_text_mask_indexing(self):
        x=torch.arange(4.).reshape(4,1,1);mask=torch.arange(8.).reshape(8,1,1)
        a,b=select_text(x,mask,[3,1]);self.assertEqual(a.flatten().tolist(),[3,1]);self.assertEqual(b.flatten().tolist(),[6,7,2,3])

    def test_fusion_initialization_and_missing_regions(self):
        torch.manual_seed(42);g=F.normalize(torch.randn(3,768),dim=-1);r=F.normalize(torch.randn(3,2,768),dim=-1);a=torch.randn_like(r);valid=torch.ones(3,2,dtype=torch.bool);confidence=torch.rand(3,2)
        for variant in ['baseline','attributes','caf','confidence','regions']:
            model=Fusion(variant).eval();out=model(g,r,a,valid,confidence);torch.testing.assert_close(out,g)
            if variant!='baseline':
                torch.nn.init.normal_(model.attribute_mlp[-1].weight,std=.01)
                torch.testing.assert_close(model(g,r,a,valid*False,confidence),g)

    def test_symmetric_objective_and_selection(self):
        im=torch.eye(3);logscale=torch.tensor(1.)
        self.assertLess(float(symmetric_loss(im,im,logscale)),float(symmetric_loss(im,im.flip(0),logscale)))
        metric=dict(harmonic=.6,zsl=.7)
        self.assertGreater(e.select_key(metric,0),e.select_key(metric,20))

if __name__=='__main__':unittest.main()
