import unittest
import torch
from three_stage_experiment import VisualLinear,random_bank,build
from multimodal.models import DIMS
from multimodal.expanded import ExpandedHead,is_better
from multimodal.experiment import split_indices


class ThreeStageTests(unittest.TestCase):
    def test_visual_head_ignores_regions_and_has_no_text(self):
        model=VisualLinear('resnet18',20)
        g=torch.randn(3,DIMS['resnet18']);r=torch.randn(3,5,DIMS['resnet18'])
        torch.testing.assert_close(model(g,r)[0],model(g,r*100)[0],atol=0,rtol=0)
        self.assertFalse(any('text' in k or 'attribute' in k for k in model.state_dict()))

    def test_random_code_control_same_capacity_independent_of_text(self):
        a=random_bank(20,58,20261002);b=random_bank(20,58,20261002)
        torch.testing.assert_close(a['class_text'],b['class_text'])
        text={k:torch.randn_like(v) for k,v in a.items()}
        for name in ['resnet18','clip_vit_b32']:
            torch.manual_seed(42);random=build(name,'random_codes',text,dict(random_code_seed=20261002))
            torch.manual_seed(42);aligned=ExpandedHead(name,text)
            self.assertEqual(sum(p.numel() for p in random.parameters()),sum(p.numel() for p in aligned.parameters()))
            torch.testing.assert_close(random.class_text,a['class_text'])
            torch.testing.assert_close(random.projection.weight,aligned.projection.weight)

    def test_filtering_test_rows_preserves_training_sample(self):
        rows=[dict(split=s,label=c,path=f'{s}/{c}/{j}') for j in range(30) for c in range(20) for s in ['train','val','test']]
        full=dict(rows=rows,classes=list(range(20)));filtered=dict(rows=[r for r in rows if r['split']!='test'],classes=list(range(20)))
        for shots in [5,10,20]:
            x=[full['rows'][i]['path'] for i in split_indices(full,'train',shots,42)]
            y=[filtered['rows'][i]['path'] for i in split_indices(filtered,'train',shots,42)]
            self.assertEqual(x,y)
        self.assertFalse(is_better(dict(macro_f1=.5,loss=2),dict(macro_f1=.5,loss=1)))


if __name__=='__main__':unittest.main()
