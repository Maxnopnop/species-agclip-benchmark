import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from multimodal.expanded import ExpandedHead,is_better,run_one,select_variants,evaluate_test
from data_tools import write_json


def bank():
    torch.manual_seed(7)
    return dict(class_text=F.normalize(torch.randn(3,512),dim=-1),
                attribute_text=F.normalize(torch.randn(9,512),dim=-1),
                class_attributes=F.normalize(torch.randn(3,512),dim=-1),
                manifest_sha256='test_manifest',attributes_sha256='test_attributes')


class ExpandedTests(unittest.TestCase):
    def test_nonzero_gate_trains_attributes_immediately(self):
        torch.set_num_threads(4)
        for variant in ('ag_mean','ag_attention','ag_aux'):
            model=ExpandedHead('resnet18',bank(),variant)
            logits,_=model(torch.randn(4,512),torch.randn(4,5,512))
            F.cross_entropy(logits,torch.tensor([0,1,2,0])).backward()
            self.assertGreater(model.token_encoder[0].weight.grad.abs().sum().item(),0)

    def test_validation_ties_choose_lower_loss(self):
        self.assertTrue(is_better(dict(macro_f1=.8,loss=.3),dict(macro_f1=.8,loss=.4)))
        self.assertFalse(is_better(dict(macro_f1=.79,loss=.1),dict(macro_f1=.8,loss=.4)))

    def test_sealed_end_to_end_and_label_independent_forward(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache/expanded20';cache.mkdir(parents=True)
            rows=[dict(path=f'{split}/{c}/{j}.jpg',split=split,label=c) for split in ('train','val','test') for c in range(3) for j in range(2)]
            features=dict(metadata={'manifest_sha256':'test_manifest'},rows=rows,
                classes=[dict(id=c,label=c,name=str(c)) for c in range(3)],
                global_features=F.normalize(torch.randn(18,512),dim=-1),
                region_features=F.normalize(torch.randn(18,5,512),dim=-1),labels=torch.tensor([r['label'] for r in rows]))
            torch.save(features,cache/'resnet18_features.pt')
            protocol=dict(backbones=['resnet18'],shots=[1],seeds=[42],
                variants=['baseline','region_only','ag_mean','ag_attention','ag_aux'],alignment_steps=20,comparison_steps=20,aux_weight=.1)
            with patch('multimodal.expanded.ROOT',root):
                run_one('resnet18',bank(),protocol,1,42,'cpu')
                folder=root/'runs/expanded20/resnet18_shots1_seed42'
                self.assertFalse((folder/'ag_aux/test_metrics.json').exists())
                select_variants(protocol)
                self.assertTrue((root/'runs/expanded20/selection_locked.json').exists())
                results=evaluate_test(protocol);self.assertEqual(len(results),5)
                self.assertTrue(all(r['images']==6 for r in results))
                saved=torch.load(folder/'ag_aux/best.pt',weights_only=True)
                self.assertGreater(saved['diagnostic']['branch_parameter_l2_change'],0)
                model=ExpandedHead('resnet18',saved['state_dict'],'ag_aux');model.load_state_dict(saved['state_dict']);model.eval()
                with torch.inference_mode():
                    logits=model(features['global_features'],features['region_features'])[0]
                    # Forward accepts only image features; changing labels cannot affect output.
                    features['labels']=2-features['labels']
                    self.assertTrue(torch.equal(logits,model(features['global_features'],features['region_features'])[0]))


if __name__=='__main__':unittest.main()
