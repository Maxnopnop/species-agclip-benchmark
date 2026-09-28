"""End-to-end cached-feature experiment check (synthetic, not a benchmark)."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from data_tools import ROOT
from multimodal.experiment import run_experiment,aggregate


class ExperimentTests(unittest.TestCase):
    def test_three_variants_share_alignment_and_complete(self):
        torch.set_num_threads(4)
        (ROOT/'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            root=Path(directory);cache_dir=root/'cache';cache_dir.mkdir()
            rows=[dict(path=f'{s}/{c}/{j}.jpg',label=c,split=s)
                  for s in ['train','val','test'] for c in range(3) for j in range(2)]
            context={'manifest_sha256':'fixture','attributes_sha256':'fixture'}
            bank=dict(context,class_text=torch.randn(3,512),attribute_text=torch.randn(4,512))
            cache={'metadata':context,'rows':rows,'classes':[{'label':c,'name':str(c)} for c in range(3)],
                   'global':torch.randn(len(rows),1280),'regions':torch.randn(len(rows),3,1280),
                   'attribute_ids':torch.zeros(len(rows),3,dtype=torch.long),
                   'region_mask':torch.ones(len(rows),3,dtype=torch.bool),
                   'labels':torch.tensor([r['label'] for r in rows]),'pilot':True}
            torch.save(bank,cache_dir/'text_bank.pt');torch.save(cache,cache_dir/'efficientnet_b0_features.pt')
            out=root/'runs'/'efficientnet_b0_shots1_seed42'
            result=run_experiment('efficientnet_b0',cache_dir,out,1,42,2,2,'cpu')
            self.assertEqual([r['variant'] for r in result],['baseline','average','agclip'])
            self.assertTrue(all(r['images']==6 for r in result))
            self.assertEqual(len(aggregate(root/'runs')),3)
            self.assertTrue((out/'alignment'/'best.pt').exists())
            self.assertEqual(len(json.loads((out/'training_paths.json').read_text())),3)
            # Completed identical runs are reused; changed settings are rejected.
            self.assertEqual(run_experiment('efficientnet_b0',cache_dir,out,1,42,2,2,'cpu'),result)
            with self.assertRaisesRegex(ValueError,'settings differ'):
                run_experiment('efficientnet_b0',cache_dir,out,2,42,2,2,'cpu')


if __name__=='__main__':
    unittest.main()
