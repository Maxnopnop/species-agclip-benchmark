import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np
import confidence_fresh as fresh

class FreshDataTests(unittest.TestCase):
    def test_fingerprints_catch_reencoded_images(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.png';b=Path(folder)/'b.png'
            im=Image.fromarray(np.random.default_rng(1).integers(0,256,(64,64,3),dtype=np.uint8));im.save(a,compress_level=1);im.save(b,compress_level=9)
            x=fresh.fingerprint(a);y=fresh.fingerprint(b)
            self.assertNotEqual(x['sha256'],y['sha256']);self.assertTrue(fresh.duplicates(x,[y],4))
    def test_holdout_grounding_requires_seal(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(fresh,'OUT',Path(folder)),patch.object(fresh,'detector_parts') as detector:
            with self.assertRaises(FileNotFoundError):fresh.ground({},dict(rows=[]),{},'test')
            detector.assert_not_called()

if __name__=='__main__':unittest.main()
