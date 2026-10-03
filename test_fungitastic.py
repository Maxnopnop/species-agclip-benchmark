import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import fungitastic_experiment as exp
from fungitastic_data import validate_rows

class TemporalTests(unittest.TestCase):
    def test_observation_and_year_guards(self):
        rows=[]
        for label in range(10):
            for split,num in [('train',20),('val',10),('test',20)]:
                for i in range(num):
                    year=2023 if split=='test' else 2022
                    rows.append(dict(label=label,split=split,year=year,date=f'{year}-05-01',observation_id=len(rows)))
        validate_rows(rows)
        for field,value in [('observation_id',rows[0]['observation_id']),('year',2021),('date','2021-05-01')]:
            bad=copy.deepcopy(rows);bad[-1][field]=value
            with self.assertRaises(AssertionError):validate_rows(bad)

    def test_test_grounding_requires_seal(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(exp,'OUT',Path(folder)),patch.object(exp,'detector_parts') as detector:
            with self.assertRaises(FileNotFoundError):exp.ground({},dict(rows=[]),{},'test')
            detector.assert_not_called()

if __name__=='__main__':unittest.main()
