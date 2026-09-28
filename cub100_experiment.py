"""Class-diversity follow-up with new project species and predeclared controls."""
import argparse
from data_tools import ROOT,digest
from multimodal import cub_token_experiment as runner
from multimodal.cub100_data import prepare,text_bank,load_data

original_provenance=runner.provenance


def provenance():
    result=original_provenance()
    for path in ['configs/cub100_v1.json','multimodal/cub100_data.py','cub100_experiment.py','multimodal/visible_data.py','configs/visible_model_revisions.json','data/cub100_v1/manifest.json']:
        result[path]=digest(ROOT/path)
    return result


def configure():
    runner.VERSION='cub100_v1';runner.OUT=ROOT/'runs'/runner.VERSION;runner.REPORT=ROOT/'reports'/runner.VERSION;runner.CONFIG=ROOT/'configs'/f'{runner.VERSION}.json'
    runner.provenance=provenance;runner.prepare=prepare;runner.load_data=load_data;runner.text_bank=text_bank


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate','all']);a=parser.parse_args();configure()
    if a.stage=='prepare':prepare()
    if a.stage in ['train','all']:runner.train_all()
    if a.stage in ['evaluate','all']:runner.final_evaluation()
