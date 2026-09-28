"""Matched token fusion using 158 train-supported attributes, same photos."""
import argparse
from data_tools import ROOT,digest
from multimodal import cub_token_experiment as runner
from multimodal.cub_rich_data import text_bank,load_data

original_provenance=runner.provenance


def provenance():
    result=original_provenance()
    for path in ['configs/cub_rich_v1.json','multimodal/cub_rich_data.py','cub_rich_experiment.py','runs/cub_coverage_v1/definition.json']:
        result[path]=digest(ROOT/path)
    return result


def configure():
    runner.VERSION='cub_rich_v1';runner.OUT=ROOT/'runs'/runner.VERSION;runner.REPORT=ROOT/'reports'/runner.VERSION;runner.CONFIG=ROOT/'configs'/f'{runner.VERSION}.json'
    runner.provenance=provenance;runner.load_data=load_data;runner.text_bank=text_bank


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','all']);a=parser.parse_args();configure()
    if a.stage in ['train','all']:runner.train_all()
    if a.stage in ['evaluate','all']:runner.final_evaluation()
