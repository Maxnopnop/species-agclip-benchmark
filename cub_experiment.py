import argparse
from multimodal.cub_experiment import setup,train_all,final_evaluation
from multimodal.cub_data import prepare,pixels

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','all']);a=p.parse_args();setup()
    if a.stage=='prepare':prepare();pixels('development')
    if a.stage in ['train','all']:train_all()
    if a.stage in ['evaluate','all']:final_evaluation()
