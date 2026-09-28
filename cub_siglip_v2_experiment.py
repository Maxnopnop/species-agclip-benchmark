"""Frozen SigLIP 2 follow-up with matched attribute ablations."""
import argparse
from multimodal import cub_siglip_v2_experiment as runner

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','all']);a=p.parse_args()
    if a.stage in ['train','all']:runner.train_all()
    if a.stage in ['evaluate','all']:runner.final_evaluation()
