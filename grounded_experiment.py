"""Run matched development matrix, then locked final evaluation."""
import argparse
from multimodal.grounded_experiment import train_all,final_evaluation

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','all']);args=parser.parse_args()
    if args.stage in ('train','all'):train_all()
    if args.stage in ('evaluate','all'):final_evaluation()
