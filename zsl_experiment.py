import argparse,torch
from multimodal.zsl import train_matrix,evaluate_locked,OUT
from data_tools import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','all']);args=p.parse_args();torch.set_num_threads(4)
    try:
        if args.stage in ['train','all']:train_matrix()
        if args.stage in ['evaluate','all']:evaluate_locked()
    except Exception as e:write_json(OUT/'status.json',dict(status='failed',error=repr(e)));raise
