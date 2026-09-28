import argparse
from multimodal.cub_token_experiment import train_all, final_evaluation

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train', 'evaluate', 'all'])
    args = p.parse_args()
    if args.stage in ['train', 'all']: train_all()
    if args.stage in ['evaluate', 'all']: final_evaluation()
