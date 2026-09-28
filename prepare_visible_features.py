import argparse
import torch
from multimodal.visible_data import prepare_features

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['clip_b16','siglip2_b16']);args=p.parse_args()
    torch.set_num_threads(4)
    for name in args.models:
        prepare_features(name);print(f'Finished {name}',flush=True)
