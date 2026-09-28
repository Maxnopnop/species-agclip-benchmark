import argparse,torch
from multimodal.grounded_data import prepare,ground,render_audit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','evaluation'],default='development');p.add_argument('--limit',type=int);p.add_argument('--audit',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    prepare();ground(args.stage,args.limit)
    if args.audit:render_audit()
