"""Unified five-backbone image-text / attribute-guided experiments."""
import argparse
import json
from pathlib import Path
import torch
from data_tools import ROOT,write_json
from multimodal.models import BACKBONES
from multimodal.preparation import text_bank,ground_regions,extract_features
from multimodal.experiment import run_experiment,aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train','all'])
    p.add_argument('--manifest',default=str(ROOT/'data'/'pilot'/'manifest.json'))
    p.add_argument('--attributes',default=str(ROOT/'configs'/'pilot_attributes.json'))
    p.add_argument('--cache',default=str(ROOT/'cache'/'pilot_multimodal'))
    p.add_argument('--output',default=str(ROOT/'runs'/'pilot_multimodal'))
    p.add_argument('--models',nargs='+',choices=BACKBONES,default=list(BACKBONES))
    p.add_argument('--shots',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--alignment-epochs',type=int,default=10)
    p.add_argument('--comparison-epochs',type=int,default=10)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    if min(args.shots,args.alignment_epochs,args.comparison_epochs)<1:
        p.error('shots and epoch counts must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for debugging.')
    torch.set_num_threads(4)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    try:
        if args.action in ('prepare','all'):
            write_json(status_path,{'stage':'text_bank','status':'running'})
            text_bank(args.manifest,args.attributes,args.cache,args.device)
            write_json(status_path,{'stage':'region_grounding','status':'running'})
            ground_regions(args.manifest,args.attributes,args.cache,args.device)
        for name in args.models:
            if args.action in ('prepare','all'):
                write_json(status_path,{'stage':'features','model':name,'status':'running'})
                extract_features(name,args.manifest,args.attributes,args.cache,args.device)
            if args.action in ('train','all'):
                write_json(status_path,{'stage':'training','model':name,'status':'running'})
                run_experiment(name,args.cache,output/f'{name}_shots{args.shots}_seed{args.seed}',
                               args.shots,args.seed,args.alignment_epochs,args.comparison_epochs,args.device)
                aggregate(output)
        write_json(status_path,{'stage':args.action,'status':'complete','models':args.models})
    except Exception as exc:
        previous=json.loads(status_path.read_text()) if status_path.exists() else {}
        write_json(status_path,dict(previous,status='failed',error=str(exc)))
        raise


if __name__=='__main__':
    main()
