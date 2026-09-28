"""Run the predeclared expanded benchmark; seal validation choices before test."""
import argparse
import json
import torch
from data_tools import ROOT,write_json,digest
from multimodal.expanded import prepare_text,run_one,evaluate_test


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate','all'])
    args=p.parse_args();torch.set_num_threads(4)
    protocol=json.loads((ROOT/'configs/expanded_protocol.json').read_text())
    status=ROOT/'runs/expanded20/status.json'
    try:
        bank=prepare_text()
        if args.action in ('train','all'):
            for name in protocol['backbones']:
                for shots in protocol['shots']:
                    for seed in protocol['seeds']:
                        write_json(status,dict(stage='train',model=name,shots=shots,seed=seed,status='running'))
                        run_one(name,bank,protocol,shots,seed)
        if args.action in ('evaluate','all'):
            write_json(status,dict(stage='sealed_test_evaluation',status='running'))
            evaluate_test(protocol)
        write_json(status,dict(stage=args.action,status='complete'))
    except Exception as exc:
        write_json(status,dict(status='failed',error=str(exc)));raise


if __name__=='__main__':main()
