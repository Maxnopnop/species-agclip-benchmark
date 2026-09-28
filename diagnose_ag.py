"""Run a sealed validation-only diagnosis; no test metrics are calculated."""
import argparse
import json
import copy
import torch
from data_tools import ROOT,write_json
from multimodal.models import load_backbone
from multimodal.diagnostics import prepare_prefix,Tail,run_cell


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--models',nargs='+',default=['efficientnet_b0','clip_vit_b32']);args=p.parse_args()
    torch.set_num_threads(4)
    protocol=json.loads((ROOT/'configs/diagnostics_v1.json').read_text())
    bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    status=ROOT/'runs/diagnostics_v1/status.json';results=[]
    try:
        for name in args.models:
            write_json(status,dict(stage='prefix',backbone=name,status='running'))
            cache=prepare_prefix(name)
            if args.prepare_only:continue
            encoder,_,_=load_backbone(name);tail=copy.deepcopy(Tail(name,encoder)).eval();del encoder
            for regime in protocol['regimes']:
                for mode in protocol['modes']:
                    for lr in protocol['head_learning_rates']:
                        for seed in protocol['seeds']:
                            write_json(status,dict(stage='train',backbone=name,regime=regime,mode=mode,lr=lr,seed=seed,status='running',completed=len(results)))
                            results.append(run_cell(name,cache,tail,bank,protocol,regime,mode,seed,lr))
            del cache,tail
        write_json(ROOT/'runs/diagnostics_v1/results.json',results)
        write_json(status,dict(status='complete',completed=len(results),prepare_only=args.prepare_only))
    except Exception as error:
        write_json(status,dict(status='failed',error=repr(error),completed=len(results)));raise


if __name__=='__main__':main()
