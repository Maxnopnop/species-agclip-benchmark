import argparse,json
import torch
from data_tools import ROOT,write_json
from multimodal.visible_data import prepare_features
from multimodal.visible_train import run_cell


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['clip_b16','siglip2_b16']);args=p.parse_args()
    torch.set_num_threads(4);protocol=json.loads((ROOT/'configs/visible_protocol.json').read_text(encoding='utf-8'))
    status=ROOT/'runs/visible_v1/status.json';results=[]
    try:
        for name in args.models:
            cache=prepare_features(name)
            for k in ('global_features','patch_features','valid_patches','attribute_targets','evidence_masks','labels'):cache[k]=cache[k].cuda()
            for variant in protocol['variants']:
                for lr in protocol['head_learning_rates']:
                    for seed in protocol['optimization_seeds']:
                        write_json(status,dict(status='running',backbone=name,variant=variant,lr=lr,seed=seed,completed=len(results)))
                        results.append(run_cell(name,cache,variant,seed,lr,protocol))
            del cache;torch.cuda.empty_cache()
        write_json(ROOT/'runs/visible_v1/results.json',results);write_json(status,dict(status='complete',completed=len(results)))
    except Exception as e:write_json(status,dict(status='failed',error=repr(e),completed=len(results)));raise


if __name__=='__main__':main()
