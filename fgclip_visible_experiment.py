import json,torch
from data_tools import ROOT,write_json
from multimodal.fgclip_visible import prepare_features
from multimodal.visible_train import run_cell

def main():
    torch.set_num_threads(4)
    path=ROOT/'configs/fgclip_visible_protocol.json'
    if not path.exists():
        protocol=json.loads((ROOT/'configs/visible_protocol.json').read_text(encoding='utf-8'))
        protocol.update(backbones=['fgclip_b16'],optimization_seeds=[42],follow_up='Exploratory one-seed small test after weak classification gains in CLIP and SigLIP 2. Unchanged data, annotations, head, losses and selection budget. FG-CLIP already has fine-grained region-text pretraining.')
        write_json(path,protocol)
    protocol=json.loads(path.read_text(encoding='utf-8'));status=ROOT/'runs/visible_v1/fgclip_status.json';results=[]
    try:
        write_json(status,dict(status='preparing_features'))
        cache=prepare_features()
        for k in ('global_features','patch_features','valid_patches','attribute_targets','evidence_masks','labels'):cache[k]=cache[k].cuda()
        for variant in protocol['variants']:
            for lr in protocol['head_learning_rates']:
                for seed in protocol['optimization_seeds']:
                    write_json(status,dict(status='running',variant=variant,lr=lr,seed=seed,completed=len(results)))
                    results.append(run_cell('fgclip_b16',cache,variant,seed,lr,protocol))
        write_json(ROOT/'runs/visible_v1/fgclip_results.json',results);write_json(status,dict(status='complete',completed=len(results)))
    except Exception as e:write_json(status,dict(status='failed',error=repr(e),completed=len(results)));raise

if __name__=='__main__':main()
