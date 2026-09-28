"""Development-only privileged-crop fusion diagnostic, never deployment."""
import json
import torch
from data_tools import ROOT,digest,write_json
from multimodal import cub_token_experiment as runner
from multimodal.cub_rich_data import text_bank,load_data as detected_data
from diagnose_cub_oracle_regions import build_cache

original_provenance=runner.provenance


def provenance():
    result=original_provenance()
    for path in ['cub_oracle_fusion_experiment.py','configs/cub_oracle_fusion_v1.json','multimodal/cub_rich_data.py','runs/cub_coverage_v1/definition.json','diagnose_cub_oracle_regions.py','runs/cub_oracle_region_diagnosis/development.pt']:
        result[path]=digest(ROOT/path)
    return result


def load_data(stage):
    if stage!='development':raise ValueError('This diagnostic intentionally has no final/deployment loader.')
    data=detected_data(stage);oracle=build_cache(data,text_bank())
    data['native_views']=oracle['features'];data['targets']=oracle['targets'];data['valid']=oracle['valid'];return data


def configure():
    runner.VERSION='cub_oracle_fusion_v1';runner.OUT=ROOT/'runs'/runner.VERSION;runner.REPORT=ROOT/'reports'/runner.VERSION;runner.CONFIG=ROOT/'configs'/f'{runner.VERSION}.json'
    runner.provenance=provenance;runner.load_data=load_data;runner.text_bank=text_bank


if __name__=='__main__':
    configure();runner.train_all();selection=json.loads((runner.OUT/'selection_locked.json').read_text());reference=json.loads((ROOT/'runs/cub_rich_v1/selection_locked.json').read_text())
    write_json(runner.REPORT/'development_results.json',dict(selected=selection['selected'],detected_reference=reference['selected'],scope='Development-only PRIVILEGED localization diagnosis: true part positions from train/development images define two regional crops. Same data split, frozen global features, 158 attributes and fusion search as cub_rich_v1. Probe retrained on corresponding crops. No held-out final images. Selected development H is optimistic and not an independent performance estimate. Not deployable, not original-paper reproduction.'))
