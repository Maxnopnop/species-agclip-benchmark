"""Run the same locked-selection machinery with the attribute-only model."""
import argparse
import torch
from data_tools import ROOT, digest
from run import seed_all
from multimodal import cub_token_experiment as runner
from multimodal.cub_bottleneck import AttributeBottleneck

original_provenance = runner.provenance


def provenance():
    result = original_provenance()
    for path in ['configs/cub_bottleneck_v1.json', 'multimodal/cub_bottleneck.py', 'cub_bottleneck_experiment.py']:
        result[path] = digest(ROOT / path)
    return result


def build(variant, seed, bank, probe, config):
    seed_all(seed)
    return AttributeBottleneck(bank, probe['state'], probe['mean_probability'], variant, config)


def fixed_probe(data, bank, config):
    saved = torch.load(ROOT / 'runs/cub_tokens_v1/probe.pt', weights_only=True)
    assert saved['provenance'] == original_provenance()
    saved['provenance'] = provenance()
    torch.save(saved, runner.OUT / 'probe.pt')
    return saved


def configure():
    runner.VERSION = 'cub_bottleneck_v1'
    runner.OUT = ROOT / 'runs' / runner.VERSION
    runner.REPORT = ROOT / 'reports' / runner.VERSION
    runner.CONFIG = ROOT / 'configs' / f'{runner.VERSION}.json'
    runner.provenance = provenance
    runner.build = build
    runner.train_probe = fixed_probe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['train', 'evaluate', 'all']); args = parser.parse_args()
    configure()
    if args.stage in ['train', 'all']: runner.train_all()
    if args.stage in ['evaluate', 'all']: runner.final_evaluation()
