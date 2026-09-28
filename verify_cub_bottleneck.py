import torch
from data_tools import write_json
from multimodal.cub_token_experiment import setup, text_bank, load_data, OUT
from multimodal.cub_bottleneck import AttributeBottleneck
from run import seed_all
import json
from data_tools import ROOT


def main():
    setup(); config = json.loads((ROOT / 'configs/cub_bottleneck_v1.json').read_text())
    data = load_data('development'); bank = text_bank(); probe = torch.load(OUT / 'probe.pt', weights_only=True)
    views, valid = data['native_views'][:4], data['valid'][:4]; counts = []
    for variant in config['variants']:
        seed_all(42); model = AttributeBottleneck(bank, probe['state'], probe['mean_probability'], variant, config)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        out = model(views, valid)
        assert torch.isfinite(out['logits']).all()
        out['logits'].square().mean().backward()
        assert all(p.grad is None for p in model.probe.parameters())
        baseline = 20 * views[:, 0] @ bank['class_text'].T
        for mode in ['constant', 'zero_text']:
            torch.testing.assert_close(model(views, valid, mode)['logits'], baseline, rtol=1e-5, atol=1e-5)
        if variant != 'constant':
            assert (out['logits'] - baseline).abs().max() > .0001
            assert model.token_mlp[0].weight.grad.abs().sum() > 0
        else: torch.testing.assert_close(out['logits'], baseline, rtol=1e-5, atol=1e-5)
    assert len(set(counts)) == 1
    write_json(ROOT / 'reports/cub_bottleneck_v1/verification.json', dict(status='passed', trainable_parameters=counts[0], no_attribute_variation_exact_native=True, frozen_probe=True, nonzero_attribute_gradients=True, matched_capacity=True))
    print('PASS: attribute-only residual, constant/zero-attribute exact native fallback, frozen probe, matched capacity.')


if __name__ == '__main__': main()
