"""Check causal input paths, frozen assets and matched controls before training."""
import torch
from data_tools import write_json
from multimodal.cub_token_experiment import setup, load_data, text_bank, build, REPORT


def main():
    c = setup(); data = load_data('development'); bank = text_bank()
    a = len(bank['attribute_text'])
    probe = dict(state=dict(weight=10 * (bank['attribute_text'] - bank['negative_text']), bias=torch.zeros(a)), mean_probability=torch.full((3, a), .5))
    views, valid = data['native_views'][:4], data['valid'][:4]
    counts, checks = [], []
    for variant in c['variants']:
        model = build(variant, 42, bank, probe, c)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        native = 20 * views[:, 0] @ bank['class_text'].T
        o = model(views, valid)
        torch.testing.assert_close(o['logits'], native, atol=1e-5, rtol=1e-5)
        assert not any(p.requires_grad for p in model.probe.parameters())
        before = {k: v.clone() for k, v in model.probe.state_dict().items()}
        with torch.no_grad(): model.gate.fill_(.3)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        for _ in range(2):
            opt.zero_grad(); loss = model(views, valid)['logits'][:, 0].sum(); loss.backward(); opt.step()
        assert all(torch.equal(before[k], v) for k, v in model.probe.state_dict().items())
        assert model.token_mlp[0].weight.grad.abs().sum() > 0
        assert torch.equal(model(views, valid)['native_logits'], native)
        missing = torch.zeros_like(valid)
        altered = views.clone(); altered[:, 1:] = torch.randn_like(altered[:, 1:])
        torch.testing.assert_close(model(views, missing)['logits'], model(altered, missing)['logits'], rtol=0, atol=0)
        tokens, mask, _ = model.make_tokens(views, valid)
        assert tokens.shape == (4, 3 * a, c['hidden_dim']) and not mask[:, :a].any()
        if variant in ['pooled', 'region_only']:
            reshaped = tokens.reshape(4, 3, a, -1)
            torch.testing.assert_close(reshaped[:, :, 0], reshaped[:, :, -1], atol=0, rtol=0)
        if variant == 'tokens':
            wrong, _, _ = model.make_tokens(views, valid, 'permuted')
            # Attention is invariant to reordering tokens; changing text/probability
            # associations must nevertheless change their content and prediction.
            delta = (model(views, valid)['logits'] - model(views, valid, 'permuted')['logits']).abs().max()
            assert (tokens - wrong).abs().max() > .01 and delta > 1e-7
            assert (tokens[:, 0] - tokens[:, 1]).abs().max() > .01
        checks.append(dict(variant=variant, passed=True))
    assert len(set(counts)) == 1
    train = {r['path'] for r in data['rows'] if r['role'] == 'train_seen'}
    dev = {r['path'] for r in data['rows'] if r['role'] != 'train_seen'}
    assert train.isdisjoint(dev)
    write_json(REPORT / 'verification.json', dict(status='passed', trainable_parameters=counts[0], checks=checks, native_logits_unchanged=True, frozen_probe_unchanged=True, masked_regions_ignored=True, separate_attribute_tokens=True))
    print('PASS: frozen global logits/probe, trainable fusion gradients, token identity, masked regions, matched capacity, data isolation.')


if __name__ == '__main__': main()
