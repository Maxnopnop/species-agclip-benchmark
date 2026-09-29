"""Audit the pinned official DEAL ViT explanation path; no optimizer steps."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'work/four_routes_review_20260929/DEAL'
OUT = ROOT / 'reports/four_routes_v1'


def load_functions(source, names, namespace):
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<audited upstream functions>', 'exec'), namespace)
    return namespace


class TinyAttentionModel(nn.Module):
    """Small differentiable attention fixture, not a species classifier."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 5, 5))
        self.visual = nn.Module()
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList([nn.Identity()])

    def forward(self, images, texts):
        n = len(texts)
        multiplier = torch.arange(1, n + 1, device=self.weight.device)[:, None, None]
        a = (self.weight * multiplier).softmax(-1)
        self.visual.transformer.resblocks[0].attn_probs = a
        coefficients = torch.arange(25, device=a.device).reshape(1, 5, 5) / 25
        score = (a * coefficients).sum((1, 2))
        logits = score[:, None] * multiplier[:, 0, 0][None]
        return logits, logits.T


def main():
    torch.manual_seed(42)
    src = (SOURCE / 'explainer.py').read_text(encoding='utf-8')
    losses = load_functions((SOURCE / 'loss.py').read_text(),
                           ['normalize_heatmap', 'BatchSeparationLoss', 'BatchConsistencyLoss'],
                           dict(torch=torch, nn=nn))
    baseline = dict(torch=torch, np=np, start_layer=-1, start_layer_text=-1)
    official = load_functions(src, ['interpret'], baseline.copy())['interpret']
    # A separately labelled, synthetic-only differentiable candidate; not an upstream modification.
    repaired_src = src.replace(
        'torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()',
        'torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True, create_graph=True)[0]'
    ).replace('cam = blk.attn_probs.detach()', 'cam = blk.attn_probs')
    repaired = load_functions(repaired_src, ['interpret'], baseline.copy())['interpret']
    model = TinyAttentionModel().cuda()
    images = torch.zeros(1, 3, 4, 4, device='cuda')
    texts = torch.ones(3, 2, device='cuda', dtype=torch.long)
    def penalty(fn):
        maps = fn(images, texts, model, 'cuda')
        return maps, (losses['BatchSeparationLoss']()([maps[1:]]) +
                      losses['BatchConsistencyLoss']()([maps[1:]], [maps[0]]))
    maps, reg = penalty(official)
    assert not maps.requires_grad and not reg.requires_grad
    reference = model.weight.square().sum()
    g0 = torch.autograd.grad(reference, model.weight, retain_graph=True)[0]
    g1 = torch.autograd.grad(reference + reg, model.weight)[0]
    torch.testing.assert_close(g0, g1, rtol=0, atol=0)
    fixed_maps, fixed_reg = penalty(repaired)
    gradient = torch.autograd.grad(fixed_reg, model.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    result = dict(upstream_commit='3a00f994c0855e3f90e1243791a921591c6623f6',
                  source_sha256=hashlib.sha256(src.encode()).hexdigest(),
                  official_heatmap_requires_grad=maps.requires_grad,
                  official_regularizer_requires_grad=reg.requires_grad,
                  official_gradient_difference=float((g0-g1).abs().max()),
                  differentiable_candidate_gradient_norm=float(gradient.norm()),
                  scope='Exact extracted official interpret/loss functions on a synthetic attention fixture. '
                        'No training, no species accuracy, no claim about unpublished author code or RN50 path.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'deal_gradient_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
