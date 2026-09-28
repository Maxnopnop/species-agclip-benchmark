"""Frozen CLIP plus separate attribute tokens; matched fusion ablations."""
import torch
from torch import nn
from torch.nn import functional as F


class TokenFusion(nn.Module):
    def __init__(self, bank, probe_state, mean_probability, variant, config):
        super().__init__()
        self.variant = variant
        self.max_residual = config['max_residual']
        self.register_buffer('class_text', bank['class_text'].clone())
        self.register_buffer('attribute_text', bank['attribute_text'].clone())
        self.register_buffer('mean_probability', mean_probability.clone())
        self.register_buffer('permutation', torch.randperm(len(self.attribute_text), generator=torch.Generator().manual_seed(1729)))
        self.probe = nn.Linear(512, len(self.attribute_text))
        self.probe.load_state_dict(probe_state)
        self.probe.requires_grad_(False)
        h = config['hidden_dim']
        self.token_mlp = nn.Sequential(nn.Linear(1025, h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h))
        self.query = nn.Linear(512, h)
        self.attention = nn.MultiheadAttention(h, 4, dropout=0, batch_first=True)
        self.output = nn.Linear(h, 512)
        # Exact native baseline at step zero in every ablation.
        self.gate = nn.Parameter(torch.zeros(()))

    def make_tokens(self, views, valid, intervention=None):
        b, v, _ = views.shape
        a = len(self.attribute_text)
        probability = self.probe(views).sigmoid()
        if self.variant == 'constant' or intervention == 'constant':
            probability = self.mean_probability[None].expand(b, -1, -1)
        text = self.attribute_text
        if self.variant == 'permuted' or intervention == 'permuted':
            text = text[self.permutation]
        text = text[None, None].expand(b, v, -1, -1)
        confidence = 2 * probability[..., None] - 1
        if self.variant == 'pooled':
            weights = probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
            text = (weights[..., None] * text).sum(2, keepdim=True).expand(-1, -1, a, -1)
            confidence = confidence.mean(2, keepdim=True).expand(-1, -1, a, -1)
        if self.variant == 'region_only' or intervention == 'zero_text':
            text = torch.zeros_like(text)
            confidence = torch.zeros_like(confidence)
        visual = views[:, :, None].expand(-1, -1, a, -1)
        tokens = self.token_mlp(torch.cat([visual, text, confidence], -1)).flatten(1, 2)
        valid_views = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=valid.device), valid], 1)
        padding = ~valid_views[:, :, None].expand(-1, -1, a).flatten(1, 2)
        return tokens, padding, probability

    def forward(self, views, valid, intervention=None):
        tokens, padding, probability = self.make_tokens(views, valid, intervention)
        global_feature = views[:, 0]
        q = self.query(global_feature)[:, None]
        context, weights = self.attention(q, tokens, tokens, key_padding_mask=padding, need_weights=True)
        residual = F.normalize(self.output(context[:, 0]), dim=-1)
        embedding = F.normalize(global_feature + self.max_residual * self.gate.tanh() * residual, dim=-1)
        # Neither global feature, class vectors nor temperature is trainable.
        return dict(logits=20 * embedding @ self.class_text.T,
                    native_logits=20 * global_feature @ self.class_text.T,
                    embedding=embedding, global_embedding=global_feature,
                    probability=probability, attention=weights[:, 0])
