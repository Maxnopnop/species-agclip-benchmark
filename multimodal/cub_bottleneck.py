"""Attribute-only centered residual: an explicit test of the visual shortcut."""
import torch
from torch import nn
from torch.nn import functional as F
from .cub_tokens import TokenFusion


class AttributeBottleneck(TokenFusion):
    def __init__(self, bank, probe_state, mean_probability, variant, config):
        super().__init__(bank, probe_state, mean_probability, variant, config)
        # Remove every path from visual features to the residual except the
        # frozen attribute predictor. Semantic identity alone is not sufficient.
        h = config['hidden_dim']
        self.token_mlp = nn.Sequential(nn.Linear(513, h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h))
        del self.query
        self.query_token = nn.Parameter(torch.randn(1, 1, h) * .01)
        g = torch.Generator().manual_seed(1927)
        orthogonal = torch.linalg.qr(torch.randn(512, len(bank['attribute_text']), generator=g))[0].T
        self.register_buffer('identity_text', orthogonal)
        with torch.no_grad(): self.gate.fill_(.25)

    def encode_tokens(self, probability, valid, intervention=None):
        b, v, a = probability.shape
        text = F.normalize(self.attribute_text - self.attribute_text.mean(0), dim=-1)
        if self.variant == 'identity': text = self.identity_text
        if self.variant == 'permuted' or intervention == 'permuted': text = text[self.permutation]
        signed = 2 * probability[..., None] - 1
        semantic = signed * text[None, None]
        if intervention == 'zero_text': semantic = torch.zeros_like(semantic); signed = torch.zeros_like(signed)
        tokens = self.token_mlp(torch.cat([semantic, signed], -1)).flatten(1, 2)
        ok = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=valid.device), valid], 1)
        mask = ~ok[:, :, None].expand(-1, -1, a).flatten(1, 2)
        return tokens, mask

    def make_tokens(self, views, valid, intervention=None):
        probability = self.probe(views).sigmoid()
        if self.variant == 'constant' or intervention == 'constant': probability = self.mean_probability[None].expand(len(views), -1, -1)
        tokens, mask = self.encode_tokens(probability, valid, intervention)
        return tokens, mask, probability

    def forward(self, views, valid, intervention=None):
        tokens, mask, probability = self.make_tokens(views, valid, intervention)
        prior, _ = self.encode_tokens(self.mean_probability[None].expand(len(views), -1, -1), valid, intervention)
        q = self.query_token.expand(len(views), -1, -1)
        actual, attention = self.attention(q, tokens, tokens, key_padding_mask=mask)
        reference, _ = self.attention(q, prior, prior, key_padding_mask=mask)
        residual = F.normalize(self.output(actual[:, 0]), dim=-1) - F.normalize(self.output(reference[:, 0]), dim=-1)
        global_feature = views[:, 0]
        embedding = F.normalize(global_feature + self.max_residual * self.gate.tanh() * residual, dim=-1)
        return dict(logits=20 * embedding @ self.class_text.T, native_logits=20 * global_feature @ self.class_text.T,
                    embedding=embedding, global_embedding=global_feature, probability=probability, attention=attention[:, 0])
