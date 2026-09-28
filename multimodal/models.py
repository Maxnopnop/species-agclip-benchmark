import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT/'cache'/'torch'))
os.environ.setdefault('HF_HOME', str(ROOT/'cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
BACKBONES = ('efficientnet_b0', 'resnet18', 'convnext_tiny', 'vit_b_16', 'clip_vit_b32')
WEIGHT_ENUMS = {
    'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
    'convnext_tiny': models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    'vit_b_16': models.ViT_B_16_Weights.IMAGENET1K_V1,
}
DIMS = dict(efficientnet_b0=1280, resnet18=512, convnext_tiny=768, vit_b_16=768, clip_vit_b32=512)


def load_clip(pretrained=True):
    import open_clip
    # OpenAI weights via the maintained OpenCLIP implementation; no remote code.
    checkpoint = None
    if pretrained:
        from open_clip.pretrained import download_pretrained
        checkpoint = download_pretrained(open_clip.get_pretrained_cfg('ViT-B-32','openai'),
                                         prefer_hf_hub=False, cache_dir=str(ROOT/'cache'/'clip'))
    if checkpoint:
        # The official hash-verified OpenAI download is a TorchScript archive.
        # Its dedicated loader handles that format without torch.load's
        # weights_only incompatibility on PyTorch 2.6 and later.
        from open_clip.openai import load_openai_model
        from open_clip.transform import image_transform
        model = load_openai_model(checkpoint, precision='fp32', device='cpu')
        transform = image_transform(224, is_train=False,
            mean=model.visual.image_mean, std=model.visual.image_std,
            interpolation='bicubic', resize_mode='shortest')
    else:
        model, _, transform = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=None, force_quick_gelu=True)
    return model.eval(), transform, open_clip.get_tokenizer('ViT-B-32')


def load_backbone(name, pretrained=True):
    if name == 'clip_vit_b32':
        clip, transform, _ = load_clip(pretrained)
        visual = clip.visual
        visual.requires_grad_(False)
        return visual.eval(), transform, DIMS[name]
    weights = WEIGHT_ENUMS[name]
    model = getattr(models, name)(weights=weights if pretrained else None)
    if name == 'efficientnet_b0':
        model.classifier = nn.Identity()
    elif name == 'resnet18':
        model.fc = nn.Identity()
    elif name == 'convnext_tiny':
        model.classifier[2] = nn.Identity()
    elif name == 'vit_b_16':
        model.heads = nn.Identity()
    model.requires_grad_(False)
    return model.eval(), weights.transforms(), DIMS[name]


class AlignedClassifier(nn.Module):
    """Image features -> frozen CLIP text space. Baseline, AVG or learned CAF.

    Small-data setting: pretrained visual backbones stay frozen, and only the
    projection / residual adapter and attribute modules are trained.
    This is supervised adaptation, not CLIP pretraining from scratch or ZSL.
    """
    def __init__(self, backbone, class_text, attribute_text, variant='baseline'):
        super().__init__()
        self.backbone, self.variant = backbone, variant
        self.register_buffer('class_text', F.normalize(class_text.float(), dim=-1))
        self.register_buffer('attribute_text', F.normalize(attribute_text.float(), dim=-1))
        dim = DIMS[backbone]
        self.native_clip = backbone == 'clip_vit_b32'
        self.projection = nn.Linear(dim, 512)
        if self.native_clip:
            # Initially preserve OpenAI's native CLIP feature geometry.
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        self.logit_scale = nn.Parameter(torch.tensor(2.65926))
        if variant in ('average', 'agclip'):
            self.attribute_encoder = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 512))
        if variant == 'agclip':
            self.cross_attention = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
            self.norm1 = nn.LayerNorm(512)
            self.ffn = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Dropout(.1), nn.Linear(1024, 512))
            self.gate = nn.Parameter(torch.tensor(0.0))

    def project(self, x):
        return x + self.projection(x) if self.native_clip else self.projection(x)

    def forward(self, global_features, region_features=None, attribute_ids=None, region_mask=None):
        global_embed = self.project(global_features.float())
        if self.variant != 'baseline':
            if region_features is None or attribute_ids is None or region_mask is None:
                raise ValueError('Attribute variants require region features, IDs and valid mask.')
            regions = self.project(region_features.float())
            text = self.attribute_text[attribute_ids]
            tokens = self.attribute_encoder(torch.cat([regions, text], dim=-1))
            valid = region_mask.bool()
            # A missing detection must not create arbitrary visual evidence.
            has_region = valid.any(dim=1, keepdim=True)
            if self.variant == 'average':
                pooled = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
                global_embed = torch.where(has_region, .5*(global_embed+pooled), global_embed)
            else:
                safe_mask = valid.clone()
                safe_mask[~has_region.squeeze(1), 0] = True  # avoid all-masked attention NaNs
                context, _ = self.cross_attention(global_embed.unsqueeze(1), tokens, tokens,
                                                  key_padding_mask=~safe_mask, need_weights=False)
                fused = global_embed + self.gate.tanh() * context.squeeze(1)
                fused = fused + self.gate.tanh() * self.ffn(self.norm1(fused))
                global_embed = torch.where(has_region, fused, global_embed)
        embedding = F.normalize(global_embed, dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * embedding @ self.class_text.T
        return logits, embedding

    def load_alignment(self, state):
        keys = ('projection.', 'logit_scale')
        selected = {k: v for k, v in state.items() if k.startswith(keys)}
        self.load_state_dict(selected, strict=False)
