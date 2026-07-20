import math
from typing import Iterable, Tuple

import timm
import torch
from torch import nn
from torch.nn import functional as F


def _grid_size(num_tokens: int) -> int:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected square patch token count, got {num_tokens}.")
    return side


class GAM(nn.Module):
    """Global Adaptive Module applied to ViT patch tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden_dim = max(dim // 2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = patch_tokens.shape
        side = _grid_size(num_tokens)
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, dim, side, side)
        delta = self.net(feature_map).flatten(2).transpose(1, 2)
        return patch_tokens + delta


class CNNLocalExtractor(nn.Module):
    """Lightweight CNN branch for local forgery clues."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LAM(nn.Module):
    """Local Adaptive Module using cross-attention from patch tokens to CNN tokens."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.zeros(1))

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        return x.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, patch_tokens: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = patch_tokens.shape
        side = _grid_size(num_tokens)
        if cnn_feat.shape[-2:] != (side, side):
            cnn_feat = F.interpolate(cnn_feat, size=(side, side), mode="bilinear", align_corners=False)

        cnn_tokens = cnn_feat.flatten(2).transpose(1, 2)
        if cnn_tokens.shape != patch_tokens.shape:
            raise ValueError(
                f"CNN tokens shape {tuple(cnn_tokens.shape)} must match patch tokens "
                f"shape {tuple(patch_tokens.shape)}."
            )

        q = self._to_heads(self.q(patch_tokens))
        k = self._to_heads(self.k(cnn_tokens))
        v = self._to_heads(self.v(cnn_tokens))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        out = self.proj(out)
        return patch_tokens + self.beta * out


class RedesignedFAViT(nn.Module):
    """Simplified FA-ViT style model with a frozen ViT backbone and trainable adapters."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1,
        lam_block_indices: Iterable[int] = (0, 3, 6),
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("RedesignedFAViT is configured for binary classification with one logit.")

        self.vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = int(self.vit.num_features)
        for param in self.vit.parameters():
            param.requires_grad = False

        num_blocks = len(self.vit.blocks)
        self.gams = nn.ModuleList(GAM(self.embed_dim) for _ in range(num_blocks))
        self.lam_block_indices = tuple(lam_block_indices)
        invalid_indices = [idx for idx in self.lam_block_indices if idx < 0 or idx >= num_blocks]
        if invalid_indices:
            raise ValueError(f"LAM block indices out of range for {num_blocks} ViT blocks: {invalid_indices}")

        self.local_extractor = CNNLocalExtractor(self.embed_dim)
        self.lams = nn.ModuleDict({str(idx): LAM(self.embed_dim, num_heads=8) for idx in self.lam_block_indices})
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_feat = self.local_extractor(x)
        tokens = self._embed_tokens(x)

        for idx, block in enumerate(self.vit.blocks):
            cls_token = tokens[:, :1]
            patch_tokens = tokens[:, 1:]

            patch_tokens = self.gams[idx](patch_tokens)
            if idx in self.lam_block_indices:
                patch_tokens = self.lams[str(idx)](patch_tokens, cnn_feat)

            tokens = torch.cat((cls_token, patch_tokens), dim=1)
            tokens = block(tokens)

        tokens = self.vit.norm(tokens)
        features = tokens[:, 0]
        logits = self.classifier(features)
        return logits, features


class FALoss(nn.Module):
    """Forgery-aware loss using the binary classifier weight as the real prototype."""

    def __init__(self, margin: float = 0.25, scale: float = 32.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.view(-1).long()
        real_mask = labels == 0
        fake_mask = labels == 1
        if not torch.any(real_mask) or not torch.any(fake_mask):
            return features.new_zeros(())

        features = F.normalize(features, dim=1)
        prototype = F.normalize(classifier_weight[0].view(1, -1), dim=1)
        similarities = (features * prototype).sum(dim=1)

        sp = similarities[real_mask].mean()
        sn = similarities[fake_mask].mean()
        return F.softplus(self.scale * (sn - sp + self.margin))
