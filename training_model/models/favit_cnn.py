import math
from typing import Tuple

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
    """Global Adaptive Module applied to ViT patch tokens only."""

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
        delta_tokens = self.net(feature_map).flatten(2).transpose(1, 2)
        return patch_tokens + delta_tokens


class CNNLocalExtractor(nn.Module):
    """Small CNN that provides local spatial features for LAM."""

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
    """Local Adaptive Module with cross-attention from patch tokens to CNN tokens."""

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

    def _to_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        return tokens.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

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

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        out = self.proj(out)
        return patch_tokens + self.beta * out


class CNN_feature_extractor_branch(nn.Module):
    """Lightweight CNN branch for FFT/SRM frequency or noise inputs."""

    def __init__(self, freq_in_channels: int = 3, freq_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(freq_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, freq_dim),
            nn.LayerNorm(freq_dim),
        )

    def forward(self, freq_x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(freq_x))


class FAViTCNN(nn.Module):
    """FA-ViT RGB branch plus a lightweight frequency branch for binary detection."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1,
        freq_in_channels: int = 3,
        freq_dim: int = 128,
        use_freq: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FAViTCNN is configured for binary classification with one logit.")

        self.use_freq = use_freq
        self.vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.vit_dim = int(self.vit.num_features)
        self.rgb_norm = None if hasattr(self.vit, "norm") else nn.LayerNorm(self.vit_dim)
        for param in self.vit.parameters():
            param.requires_grad = False

        num_blocks = len(self.vit.blocks)
        required_lam_indices = (0, 3, 6)
        invalid_indices = [idx for idx in required_lam_indices if idx >= num_blocks]
        if invalid_indices:
            raise ValueError(f"LAM block indices out of range for {num_blocks} ViT blocks: {invalid_indices}")

        self.local_cnn = CNNLocalExtractor(self.vit_dim)
        self.gam_blocks = nn.ModuleList(GAM(self.vit_dim) for _ in range(num_blocks))
        self.lam1 = LAM(self.vit_dim, num_heads=8)
        self.lam2 = LAM(self.vit_dim, num_heads=8)
        self.lam3 = LAM(self.vit_dim, num_heads=8)
        self.freq_branch = CNN_feature_extractor_branch(
            freq_in_channels=freq_in_channels,
            freq_dim=freq_dim,
        )

        input_dim = self.vit_dim + freq_dim if use_freq else self.vit_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.3),
            nn.Linear(input_dim, num_classes),
        )

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        return x

    def forward_rgb_features(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.local_cnn(x)
        tokens = self._embed_tokens(x)

        for idx, block in enumerate(self.vit.blocks):
            cls_token = tokens[:, :1]
            patch_tokens = tokens[:, 1:]
            patch_tokens = self.gam_blocks[idx](patch_tokens)

            if idx == 0:
                patch_tokens = self.lam1(patch_tokens, cnn_feat)
            elif idx == 3:
                patch_tokens = self.lam2(patch_tokens, cnn_feat)
            elif idx == 6:
                patch_tokens = self.lam3(patch_tokens, cnn_feat)

            tokens = torch.cat((cls_token, patch_tokens), dim=1)
            tokens = block(tokens)

        norm = self.vit.norm if hasattr(self.vit, "norm") else self.rgb_norm
        tokens = norm(tokens)
        return tokens[:, 0]

    def forward(self, x: torch.Tensor, freq_x: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_feature = self.forward_rgb_features(x)

        if self.use_freq:
            if freq_x is None:
                freq_x = x
            freq_feature = self.freq_branch(freq_x)
            fused_feature = torch.cat((rgb_feature, freq_feature), dim=1)
        else:
            fused_feature = rgb_feature

        logits = self.classifier(fused_feature)
        return logits, fused_feature


class FALoss(nn.Module):
    """Forgery-aware loss using the final classifier Linear weight as the real prototype."""

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
