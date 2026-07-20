"""Frequency-aware ViT with spatial, frequency, and adaptive token branches."""

import math
from typing import Iterable, Optional

import timm
import torch
from torch import nn
from torch.nn import functional as F


def _grid_size(num_tokens: int) -> int:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected square patch token count, got {num_tokens}.")
    return side


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep_prob) / keep_prob


class GAM(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden_dim = max(dim // 2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.drop_path = DropPath(drop_path)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = patch_tokens.shape
        side = _grid_size(num_tokens)
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, dim, side, side)
        delta = self.net(feature_map).flatten(2).transpose(1, 2)
        return patch_tokens + self.drop_path(delta)


def _group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(groups, channels) or 1, channels)


class CNNLocalExtractor(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), _group_norm(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), _group_norm(128), nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            _group_norm(out_channels), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _srm_kernels() -> torch.Tensor:
    k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
    k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    k3 = torch.zeros(5, 5)
    k3[1:4, 1:4] = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    return torch.stack((k1, k2, k3))


class FrequencyExtractor(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels().unsqueeze(1).repeat(1, 3, 1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), _group_norm(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), _group_norm(128), nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            _group_norm(out_channels), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The fixed filter has no parameters; keeping this outside no_grad also
        # allows gradients to reach input-side augmentation modules if present.
        return self.net(F.conv2d(x, self.srm_weight, padding=2))


class CrossModalFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q, self.k, self.v, self.proj = (nn.Linear(dim, dim) for _ in range(4))
        self.beta = nn.Parameter(torch.zeros(1))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, patch_tokens: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        q, k, v = self._heads(self.q(patch_tokens)), self._heads(self.k(aux_tokens)), self._heads(self.v(aux_tokens))
        out = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1) @ v
        out = out.transpose(1, 2).reshape_as(patch_tokens)
        return patch_tokens + self.beta * self.proj(out)


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm, self.drop, self.fc = nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(x)))


class FrequencyAwareFAViT(nn.Module):
    def __init__(self, backbone_name: str = "vit_base_patch16_224", pretrained: bool = True,
                 num_classes: int = 1, lam_block_indices: Iterable[int] = (0, 3, 6),
                 freq_block_indices: Iterable[int] = (1, 4, 7),
                 aux_head_block_index: Optional[int] = 6, num_unfrozen_blocks: int = 2,
                 gam_drop_path: float = 0.1, head_dropout: float = 0.2) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FrequencyAwareFAViT supports binary classification with one logit.")
        self.vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = int(self.vit.num_features)
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        num_blocks = len(self.vit.blocks)
        if not 0 <= num_unfrozen_blocks <= num_blocks:
            raise ValueError(f"num_unfrozen_blocks must be between 0 and {num_blocks}.")
        for block in self.vit.blocks[num_blocks - num_unfrozen_blocks:] if num_unfrozen_blocks else ():
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.vit.norm.parameters():
            parameter.requires_grad = True

        self.lam_block_indices = tuple(lam_block_indices)
        self.freq_block_indices = tuple(freq_block_indices)
        for name, indices in (("lam_block_indices", self.lam_block_indices),
                              ("freq_block_indices", self.freq_block_indices)):
            invalid = [index for index in indices if not 0 <= index < num_blocks]
            if invalid:
                raise ValueError(f"{name} out of range for {num_blocks} blocks: {invalid}")
        if aux_head_block_index is not None and not 0 <= aux_head_block_index < num_blocks:
            raise ValueError("aux_head_block_index out of range.")

        self.gams = nn.ModuleList(GAM(self.embed_dim, gam_drop_path) for _ in range(num_blocks))
        self.local_extractor, self.freq_extractor = CNNLocalExtractor(self.embed_dim), FrequencyExtractor(self.embed_dim)
        self.lams = nn.ModuleDict({str(i): CrossModalFusion(self.embed_dim) for i in self.lam_block_indices})
        self.freq_fusions = nn.ModuleDict({str(i): CrossModalFusion(self.embed_dim) for i in self.freq_block_indices})
        self.classifier = ClassifierHead(self.embed_dim, head_dropout)
        self.aux_head_block_index = aux_head_block_index
        self.aux_classifier = ClassifierHead(self.embed_dim, head_dropout) if aux_head_block_index is not None else None

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.norm_pre(self.vit.patch_drop(self.vit._pos_embed(self.vit.patch_embed(x))))

    @staticmethod
    def _to_tokens(feature_map: torch.Tensor, side: int) -> torch.Tensor:
        if feature_map.shape[-2:] != (side, side):
            feature_map = F.interpolate(feature_map, (side, side), mode="bilinear", align_corners=False)
        return feature_map.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor):
        tokens = self._embed_tokens(x)
        side = _grid_size(tokens.shape[1] - 1)
        cnn_tokens = self._to_tokens(self.local_extractor(x), side)
        freq_tokens = self._to_tokens(self.freq_extractor(x), side)
        aux_logits = None
        for index, block in enumerate(self.vit.blocks):
            cls_token, patch_tokens = tokens[:, :1], self.gams[index](tokens[:, 1:])
            if index in self.lam_block_indices:
                patch_tokens = self.lams[str(index)](patch_tokens, cnn_tokens)
            if index in self.freq_block_indices:
                patch_tokens = self.freq_fusions[str(index)](patch_tokens, freq_tokens)
            tokens = block(torch.cat((cls_token, patch_tokens), dim=1))
            if self.aux_classifier is not None and index == self.aux_head_block_index:
                aux_logits = self.aux_classifier(self.vit.norm(tokens)[:, 0])
        features = self.vit.norm(tokens)[:, 0]
        return self.classifier(features), features, aux_logits


class PrototypeFALoss(nn.Module):
    def __init__(self, dim: int, margin: float = 0.25, scale: float = 32.0,
                 momentum: float = 0.99) -> None:
        super().__init__()
        self.margin, self.scale, self.momentum = margin, scale, momentum
        self.register_buffer("real_prototype", F.normalize(torch.randn(1, dim), dim=1))
        self.register_buffer("fake_prototype", F.normalize(torch.randn(1, dim), dim=1))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long()
        real, fake = labels == 0, labels == 1
        if not torch.any(real) or not torch.any(fake):
            return features.new_zeros(())
        normalized = F.normalize(features, dim=1)
        if self.training:
            with torch.no_grad():
                for prototype, selected in ((self.real_prototype, real), (self.fake_prototype, fake)):
                    mean = F.normalize(features[selected].mean(0, keepdim=True), dim=1)
                    prototype.copy_(F.normalize(self.momentum * prototype + (1 - self.momentum) * mean, dim=1))
        sim_real = (normalized * self.real_prototype).sum(1)
        sim_fake = (normalized * self.fake_prototype).sum(1)
        separation = ((sim_real[real] - sim_fake[real]).mean() +
                      (sim_fake[fake] - sim_real[fake]).mean()) / 2
        return F.softplus(-self.scale * (separation - self.margin))
