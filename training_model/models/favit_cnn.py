import math
from typing import Dict, Iterable, Optional, Tuple

import timm
import torch
from torch import nn
from torch.nn import functional as F


def _grid_size(num_tokens: int) -> int:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected square patch token count, got {num_tokens}.")
    return side


def _group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(groups, channels) or 1, channels)


def _srm_kernels() -> torch.Tensor:
    k1 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4.0
    k2 = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12.0
    k3 = torch.zeros(5, 5, dtype=torch.float32)
    k3[1:4, 1:4] = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    return torch.stack((k1, k2, k3))


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
        delta_tokens = self.net(feature_map).flatten(2).transpose(1, 2)
        return patch_tokens + delta_tokens


class LearnableRadialFFT(nn.Module):
    """Differentiable 2D FFT spectral decomposition with learnable radial band weights."""

    def __init__(self, num_bands: int = 3) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.band_weights = nn.Parameter(torch.ones(num_bands, 3))
        self.band_centers = nn.Parameter(torch.linspace(0.15, 0.85, num_bands))
        self.band_sharpness = nn.Parameter(torch.full((num_bands,), 8.0))
        self._cached_radius: Optional[torch.Tensor] = None
        self._cached_shape: Optional[Tuple[int, int]] = None

    def _radial_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if (
            self._cached_radius is not None
            and self._cached_shape == (height, width)
            and self._cached_radius.device == device
        ):
            return self._cached_radius
        fy = torch.fft.fftfreq(height, d=1.0, device=device)
        fx = torch.fft.rfftfreq(width, d=1.0, device=device)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(grid_y ** 2 + grid_x ** 2)
        radius = radius / (radius.max().clamp(min=1e-6))
        self._cached_radius = radius
        self._cached_shape = (height, width)
        return radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        height, width = x.shape[-2:]
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            fft_coeff = torch.fft.rfft2(x_fp32, norm="ortho")
            radius = self._radial_grid(height, width, x.device)

            # Soft radial band masks sum to a learnable spectral filter.
            centers = self.band_centers.sigmoid().view(self.num_bands, 1, 1)
            sharpness = F.softplus(self.band_sharpness).view(self.num_bands, 1, 1)
            masks = torch.exp(-sharpness * (radius.unsqueeze(0) - centers) ** 2)
            masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-6)

            weighted_mask = torch.einsum("bhw,bc->chw", masks, F.softplus(self.band_weights))
            filtered_fft = fft_coeff * weighted_mask.unsqueeze(0)
            reconstructed = torch.fft.irfft2(filtered_fft, s=(height, width), norm="ortho")
        return reconstructed.to(orig_dtype)


class DualDomainFrequencyExtractor(nn.Module):
    """Extracts complementary local RGB, SRM noise residual, and learnable FFT features."""

    def __init__(self, out_channels: int, use_learnable_fft: bool = True, use_freq: bool = True) -> None:
        super().__init__()
        self.use_freq = use_freq
        self.use_learnable_fft = use_learnable_fft and use_freq

        in_channels = 3
        if use_freq:
            self.register_buffer("srm_weight", _srm_kernels().unsqueeze(1).repeat(1, 3, 1, 1))
            in_channels += 3
        if self.use_learnable_fft:
            self.fft_filter = LearnableRadialFFT(num_bands=3)
            in_channels += 3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            _group_norm(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            _group_norm(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            _group_norm(256),
            nn.GELU(),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        streams = [x]
        if self.use_freq:
            srm_res = F.conv2d(x, self.srm_weight.to(dtype=x.dtype), padding=2)
            streams.append(srm_res)
        if self.use_learnable_fft:
            fft_res = self.fft_filter(x)
            streams.append(fft_res)
        return self.stem(torch.cat(streams, dim=1))


class BidirectionalGatedFusion(nn.Module):
    """Bidirectional gated cross-attention between ViT patch tokens and CNN/freq tokens."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Freq/Local -> Spatial cross-attention
        self.q_s = nn.Linear(dim, dim)
        self.k_f = nn.Linear(dim, dim)
        self.v_f = nn.Linear(dim, dim)
        self.proj_f2s = nn.Linear(dim, dim)
        self.beta_f2s = nn.Parameter(torch.zeros(1))

        # Spatial -> Freq/Local semantic gating
        self.spatial_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        self.beta_s2f = nn.Parameter(torch.zeros(1))

    def _to_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tokens.shape
        return tokens.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        freq_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = patch_tokens.shape
        q = self._to_heads(self.q_s(patch_tokens))
        k = self._to_heads(self.k_f(freq_tokens))
        v = self._to_heads(self.v_f(freq_tokens))

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        f2s = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        updated_patch = patch_tokens + self.beta_f2s * self.proj_f2s(f2s)

        gate = self.spatial_gate(updated_patch)
        updated_freq = freq_tokens + self.beta_s2f * (freq_tokens * gate)
        return updated_patch, updated_freq


class FAViTCNN(nn.Module):
    """Upgraded Unified FA-ViT-CNN (CLIP/DINOv2 + Learnable FFT/SRM + Bidirectional Gated Fusion)."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224",
        pretrained: bool = True,
        num_classes: int = 1,
        freq_in_channels: int = 3,
        freq_dim: int = 128,
        use_freq: bool = True,
        use_cnn_branch: bool = True,
        use_learnable_fft: bool = True,
        use_bidirectional_fusion: bool = True,
        use_aux_head: bool = True,
        tune_vit_layernorm: bool = True,
        fusion_block_indices: Iterable[int] = (2, 5, 8, 11),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FAViTCNN is configured for binary classification with one logit.")

        self.backbone_name = backbone_name
        self.use_freq = use_freq
        self.use_cnn_branch = use_cnn_branch
        self.use_learnable_fft = use_learnable_fft
        self.use_bidirectional_fusion = use_bidirectional_fusion and use_cnn_branch
        self.use_aux_head = use_aux_head

        self.vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.vit_dim = int(self.vit.num_features)
        self.rgb_norm = None if hasattr(self.vit, "norm") else nn.LayerNorm(self.vit_dim)

        # Freeze ViT attention/MLP blocks; optionally tune LayerNorms (PEFT).
        for name, param in self.vit.named_parameters():
            if tune_vit_layernorm and ("norm" in name or "ln" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        num_blocks = len(self.vit.blocks)
        self.fusion_block_indices = tuple(idx for idx in fusion_block_indices if 0 <= idx < num_blocks)
        if not self.fusion_block_indices:
            self.fusion_block_indices = (0, min(3, num_blocks - 1), min(6, num_blocks - 1))

        self.gam_blocks = nn.ModuleList(GAM(self.vit_dim) for _ in range(num_blocks))
        self.freq_extractor = (
            DualDomainFrequencyExtractor(
                out_channels=self.vit_dim,
                use_learnable_fft=use_learnable_fft,
                use_freq=use_freq,
            )
            if use_cnn_branch
            else None
        )
        self.fusions = nn.ModuleDict(
            {str(idx): BidirectionalGatedFusion(self.vit_dim, num_heads=8) for idx in self.fusion_block_indices}
            if use_cnn_branch
            else {}
        )

        self.freq_proj = (
            nn.Sequential(
                nn.LayerNorm(self.vit_dim),
                nn.Linear(self.vit_dim, freq_dim),
                nn.GELU(),
            )
            if use_cnn_branch
            else None
        )
        joint_dim = self.vit_dim + freq_dim
        self.joint_proj = (
            nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, self.vit_dim),
            )
            if use_cnn_branch
            else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit_dim),
            nn.Dropout(dropout),
            nn.Linear(self.vit_dim, num_classes),
        )
        self.aux_patch_head = (
            nn.Sequential(
                nn.LayerNorm(self.vit_dim),
                nn.Linear(self.vit_dim, 1),
            )
            if use_aux_head
            else None
        )
        self.last_patch_anomaly_map: Optional[torch.Tensor] = None

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        if hasattr(self.vit, "patch_drop"):
            x = self.vit.patch_drop(x)
        if hasattr(self.vit, "norm_pre"):
            x = self.vit.norm_pre(x)
        return x

    @staticmethod
    def _to_tokens(feature_map: torch.Tensor, side: int) -> torch.Tensor:
        if feature_map.shape[-2:] != (side, side):
            feature_map = F.interpolate(feature_map, size=(side, side), mode="bilinear", align_corners=False)
        return feature_map.flatten(2).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        freq_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        tokens = self._embed_tokens(x)
        num_prefix = tokens.shape[1] - int(math.isqrt(tokens.shape[1] - 1) ** 2)
        side = _grid_size(tokens.shape[1] - num_prefix)

        freq_input = x if freq_x is None else freq_x
        freq_tokens = (
            self._to_tokens(self.freq_extractor(freq_input), side)
            if self.freq_extractor is not None
            else None
        )

        for idx, block in enumerate(self.vit.blocks):
            prefix_tokens = tokens[:, :num_prefix]
            patch_tokens = self.gam_blocks[idx](tokens[:, num_prefix:])

            if self.use_bidirectional_fusion and freq_tokens is not None and idx in self.fusion_block_indices:
                patch_tokens, freq_tokens = self.fusions[str(idx)](patch_tokens, freq_tokens)

            tokens = torch.cat((prefix_tokens, patch_tokens), dim=1)
            tokens = block(tokens)

        norm = self.vit.norm if hasattr(self.vit, "norm") else self.rgb_norm
        tokens = norm(tokens)

        cls_feature = tokens[:, 0]
        final_patches = tokens[:, num_prefix:]

        aux_logits = None
        if self.aux_patch_head is not None:
            patch_scores = self.aux_patch_head(final_patches).squeeze(-1)
            self.last_patch_anomaly_map = patch_scores.detach().reshape(x.shape[0], side, side)
            aux_logits = patch_scores.mean(dim=1, keepdim=True)

        if self.freq_proj is not None and self.joint_proj is not None and freq_tokens is not None:
            freq_pooled = self.freq_proj(freq_tokens.mean(dim=1))
            joint_feature = self.joint_proj(torch.cat((cls_feature, freq_pooled), dim=1))
        else:
            joint_feature = cls_feature
        # FALoss uses the same feature space as the final Linear weight,
        # before stochastic dropout is applied.
        normed_feature = self.classifier[0](joint_feature)
        logits = self.classifier[2](self.classifier[1](normed_feature))
        return logits, normed_feature, aux_logits


UnifiedFAViT = FAViTCNN


class FALoss(nn.Module):
    """Hyperspherical Forgery-Aware Loss using L2-normalized features and real prototype."""

    def __init__(self, margin: float = 0.25, scale: float = 32.0, real_compact_weight: float = 0.5) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.real_compact_weight = real_compact_weight

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

        features_norm = F.normalize(features.float(), dim=1)
        # Since real=0 and fake=1, classifier_weight[0] points toward fake; -classifier_weight[0] is the real prototype.
        prototype = -F.normalize(classifier_weight[0].float().view(1, -1), dim=1)
        similarities = (features_norm * prototype).sum(dim=1)

        sp = similarities[real_mask].mean()
        sn = similarities[fake_mask].mean()
        margin_loss = F.softplus(self.scale * (sn - sp + self.margin))
        compact_loss = (1.0 - similarities[real_mask]).mean()
        return margin_loss + self.real_compact_weight * compact_loss


class SpectralConsistencyLoss(nn.Module):
    """Enforces feature & logit consistency under mild compression/blur perturbations."""

    def __init__(self, feat_weight: float = 1.0, kl_weight: float = 0.5) -> None:
        super().__init__()
        self.feat_weight = feat_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        feat_clean: torch.Tensor,
        feat_perturbed: torch.Tensor,
        logits_clean: torch.Tensor,
        logits_perturbed: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(
            F.normalize(feat_clean.float(), dim=1),
            F.normalize(feat_perturbed.float(), dim=1),
            dim=1,
        )
        feat_loss = (1.0 - cos_sim).mean()

        p_clean = torch.sigmoid(logits_clean.float().view(-1)).clamp(1e-5, 1.0 - 1e-5)
        p_pert = torch.sigmoid(logits_perturbed.float().view(-1)).clamp(1e-5, 1.0 - 1e-5)
        kl_1 = p_clean * torch.log(p_clean / p_pert) + (1.0 - p_clean) * torch.log((1.0 - p_clean) / (1.0 - p_pert))
        kl_2 = p_pert * torch.log(p_pert / p_clean) + (1.0 - p_pert) * torch.log((1.0 - p_pert) / (1.0 - p_clean))
        kl_loss = 0.5 * (kl_1 + kl_2).mean()
        return self.feat_weight * feat_loss + self.kl_weight * kl_loss


def perturb_batch_for_debiasing(images: torch.Tensor) -> torch.Tensor:
    """Apply fast differentiable downscale-upscale + mild Gaussian noise to simulate compression."""
    batch_size, channels, height, width = images.shape
    scale = 0.5 if batch_size % 2 == 0 else 0.75
    low_h = max(int(height * scale), 32)
    low_w = max(int(width * scale), 32)
    degraded = F.interpolate(images, size=(low_h, low_w), mode="bilinear", align_corners=False)
    degraded = F.interpolate(degraded, size=(height, width), mode="bilinear", align_corners=False)
    noise = torch.randn_like(degraded) * 0.02
    return degraded + noise


# Preserve symbol compatibility if anything imports CNN_feature_extractor_branch
CNN_feature_extractor_branch = DualDomainFrequencyExtractor
