import timm
from torch import nn

from .favit_cnn import FAViTCNN
from .frequency_aware_favit import FrequencyAwareFAViT
from .redesigned_favit import RedesignedFAViT


class TimmBackbone(nn.Module):
    """Generic timm backbone with a dropout classifier head."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dropout: float = 0.4,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        model_kwargs = {}
        if image_size is not None and model_name.startswith("swin_"):
            model_kwargs["img_size"] = image_size

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **model_kwargs,
        )
        self.feature_channels = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_channels, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(1)


class EfficientNetB4(TimmBackbone):
    """Plain EfficientNet-B4 backbone with a dropout classifier head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.4) -> None:
        super().__init__(
            model_name="efficientnet_b4",
            pretrained=pretrained,
            dropout=dropout,
        )


class SwinTransformerSmall(TimmBackbone):
    """Swin Transformer Small backbone with a dropout classifier head."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.4,
        image_size: int | None = None,
    ) -> None:
        super().__init__(
            model_name="swin_small_patch4_window7_224",
            pretrained=pretrained,
            dropout=dropout,
            image_size=image_size,
        )


_BACKBONE_ALIASES = {
    "efficientnet_b4": "efficientnetb4",
    "efficientnet-b4": "efficientnetb4",
    "efficientnetb4": "efficientnetb4",
    "resnet50": "resnet50",
    "resnet_50": "resnet50",
    "swin_tiny": "swin_tiny",
    "swin-tiny": "swin_tiny",
    "swin_t": "swin_tiny",
    "swin-t": "swin_tiny",
    "swin_small": "swin_small",
    "swin-small": "swin_small",
    "swin_s": "swin_small",
    "swin-s": "swin_small",
    "redesigned_favit": "redesigned_favit",
    "redesigned-favit": "redesigned_favit",
    "favit": "redesigned_favit",
    "favit_cnn": "favit_cnn",
    "favit-cnn": "favit_cnn",
    "frequency_aware_favit": "frequency_aware_favit",
    "frequency-aware-favit": "frequency_aware_favit",
    "fa_vit_v2": "frequency_aware_favit",
}

_TIMM_BACKBONES = {
    "resnet50": "resnet50",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def normalize_backbone_name(backbone: str) -> str:
    key = backbone.strip().lower()
    if key not in _BACKBONE_ALIASES:
        valid_backbones = ", ".join(sorted(_BACKBONE_ALIASES))
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose one of: {valid_backbones}")
    return _BACKBONE_ALIASES[key]


def build_model(
    backbone: str = "efficientnetb4",
    pretrained: bool = True,
    dropout: float = 0.4,
    image_size: int | None = None,
    freq_in_channels: int = 3,
    freq_dim: int = 128,
    use_freq: bool = True,
) -> nn.Module:
    backbone_name = normalize_backbone_name(backbone)
    if backbone_name == "efficientnetb4":
        return EfficientNetB4(pretrained=pretrained, dropout=dropout)
    if backbone_name == "swin_small":
        return SwinTransformerSmall(
            pretrained=pretrained,
            dropout=dropout,
            image_size=image_size,
        )
    if backbone_name == "redesigned_favit":
        return RedesignedFAViT(pretrained=pretrained, num_classes=1)
    if backbone_name == "frequency_aware_favit":
        return FrequencyAwareFAViT(pretrained=pretrained, num_classes=1)
    if backbone_name == "favit_cnn":
        return FAViTCNN(
            pretrained=pretrained,
            num_classes=1,
            freq_in_channels=freq_in_channels,
            freq_dim=freq_dim,
            use_freq=use_freq,
        )
    return TimmBackbone(
        model_name=_TIMM_BACKBONES[backbone_name],
        pretrained=pretrained,
        dropout=dropout,
        image_size=image_size,
    )
