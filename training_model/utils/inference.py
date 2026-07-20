from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from train import unpack_model_output


def predict(model, loader, device, desc: str = "Test") -> Tuple[List[str], List[float], List[float]]:
    model.eval()
    image_paths: List[str] = []
    labels_all: List[float] = []
    probs_all: List[float] = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            logits, _, _ = unpack_model_output(model(images))
            probs = torch.sigmoid(logits.view(-1))
            image_paths.extend(paths)
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    return image_paths, labels_all, probs_all


def predict_tta(model, loader, device, desc: str = "Test TTA") -> Tuple[List[str], List[float], List[float]]:
    """Inference with test-time augmentation: average over original + hflip + rotate ±5°."""
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: TF.rotate(x, 5),
        lambda x: TF.rotate(x, -5),
    ]

    model.eval()
    image_paths: List[str] = []
    labels_all: List[float] = []
    probs_all: List[float] = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            aug_probs = torch.stack([
                torch.sigmoid(unpack_model_output(model(aug(images)))[0].view(-1))
                for aug in tta_transforms
            ]).mean(0)
            image_paths.extend(paths)
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(aug_probs.cpu().numpy().tolist())

    return image_paths, labels_all, probs_all
