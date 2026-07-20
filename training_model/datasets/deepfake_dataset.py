import inspect
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_class_images(class_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_dir(split_dir: Path, class_name: str) -> Path:
    exact = split_dir / class_name
    if exact.exists():
        return exact
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == class_name:
            return child
    return exact


ORIGIN_DATASET_LABELS: Dict[str, int] = {
    "original": 0,
    "deepfakes": 1,
    "face2face": 1,
    "faceshifter": 1,
    "faceswap": 1,
    "neuraltextures": 1,
}

LEGACY_DATASET_LABELS: Dict[str, int] = {
    "real": 0,
    "fake": 1,
}

CROSS_DATASET_LABELS: Dict[str, int] = {**ORIGIN_DATASET_LABELS, **LEGACY_DATASET_LABELS}


def _gauss_noise():
    signature = inspect.signature(A.GaussNoise)
    if "std_range" in signature.parameters:
        return A.GaussNoise(std_range=(0.01, 0.03), p=0.3)
    return A.GaussNoise(var_limit=(6.5, 58.5), mean=0, p=0.3)


def _image_compression():
    signature = inspect.signature(A.ImageCompression)
    if "quality_range" in signature.parameters:
        return A.ImageCompression(quality_range=(30, 90), p=0.5)
    return A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5)


def get_train_transform(image_size: int = 380) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                p=0.5,
            ),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.4,
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.3,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            _gauss_noise(),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            _image_compression(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_eval_transform(image_size: int = 380) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class DeepfakeFrameDataset(Dataset):
    """Frame-level dataset for FF++ split folders."""

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        dataset_type: str = "ffpp",
        train_transform: Optional[A.Compose] = None,
        eval_transform: Optional[A.Compose] = None,
        original_upsample_factor: Optional[Union[int, str]] = 0,
        train_real_percent: Optional[Union[float, int, str]] = 100,
        seed: int = 42,
        mode: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_type = dataset_type.lower()
        self.mode = (mode or split or "test").lower()
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.original_upsample_factor = self._normalize_upsample_factor(original_upsample_factor)
        self.train_real_percent = self._normalize_percent(train_real_percent)
        self.seed = int(seed)

        self.split_dir = self._resolve_split_dir()
        self.class_to_label = self._get_class_mapping()
        self.samples: List[Tuple[Path, float, bool]] = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No supported images found in {self.split_dir}")

    def _resolve_split_dir(self) -> Path:
        split_dir = self.root_dir / self.split if self.split else self.root_dir
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {split_dir}")
        return split_dir

    def _get_class_mapping(self) -> Dict[str, int]:
        if self.dataset_type in {"ffpp", "origin", "original"}:
            return {**ORIGIN_DATASET_LABELS, **LEGACY_DATASET_LABELS}
        if self.dataset_type in {"celebdf", "cross"}:
            return CROSS_DATASET_LABELS
        raise ValueError("dataset_type must be one of: ffpp, origin, original, celebdf, cross")

    @staticmethod
    def _normalize_upsample_factor(value: Optional[Union[int, str]]) -> int:
        if value in (None, "", "none", "None", "null"):
            return 0
        return max(0, int(value))

    @staticmethod
    def _normalize_percent(value: Optional[Union[float, int, str]]) -> float:
        if value in (None, "", "none", "None", "null"):
            return 100.0
        percent = float(value)
        if not 0.0 <= percent <= 100.0:
            raise ValueError("train_real_percent must be between 0 and 100.")
        return percent

    def _collect_class_images(self, class_dir: Path) -> List[Path]:
        return collect_class_images(class_dir)

    def _select_train_real_samples(
        self,
        class_samples: List[Tuple[Path, float, bool]],
    ) -> List[Tuple[Path, float, bool]]:
        if self.mode != "train" or self.train_real_percent >= 100.0:
            return class_samples
        if self.train_real_percent <= 0.0:
            return []

        keep_count = int(len(class_samples) * self.train_real_percent / 100.0)
        keep_count = max(1, min(len(class_samples), keep_count))
        shuffled_samples = list(class_samples)
        random.Random(self.seed).shuffle(shuffled_samples)
        return sorted(shuffled_samples[:keep_count], key=lambda sample: str(sample[0]))

    def _collect_samples(self) -> List[Tuple[Path, float, bool]]:
        samples: List[Tuple[Path, float, bool]] = []
        original_samples: List[Tuple[Path, float, bool]] = []

        for class_name, label in self.class_to_label.items():
            class_dir = self._find_class_dir(class_name)
            if not class_dir.exists():
                continue

            class_samples = [(path, float(label), False) for path in self._collect_class_images(class_dir)]
            if label == 0:
                class_samples = self._select_train_real_samples(class_samples)
                original_samples.extend(class_samples)

            samples.extend(class_samples)

        if self.mode == "train" and self.original_upsample_factor > 0:
            for _ in range(self.original_upsample_factor):
                samples.extend((path, label, True) for path, label, _ in original_samples)

        return samples

    def _find_class_dir(self, class_name: str) -> Path:
        return find_class_dir(self.split_dir, class_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, augmented = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self._select_transform(label=label, augmented=augmented)
        if transform is not None:
            image = transform(image=image)["image"]

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor, str(image_path)

    def _select_transform(self, label: float, augmented: bool):
        if self.mode == "train":
            return self.train_transform or self.eval_transform
        return self.eval_transform


class GANFrameDataset(Dataset):
    """Frame-level dataset for GAN fake and real image folders."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: Optional[str] = None,
        fake_dir: Optional[str] = None,
        real_dir: Optional[str] = None,
        train_transform: Optional[A.Compose] = None,
        eval_transform: Optional[A.Compose] = None,
        mode: Optional[str] = None,
        label: float = 1.0,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.split = split
        self.fake_dir = Path(fake_dir) if fake_dir is not None else None
        self.real_dir = Path(real_dir) if real_dir is not None else None
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.mode = (mode or split or "train").lower()
        self.label = float(label)

        self.sources = self._resolve_sources()
        self.samples: List[Tuple[Path, float]] = self._collect_samples()

        if not self.samples:
            source_paths = ", ".join(str(source_dir) for source_dir, _ in self.sources)
            raise RuntimeError(f"No supported GAN images found in: {source_paths}")

    def _resolve_image_dir(self, image_dir: Path, description: str) -> Path:
        split_dir = image_dir / self.split if self.split else image_dir
        if not split_dir.exists():
            raise FileNotFoundError(f"{description} folder does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"{description} path is not a directory: {split_dir}")
        return split_dir

    def _resolve_sources(self) -> List[Tuple[Path, float]]:
        if self.fake_dir is not None or self.real_dir is not None:
            if self.fake_dir is None or self.real_dir is None:
                raise ValueError("Set both fake_dir and real_dir for GANFrameDataset.")
            return [
                (self._resolve_image_dir(self.fake_dir, "GAN fake"), 1.0),
                (self._resolve_image_dir(self.real_dir, "GAN real"), 0.0),
            ]

        if self.root_dir is None:
            raise ValueError("Set either root_dir or both fake_dir and real_dir for GANFrameDataset.")
        return [(self._resolve_image_dir(self.root_dir, "GAN dataset"), self.label)]

    def _collect_class_images(self, class_dir: Path) -> List[Path]:
        return collect_class_images(class_dir)

    def _collect_samples(self) -> List[Tuple[Path, float]]:
        samples: List[Tuple[Path, float]] = []
        for image_dir, label in self.sources:
            samples.extend((path, label) for path in self._collect_class_images(image_dir))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read GAN image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self.train_transform if self.mode == "train" else self.eval_transform
        transform = transform or self.eval_transform
        if transform is not None:
            image = transform(image=image)["image"]

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor, str(image_path)
