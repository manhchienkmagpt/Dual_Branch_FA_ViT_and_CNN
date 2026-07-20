import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import DataLoader

from datasets import DeepfakeFrameDataset, get_eval_transform
from models import build_model
from train import load_config, resolve_device
from utils.checkpoint import load_checkpoint
from utils.inference import predict, predict_tta
from utils.metrics import binary_confusion_matrix, compute_binary_metrics, format_metrics


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NestedCrossDataset(DeepfakeFrameDataset):
    """Cross dataset with layout: root/{real,fake}/subfolder/frames."""

    def __init__(self, *args, data_percent: float = 100.0, **kwargs):
        self.data_percent = data_percent
        super().__init__(*args, **kwargs)

    def _collect_class_images(self, class_dir: Path) -> List[Path]:
        image_paths: List[Path] = []

        for subfolder in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            image_paths.extend(
                path
                for path in subfolder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

        image_paths = sorted(image_paths)
        if self.data_percent >= 100.0 or not image_paths:
            return image_paths

        keep_count = max(1, math.ceil(len(image_paths) * self.data_percent / 100.0))
        return image_paths[-keep_count:]


def percent(value: str) -> float:
    parsed_value = float(value)
    if not 0.0 < parsed_value <= 100.0:
        raise argparse.ArgumentTypeError("percentage must be greater than 0 and at most 100")
    return parsed_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a configurable backbone on a nested cross dataset"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint path"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="cross_dataset_2_predictions.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--tta", action="store_true", help="Enable test-time augmentation (hflip + rotate +/-5 degrees)"
    )
    parser.add_argument(
        "--data-percent",
        type=percent,
        default=100.0,
        help="Percentage of images to use from the end of each class (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(config.get("threshold", 0.5))

    dataset = NestedCrossDataset(
        root_dir=config["cross_dataset_root"],
        split=None,
        dataset_type="cross",
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=0,
        mode="test",
        data_percent=args.data_percent,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )

    model = build_model(
        backbone=str(config.get("backbone", "efficientnetb4")),
        pretrained=False,
        dropout=float(config.get("dropout", 0.4)),
        image_size=int(config["image_size"]),
        freq_in_channels=int(config.get("freq_in_channels", 3)),
        freq_dim=int(config.get("freq_dim", 128)),
        use_freq=bool(config.get("use_freq", True)),
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    run_predict = predict_tta if args.tta else predict
    image_paths, labels, probs = run_predict(model, loader, device, desc="Test cross dataset 2")
    metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    cm = binary_confusion_matrix(labels, probs, threshold=threshold)
    preds = [int(prob >= threshold) for prob in probs]

    output_csv = Path(args.output_csv)
    if output_csv.parent != Path("."):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "image_path": image_paths,
            "label": labels,
            "probability": probs,
            "prediction": preds,
        }
    ).to_csv(output_csv, index=False)

    print(f"Using the last {args.data_percent:g}% of each class ({len(dataset)} images)")
    print(f"Cross Dataset 2 Test | {format_metrics(metrics)}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
