import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from datasets import DeepfakeFrameDataset, get_eval_transform
from models import build_model
from train import load_config, resolve_device
from utils.checkpoint import load_checkpoint
from utils.inference import predict, predict_tta
from utils.metrics import binary_confusion_matrix, compute_binary_metrics, format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Test a configurable backbone on the origin dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint path")
    parser.add_argument("--output-csv", type=str, default="origin_dataset_predictions.csv", help="CSV output path")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (hflip + rotate ±5°)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(config.get("threshold", 0.5))

    dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["test_dir"],
        dataset_type="ffpp",
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=0,
        mode="test",
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
    image_paths, labels, probs = run_predict(model, loader, device, desc="Test origin dataset")
    metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    cm = binary_confusion_matrix(labels, probs, threshold=threshold)
    preds = [int(prob >= threshold) for prob in probs]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True) if output_csv.parent != Path(".") else None
    pd.DataFrame(
        {
            "image_path": image_paths,
            "label": labels,
            "probability": probs,
            "prediction": preds,
        }
    ).to_csv(output_csv, index=False)

    print(f"Origin Dataset Test | {format_metrics(metrics)}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
