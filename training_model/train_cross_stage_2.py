import argparse
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader

from datasets import DeepfakeFrameDataset, get_eval_transform
from train import load_config, resolve_device, set_seed
from train_cross import run_training_loop
from train_stage_2 import (
    build_stage_two_train_loader,
    positive_int,
    require_directory,
    resolve_resume_checkpoint,
    validate_binary_directory,
    validate_stage_two_backbone,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a supported FA-ViT backbone on WildDeepfake and validate "
            "on the configured cross dataset"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint used to resume complete training state",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a new training run and ignore config key 'resume_from'",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Limit train and validation batches per epoch (verification only)",
    )
    return parser.parse_args()


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    train_loader = build_stage_two_train_loader(config)
    cross_root = require_directory(config, "cross_dataset_root")
    validate_binary_directory(cross_root, "cross-dataset validation")

    cross_dataset = DeepfakeFrameDataset(
        root_dir=str(cross_root),
        split=None,
        dataset_type="cross",
        train_transform=None,
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=0,
        mode="val",
    )
    cross_loader = DataLoader(
        cross_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, cross_loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    backbone = validate_stage_two_backbone(config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "cuda")))
    train_loader, cross_loader = build_loaders(config)
    if args.no_resume and args.resume:
        raise ValueError("--resume and --no-resume cannot be used together.")

    if args.no_resume:
        resume_path, checkpoint = None, None
        print("Starting a new cross stage-two training run.")
    else:
        resume_path, checkpoint = resolve_resume_checkpoint(
            config=config,
            requested_backbone=backbone,
            device=device,
            resume_arg=args.resume,
        )
        if resume_path is None:
            print("Starting a new cross stage-two training run.")

    save_dir = Path(
        config.get("cross_stage_2_save_dir", "checkpoints_cross_stage_2")
    )
    checkpoint_path = save_dir / str(
        config.get("cross_stage_2_checkpoint_name", "best_cross_stage_2.pth")
    )
    print(f"WildDeepfake train samples: {len(train_loader.dataset)}")
    print(f"Cross validation samples: {len(cross_loader.dataset)}")
    run_training_loop(
        config=config,
        train_loader=train_loader,
        val_loader=cross_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        resume_path=resume_path,
        checkpoint_data=checkpoint,
        max_steps=args.max_steps,
        log_parameter_counts=True,
    )


if __name__ == "__main__":
    main()
