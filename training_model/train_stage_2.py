import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from datasets import (
    DeepfakeFrameDataset,
    find_class_dir,
    get_eval_transform,
    get_train_transform,
)
from train import load_config, resolve_device, run_training_loop, set_seed
from utils.checkpoint import load_checkpoint


ALLOWED_STAGE_TWO_BACKBONES = ("redesigned_favit", "favit_cnn")
RESUME_CHECKPOINT_FIELDS = (
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_auc",
    "best_accuracy",
    "config",
    "epochs_without_improvement",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a supported FA-ViT backbone on WildDeepfake"
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


def validate_stage_two_backbone(config: Dict) -> str:
    if not config.get("backbone"):
        raise KeyError("Missing required config key 'backbone' for stage-two training")
    backbone = str(config["backbone"])
    if backbone not in ALLOWED_STAGE_TWO_BACKBONES:
        supported = ", ".join(ALLOWED_STAGE_TWO_BACKBONES)
        raise ValueError(
            f"Unsupported stage-two backbone '{backbone}'. Supported values: {supported}"
        )
    return backbone


def require_directory(config: Dict, key: str) -> Path:
    value = config.get(key)
    if value in (None, ""):
        raise KeyError(f"Missing required config key '{key}'")
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config key '{key}' points to a missing path: {path}")
    if not path.is_dir():
        raise NotADirectoryError(
            f"Config key '{key}' must point to a directory, but found: {path}"
        )
    return path


def require_split_name(config: Dict, key: str, default: str) -> str:
    value = str(config.get(key, default)).strip()
    if not value:
        raise ValueError(f"Config key '{key}' must name a dataset directory")
    return value


def validate_binary_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description} directory: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} path is not a directory: {path}")

    missing_classes = [
        class_name
        for class_name in ("real", "fake")
        if not find_class_dir(path, class_name).is_dir()
    ]
    if missing_classes:
        missing = ", ".join(missing_classes)
        raise FileNotFoundError(
            f"{description} directory '{path}' is missing class folder(s): {missing}"
        )


def get_stage_two_data_root(config: Dict) -> Path:
    return require_directory(config, "data_root")


def build_stage_two_train_loader(config: Dict) -> DataLoader:
    root = get_stage_two_data_root(config)
    split = require_split_name(config, "train_dir", "train")
    split_path = root / split
    validate_binary_directory(split_path, "WildDeepfake training")

    image_size = int(config["image_size"])
    train_dataset = DeepfakeFrameDataset(
        root_dir=str(root),
        split=split,
        dataset_type="cross",
        train_transform=get_train_transform(image_size),
        eval_transform=get_eval_transform(image_size),
        original_upsample_factor=config.get("original_upsample_factor"),
        train_real_percent=config.get("train_real_percent", 100),
        seed=int(config.get("seed", 42)),
        mode="train",
    )
    return DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    train_loader = build_stage_two_train_loader(config)
    root = get_stage_two_data_root(config)
    split = require_split_name(config, "val_dir", "val")
    split_path = root / split
    validate_binary_directory(split_path, "WildDeepfake validation")

    eval_transform = get_eval_transform(int(config["image_size"]))
    val_dataset = DeepfakeFrameDataset(
        root_dir=str(root),
        split=split,
        dataset_type="cross",
        train_transform=None,
        eval_transform=eval_transform,
        original_upsample_factor=0,
        mode="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def load_validated_resume_checkpoint(
    checkpoint_path: str,
    requested_backbone: str,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    metadata = checkpoint.get("config")
    saved_backbone = metadata.get("backbone") if isinstance(metadata, dict) else None
    if not saved_backbone:
        raise ValueError(
            f"The resume checkpoint '{checkpoint_path}' is missing "
            "metadata 'config.backbone', so its model architecture cannot be safely "
            "identified. Use a checkpoint created by this repository's checkpoint helper."
        )
    if str(saved_backbone) != requested_backbone:
        raise ValueError(
            f"Backbone mismatch in resume checkpoint '{checkpoint_path}': "
            f"checkpoint config.backbone='{saved_backbone}', requested "
            f"config['backbone']='{requested_backbone}'."
        )

    missing_fields = [
        field for field in RESUME_CHECKPOINT_FIELDS if field not in checkpoint
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"The resume checkpoint '{checkpoint_path}' is incomplete; "
            f"missing field(s): {missing}."
        )
    return checkpoint


def resolve_resume_checkpoint(
    config: Dict,
    requested_backbone: str,
    device: torch.device,
    resume_arg: Optional[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    resume_path = resume_arg or config.get("resume_from")
    if not resume_path:
        return None, None
    resume_path = str(resume_path)
    resume_source = "--resume" if resume_arg else "config key 'resume_from'"
    if not Path(resume_path).expanduser().is_file():
        raise FileNotFoundError(
            f"Stage-two training requires an existing checkpoint, but "
            f"{resume_source} points to: {resume_path}"
        )
    checkpoint = load_validated_resume_checkpoint(
        resume_path,
        requested_backbone,
        device,
    )
    return resume_path, checkpoint


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    backbone = validate_stage_two_backbone(config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "cuda")))
    train_loader, val_loader = build_loaders(config)
    if args.no_resume and args.resume:
        raise ValueError("--resume and --no-resume cannot be used together.")

    if args.no_resume:
        resume_path, checkpoint = None, None
        print("Starting a new stage-two training run.")
    else:
        resume_path, checkpoint = resolve_resume_checkpoint(
            config=config,
            requested_backbone=backbone,
            device=device,
            resume_arg=args.resume,
        )
        if resume_path is None:
            print("Starting a new stage-two training run.")

    save_dir = Path(config.get("stage_2_save_dir", "checkpoints_stage_2"))
    checkpoint_path = save_dir / str(
        config.get("stage_2_checkpoint_name", "best_stage_2.pth")
    )
    print(f"WildDeepfake train samples: {len(train_loader.dataset)}")
    print(f"WildDeepfake val samples: {len(val_loader.dataset)}")
    run_training_loop(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        resume_path=resume_path,
        checkpoint_data=checkpoint,
        max_steps=args.max_steps,
        log_parameter_counts=True,
    )


if __name__ == "__main__":
    main()
