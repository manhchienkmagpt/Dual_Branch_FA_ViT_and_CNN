import argparse
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import ConcatDataset, DataLoader

from datasets import DeepfakeFrameDataset, GANFrameDataset, get_eval_transform, get_train_transform
from train import load_config, resolve_device, run_training_loop
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a configurable backbone with origin and GAN data")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


def _required_gan_dir(config: Dict, key: str) -> str:
    value = config.get(key)
    if value in (None, "", "null"):
        raise ValueError(f"Missing {key} in config.")
    return str(value)


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    ffpp_train_dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["train_dir"],
        dataset_type="ffpp",
        train_transform=train_transform,
        eval_transform=eval_transform,
        original_upsample_factor=config.get("original_upsample_factor"),
        train_real_percent=config.get("train_real_percent", 100),
        seed=int(config.get("seed", 42)),
        mode="train",
    )
    gan_train_dataset = GANFrameDataset(
        fake_dir=_required_gan_dir(config, "gan_fake_dir"),
        real_dir=_required_gan_dir(config, "gan_real_dir"),
        train_transform=train_transform,
        eval_transform=eval_transform,
        mode="train",
    )
    val_dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["val_dir"],
        dataset_type="ffpp",
        train_transform=None,
        eval_transform=eval_transform,
        original_upsample_factor=0,
        mode="val",
    )

    train_dataset = ConcatDataset([ffpp_train_dataset, gan_train_dataset])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
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


def main():
    args = parse_args()
    config = load_config(args.config)
    if not config.get("gan_fake_dir") or not config.get("gan_real_dir"):
        raise ValueError("Set gan_fake_dir and gan_real_dir in config.")

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "cuda")))
    save_dir = Path(config.get("gan_save_dir") or config.get("save_dir", "checkpoints"))
    checkpoint_path = save_dir / str(config.get("gan_checkpoint_name", "best_model_with_gan.pth"))

    train_loader, val_loader = build_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    run_training_loop(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        resume_path=args.resume or config.get("resume_from"),
    )


if __name__ == "__main__":
    main()
