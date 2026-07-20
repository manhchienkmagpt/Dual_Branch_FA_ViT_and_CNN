import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets import DeepfakeFrameDataset, get_eval_transform, get_train_transform
from models import build_model
from train import (
    count_model_parameters,
    create_fa_criterion,
    current_lrs,
    format_metrics,
    load_config,
    lr_reduced,
    load_model_state_strict,
    parse_args,
    resolve_device,
    resolve_pos_weight,
    run_one_epoch,
    set_seed,
)
from utils.checkpoint import load_checkpoint, save_checkpoint


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    train_dataset = DeepfakeFrameDataset(
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
    cross_dataset = DeepfakeFrameDataset(
        root_dir=config["cross_dataset_root"],
        split=None,
        dataset_type="cross",
        train_transform=None,
        eval_transform=eval_transform,
        original_upsample_factor=0,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
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


def run_training_loop(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    device: torch.device,
    resume_path: Optional[str] = None,
    checkpoint_data: Optional[Dict[str, Any]] = None,
    max_steps: Optional[int] = None,
    log_parameter_counts: bool = False,
) -> None:
    if checkpoint_data is not None and not resume_path:
        raise ValueError("checkpoint_data requires resume_path")

    active_checkpoint = checkpoint_data
    if active_checkpoint is None and resume_path:
        active_checkpoint = load_checkpoint(str(resume_path), device)

    threshold = float(config.get("threshold", 0.5))
    backbone = str(config.get("backbone", "efficientnetb4"))
    print(f"Backbone: {backbone}")
    model = build_model(
        backbone=backbone,
        pretrained=bool(config.get("pretrained", True)),
        dropout=float(config.get("dropout", 0.4)),
        image_size=int(config["image_size"]),
        freq_in_channels=int(config.get("freq_in_channels", 3)),
        freq_dim=int(config.get("freq_dim", 128)),
        use_freq=bool(config.get("use_freq", True)),
    ).to(device)

    if log_parameter_counts:
        trainable_parameters, total_parameters = count_model_parameters(model)
        print(
            "Parameters: "
            f"{trainable_parameters:,} trainable / {total_parameters:,} total"
        )

    label_smoothing = float(config.get("label_smoothing", 0.0))
    pw = resolve_pos_weight(config, train_loader.dataset, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    lambda_fal = float(config.get("lambda_fal", 0.0))
    fa_criterion = create_fa_criterion(model, config, device, lambda_fal)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler_patience = int(config.get("lr_scheduler_patience", 2))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_scheduler_factor", 0.5)),
        patience=max(0, scheduler_patience - 1),
        min_lr=float(config.get("min_lr", 1e-7)),
    )

    best_auc = -math.inf
    best_accuracy = -math.inf
    epochs_without_auc_improvement = 0
    early_stopping_patience = int(config.get("early_stopping_patience", 5))
    total_epochs = int(config.get("epochs", 30))
    start_epoch = 1

    if resume_path:
        checkpoint = active_checkpoint
        load_model_state_strict(model, checkpoint, str(resume_path))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_auc = float(checkpoint.get("best_auc", best_auc))
        best_accuracy = float(checkpoint.get("best_accuracy", best_accuracy))
        epochs_without_auc_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        print(f"Resumed from checkpoint: {resume_path} (epoch {checkpoint_epoch})")

        if start_epoch > total_epochs:
            print(
                f"Start epoch {start_epoch} is greater than total epochs {total_epochs}. "
                "Nothing to do."
            )
            return

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            threshold=threshold,
            label_smoothing=label_smoothing,
            fa_criterion=fa_criterion,
            lambda_fal=lambda_fal,
            aux_weight=float(config.get("aux_weight", 0.3)),
            max_steps=max_steps,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            threshold=threshold,
            max_steps=max_steps,
        )

        val_auc = float(val_metrics["auc"])
        val_accuracy = float(val_metrics["accuracy"])
        auc_improved = math.isfinite(val_auc) and val_auc > best_auc
        accuracy_improved = math.isfinite(val_accuracy) and val_accuracy > best_accuracy
        checkpoint_saved = False

        if accuracy_improved:
            best_accuracy = val_accuracy

        if auc_improved:
            best_auc = val_auc
            epochs_without_auc_improvement = 0
            save_checkpoint(
                save_path=str(checkpoint_path),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_auc=best_auc,
                best_accuracy=best_accuracy,
                scheduler=scheduler,
                config=config,
                epochs_without_improvement=epochs_without_auc_improvement,
            )
            checkpoint_saved = True
        else:
            epochs_without_auc_improvement += 1

        lrs_before = current_lrs(optimizer)
        scheduler_metric = val_auc if math.isfinite(val_auc) else -math.inf
        scheduler.step(scheduler_metric)
        lrs_after = current_lrs(optimizer)
        was_lr_reduced = lr_reduced(lrs_before, lrs_after)

        print(f"Train | {format_metrics(train_metrics)}")
        print(f"Cross | {format_metrics(val_metrics)}")
        print(f"Current LR: {', '.join(f'{lr:.8g}' for lr in lrs_after)}")
        print(
            f"Best validation AUC: {best_auc:.4f}"
            if math.isfinite(best_auc)
            else "Best validation AUC: n/a"
        )
        print(f"Epochs without AUC improvement: {epochs_without_auc_improvement}")
        print(f"Checkpoint saved: {'yes' if checkpoint_saved else 'no'}")
        print(f"LR reduced: {'yes' if was_lr_reduced else 'no'}")

        if epochs_without_auc_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epochs_without_auc_improvement} epochs "
                "without validation AUC improvement."
            )
            break

    print(f"\nTraining finished. Best checkpoint: {checkpoint_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "cuda")))
    save_dir = Path(config.get("save_dir", "checkpoints"))
    checkpoint_path = save_dir / str(config.get("checkpoint_name", "best_model.pth"))

    train_loader, val_loader = build_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Cross samples: {len(val_loader.dataset)}")

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
