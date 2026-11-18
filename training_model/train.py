import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from datasets import (
    DeepfakeFrameDataset,
    get_eval_transform,
    get_normalization_stats,
    get_train_transform,
)
from models import (
    FALoss,
    SpectralConsistencyLoss,
    build_model,
    perturb_batch_for_debiasing,
)
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import aggregate_video_predictions, compute_binary_metrics, format_metrics
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a configurable backbone for deepfake detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config_device: str) -> torch.device:
    if config_device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(config_device)


def current_lrs(optimizer) -> list:
    return [group["lr"] for group in optimizer.param_groups]


def lr_reduced(before: list, after: list) -> bool:
    return any(new_lr < old_lr for old_lr, new_lr in zip(before, after))


def _count_labels(dataset) -> Tuple[int, int]:
    """Return (num_real, num_fake) by inspecting .samples on leaf datasets."""
    if isinstance(dataset, ConcatDataset):
        total_real, total_fake = 0, 0
        for sub in dataset.datasets:
            r, f = _count_labels(sub)
            total_real += r
            total_fake += f
        return total_real, total_fake
    if hasattr(dataset, "samples"):
        labels = [s[1] for s in dataset.samples]
        return sum(1 for l in labels if l == 0.0), sum(1 for l in labels if l == 1.0)
    return 0, 0


def _pos_weight(dataset, device: torch.device) -> Optional[torch.Tensor]:
    num_real, num_fake = _count_labels(dataset)
    if num_fake == 0:
        return None
    pw = num_real / num_fake
    print(f"Class distribution — real: {num_real}, fake: {num_fake}, pos_weight: {pw:.4f}")
    return torch.tensor([pw], device=device)


def resolve_pos_weight(config: Dict, dataset, device: torch.device) -> Optional[torch.Tensor]:
    """Return a manual pos_weight, an automatically calculated value, or None."""
    if not config.get("use_pos_weight", False):
        return None

    manual_pos_weight = config.get("pos_weight")
    if manual_pos_weight is None:
        return _pos_weight(dataset, device)

    pw = float(manual_pos_weight)
    if pw <= 0:
        raise ValueError("pos_weight must be greater than 0")
    print(f"Using manual pos_weight: {pw:.4f}")
    return torch.tensor([pw], device=device)


def unpack_model_output(output):
    if isinstance(output, tuple):
        if len(output) == 2:
            logits, features = output
            return logits, features, None
        if len(output) == 3:
            return output
        raise ValueError(f"Unsupported model output tuple length: {len(output)}")
    return output, None, None


def get_classifier_weight(model) -> Optional[torch.Tensor]:
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        return None
    if hasattr(classifier, "weight"):
        return classifier.weight
    if isinstance(classifier, nn.Sequential):
        for module in reversed(classifier):
            if isinstance(module, nn.Linear):
                return module.weight
    return None


def count_model_parameters(model) -> Tuple[int, int]:
    """Return the trainable and total parameter counts for a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def load_model_state_strict(
    model,
    checkpoint: Dict[str, Any],
    checkpoint_path: str,
) -> None:
    """Load model weights strictly and add checkpoint/config context to failures."""
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict': {checkpoint_path}")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Strict model-state loading failed for checkpoint "
            f"'{checkpoint_path}'. Check that the backbone and all model-related "
            "configuration values match the run that created the checkpoint. "
            f"Original error: {error}"
        ) from error


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    mean, std = get_normalization_stats(
        backbone=str(config.get("backbone", "favit_cnn")),
        vit_backbone_name=str(config.get("vit_backbone_name", "vit_base_patch16_clip_224")),
    )
    train_transform = get_train_transform(image_size, mean=mean, std=std)
    eval_transform = get_eval_transform(image_size, mean=mean, std=std)

    exclude_manipulations = config.get("exclude_manipulations")
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
        exclude_classes=exclude_manipulations,
    )
    val_dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["val_dir"],
        dataset_type="ffpp",
        train_transform=None,
        eval_transform=eval_transform,
        original_upsample_factor=0,
        mode="val",
        exclude_classes=exclude_manipulations,
    )

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


def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    threshold: float = 0.5,
    label_smoothing: float = 0.0,
    fa_criterion=None,
    lambda_fal: float = 0.0,
    aux_weight: float = 0.0,
    spectral_criterion=None,
    lambda_freq_cons: float = 0.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    grad_accum_steps: int = 1,
    max_steps: Optional[int] = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    processed_samples = 0
    labels_all = []
    probs_all = []
    paths_all = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    grad_accum_steps = max(1, int(grad_accum_steps))
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for step, (images, labels, paths) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, features, aux_logits = unpack_model_output(model(images))
                logits_for_loss = logits.view_as(labels)
                smooth_labels = (
                    labels * (1.0 - label_smoothing) + label_smoothing * 0.5
                    if is_train and label_smoothing > 0.0
                    else labels
                )
                loss = criterion(logits_for_loss, smooth_labels)

                if is_train and aux_logits is not None and aux_weight > 0.0:
                    loss = loss + aux_weight * criterion(aux_logits.view_as(labels), smooth_labels)

                if (
                    is_train
                    and fa_criterion is not None
                    and lambda_fal > 0.0
                    and features is not None
                ):
                    classifier_weight = get_classifier_weight(model)
                    if classifier_weight is not None:
                        loss = loss + lambda_fal * fa_criterion(features, labels, classifier_weight)

                if (
                    is_train
                    and spectral_criterion is not None
                    and lambda_freq_cons > 0.0
                    and features is not None
                ):
                    perturbed_images = perturb_batch_for_debiasing(images)
                    logits_pert, feat_pert, _ = unpack_model_output(model(perturbed_images))
                    if feat_pert is not None:
                        loss = loss + lambda_freq_cons * spectral_criterion(
                            features, feat_pert, logits_for_loss, logits_pert.view_as(labels)
                        )

            if is_train:
                window_start = ((step - 1) // grad_accum_steps) * grad_accum_steps
                window_end = min(window_start + grad_accum_steps, total_steps)
                window_size = max(1, window_end - window_start)
                scaled_loss = loss / window_size
                scaled_loss.backward()
                if step % grad_accum_steps == 0 or step == total_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        processed_samples += batch_size
        probs = torch.sigmoid(logits.detach().float().view(-1))

        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
        if not is_train:
            paths_all.extend(paths)
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if max_steps is not None and step >= max_steps:
            break

    if processed_samples == 0:
        raise RuntimeError("The data loader did not yield any samples.")
    avg_loss = total_loss / processed_samples
    metrics = compute_binary_metrics(labels_all, probs_all, threshold=threshold, loss=avg_loss)
    if not is_train and paths_all:
        video_metrics, _, _, _ = aggregate_video_predictions(paths_all, labels_all, probs_all, threshold=threshold)
        metrics["video_auc"] = video_metrics["auc"]
        metrics["video_eer"] = video_metrics["eer"]
    return metrics


def create_fa_criterion(model, config: Dict, device: torch.device, lambda_fal: float):
    """Build the optional forgery-aware loss for classifier features."""
    if lambda_fal <= 0.0:
        return None
    return FALoss(
        margin=float(config.get("fal_margin", 0.25)),
        scale=float(config.get("fal_scale", 32.0)),
    )


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
    selection_metric = str(config.get("selection_metric", "video_auc")).strip().lower()
    if selection_metric not in {"auc", "video_auc"}:
        raise ValueError("selection_metric must be 'auc' or 'video_auc'")
    print(f"Checkpoint selection metric: val_{selection_metric}")
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
        use_cnn_branch=bool(config.get("use_cnn_branch", True)),
        vit_backbone_name=str(config.get("vit_backbone_name", "vit_base_patch16_clip_224")),
        use_learnable_fft=bool(config.get("use_learnable_fft", True)),
        use_bidirectional_fusion=bool(config.get("use_bidirectional_fusion", True)),
        use_aux_head=bool(config.get("use_aux_head", True)),
        tune_vit_layernorm=bool(config.get("tune_vit_layernorm", True)),
        num_unfrozen_blocks=int(config.get("num_unfrozen_blocks", 2)),
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
    aux_weight = float(config.get("aux_weight", 0.3))
    lambda_freq_cons = float(config.get("lambda_freq_cons", 0.0))
    spectral_criterion = SpectralConsistencyLoss() if lambda_freq_cons > 0.0 else None
    use_amp = bool(config.get("use_amp", True))
    amp_dtype_name = str(config.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name in {"fp16", "float16"} else torch.bfloat16
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
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
        # PyTorch reduces after bad_epochs > patience. This makes the config value
        # mean "reduce after this many consecutive non-improving epochs."
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
            aux_weight=aux_weight,
            spectral_criterion=spectral_criterion,
            lambda_freq_cons=lambda_freq_cons,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            max_steps=max_steps,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            threshold=threshold,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_steps=max_steps,
        )

        val_auc = float(val_metrics[selection_metric])
        val_accuracy = float(val_metrics["accuracy"])
        auc_improved = math.isfinite(val_auc) and val_auc > best_auc
        accuracy_improved = math.isfinite(val_accuracy) and val_accuracy > best_accuracy
        checkpoint_saved = False

        if accuracy_improved:
            best_accuracy = val_accuracy

        if auc_improved:
            best_auc = val_auc
            epochs_without_auc_improvement = 0
        else:
            epochs_without_auc_improvement += 1

        lrs_before = current_lrs(optimizer)
        scheduler_metric = val_auc if math.isfinite(val_auc) else -math.inf
        scheduler.step(scheduler_metric)
        lrs_after = current_lrs(optimizer)
        was_lr_reduced = lr_reduced(lrs_before, lrs_after)

        if auc_improved:
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

        last_checkpoint_path = checkpoint_path.with_name(f"last_{checkpoint_path.name}")
        save_checkpoint(
            save_path=str(last_checkpoint_path),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_auc=best_auc,
            best_accuracy=best_accuracy,
            scheduler=scheduler,
            config=config,
            epochs_without_improvement=epochs_without_auc_improvement,
        )

        print(f"Train | {format_metrics(train_metrics)}")
        print(f"Val   | {format_metrics(val_metrics)}")
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
