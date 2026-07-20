from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    save_path: str,
    epoch: int,
    model,
    optimizer,
    best_auc: float,
    config: Dict[str, Any],
    best_accuracy: Optional[float] = None,
    scheduler=None,
    epochs_without_improvement: int = 0,
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auc": best_auc,
        "config": config,
        "epochs_without_improvement": int(epochs_without_improvement),
    }
    if best_accuracy is not None:
        checkpoint["best_accuracy"] = best_accuracy
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_path: str, device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False)
