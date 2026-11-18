import argparse
import json
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from datasets import DeepfakeFrameDataset, get_eval_transform, get_normalization_stats
from eval_suite import build_model_from_config
from train import load_config, resolve_device, unpack_model_output
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication figures (Anomaly maps, t-SNE, Robustness curves)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-json", type=str, default=None, help="Optional eval_summary.json for robustness curves")
    parser.add_argument("--output-dir", type=str, default="paper_figures")
    parser.add_argument("--max-samples", type=int, default=400, help="Max samples for t-SNE embedding plot")
    parser.add_argument("--num-heatmaps", type=int, default=6, help="Number of sample heatmaps to render")
    return parser.parse_args()


def _denormalize(tensor_img: torch.Tensor, mean, std) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    img = tensor_img.permute(1, 2, 0).cpu().numpy() * std + mean
    return np.clip(img, 0.0, 1.0)


def plot_patch_heatmaps(model, dataset, device, output_dir: Path, num_heatmaps: int = 6, *, mean, std) -> None:
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(num_heatmaps, len(dataset)), dtype=int)
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        img_tensor, label, _ = dataset[int(idx)]
        batch = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, _ = unpack_model_output(model(batch))
            prob = float(torch.sigmoid(logits.view(-1)).item())
            amap = getattr(model, "last_patch_anomaly_map", None)

        rgb = _denormalize(img_tensor, mean, std)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"GT: {'Fake' if label >= 0.5 else 'Real'} | p={prob:.2f}")
        axes[row, 0].axis("off")

        if amap is not None:
            heat = amap[0].cpu().numpy()
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            heat_resized = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            axes[row, 1].imshow(heat_resized, cmap="jet")
            axes[row, 1].set_title("Patch Anomaly Map")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(rgb)
            axes[row, 2].imshow(heat_resized, cmap="jet", alpha=0.45)
            axes[row, 2].set_title("Overlay")
            axes[row, 2].axis("off")
        else:
            axes[row, 1].axis("off")
            axes[row, 2].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "patch_anomaly_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(model, dataset, device, output_dir: Path, max_samples: int = 400) -> None:
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    feats: List[np.ndarray] = []
    labels: List[int] = []

    with torch.no_grad():
        for images, lbls, _ in loader:
            images = images.to(device)
            _, feat, _ = unpack_model_output(model(images))
            if feat is None:
                return
            feats.append(feat.float().cpu().numpy())
            labels.extend([int(round(x)) for x in lbls.numpy().tolist()])
            if len(labels) >= max_samples:
                break

    feat_mat = np.concatenate(feats, axis=0)[:max_samples]
    lbl_arr = np.asarray(labels[:max_samples])
    if len(feat_mat) < 10:
        return

    perplexity = min(30, max(5, len(feat_mat) // 4))
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(feat_mat)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[lbl_arr == 0, 0], emb[lbl_arr == 0, 1], c="#1f77b4", label="Real", alpha=0.65, s=18)
    ax.scatter(emb[lbl_arr == 1, 0], emb[lbl_arr == 1, 1], c="#d62728", label="Fake", alpha=0.65, s=18)
    ax.legend()
    ax.set_title("Feature Manifold (t-SNE)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(output_dir / "feature_tsne.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_curves(eval_json_path: Path, output_dir: Path) -> None:
    with eval_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    robust = data.get("robustness", {})
    if not robust:
        return

    groups = {
        "JPEG Quality": ["jpeg_q75", "jpeg_q50", "jpeg_q35", "jpeg_q20"],
        "Gaussian Blur": ["blur_k3", "blur_k5", "blur_k7"],
        "Gaussian Noise": ["noise_0.01", "noise_0.02", "noise_0.04"],
        "Downsample": ["resize_0.75", "resize_0.50", "resize_0.25"],
    }
    fig, axes = plt.subplots(1, len(groups), figsize=(16, 3.5))
    for ax, (title, keys) in zip(axes, groups.items()):
        xs = [k.split("_")[-1] for k in keys if k in robust]
        ys = [robust[k]["video"]["auc"] * 100.0 for k in keys if k in robust]
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2)
            ax.set_ylim(50, 101)
            ax.set_title(title)
            ax.set_ylabel("Video AUC (%)")
            ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "robustness_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "cuda")))
    image_size = int(config.get("image_size", 224))
    mean, std = get_normalization_stats(
        backbone=str(config.get("backbone", "favit_cnn")),
        vit_backbone_name=str(config.get("vit_backbone_name", "vit_base_patch16_clip_224")),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config, device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    root = config.get("cross_dataset_root") or config.get("data_root")
    split = None if config.get("cross_dataset_root") else config.get("test_dir", "test")
    dtype = "cross" if config.get("cross_dataset_root") else "ffpp"
    if root:
        dataset = DeepfakeFrameDataset(
            root_dir=root,
            split=split,
            dataset_type=dtype,
            eval_transform=get_eval_transform(image_size, mean=mean, std=std),
            original_upsample_factor=0,
            mode="test",
        )
        plot_patch_heatmaps(
            model, dataset, device, output_dir,
            num_heatmaps=args.num_heatmaps, mean=mean, std=std,
        )
        plot_tsne(model, dataset, device, output_dir, max_samples=args.max_samples)

    if args.eval_json:
        plot_robustness_curves(Path(args.eval_json), output_dir)
    print(f"Saved paper figures to: {output_dir}")


if __name__ == "__main__":
    main()
