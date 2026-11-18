import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets import DeepfakeFrameDataset, get_eval_transform, get_normalization_stats
from models import build_model
from train import count_model_parameters, load_config, resolve_device
from utils.checkpoint import load_checkpoint
from utils.inference import predict, predict_tta
from utils.metrics import aggregate_video_predictions, compute_binary_metrics, format_metrics

FFPP_MANIPULATIONS = ("deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter")


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive paper evaluation suite (Frame + Video + Robustness + Efficiency)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=str, default="paper_eval_results", help="Directory to save JSON/CSV summaries")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip perturbation robustness benchmark")
    parser.add_argument(
        "--extra-cross",
        nargs="*",
        default=[],
        help="Optional extra cross-dataset roots in name=path format (e.g. wilddeepfake=D:/data/wild/test)",
    )
    return parser.parse_args()


def build_model_from_config(config: Dict, device: torch.device) -> torch.nn.Module:
    return build_model(
        backbone=str(config.get("backbone", "favit_cnn")),
        pretrained=False,
        dropout=float(config.get("dropout", 0.4)),
        image_size=int(config.get("image_size", 224)),
        freq_in_channels=int(config.get("freq_in_channels", 3)),
        freq_dim=int(config.get("freq_dim", 128)),
        use_freq=bool(config.get("use_freq", True)),
        use_cnn_branch=bool(config.get("use_cnn_branch", True)),
        vit_backbone_name=str(config.get("vit_backbone_name", "vit_base_patch16_clip_224")),
        use_learnable_fft=bool(config.get("use_learnable_fft", True)),
        use_bidirectional_fusion=bool(config.get("use_bidirectional_fusion", True)),
        use_aux_head=bool(config.get("use_aux_head", True)),
        tune_vit_layernorm=bool(config.get("tune_vit_layernorm", True)),
    ).to(device)


def _jpeg_transform(image_size: int, quality: int, mean, std) -> A.Compose:
    import inspect

    sig = inspect.signature(A.ImageCompression)
    comp = (
        A.ImageCompression(quality_range=(quality, quality), p=1.0)
        if "quality_range" in sig.parameters
        else A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)
    )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            comp,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def _blur_transform(image_size: int, ksize: int, mean, std) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.GaussianBlur(blur_limit=(ksize, ksize), p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def _noise_transform(image_size: int, sigma: float, mean, std) -> A.Compose:
    import inspect

    sig = inspect.signature(A.GaussNoise)
    noise = (
        A.GaussNoise(std_range=(sigma, sigma), p=1.0)
        if "std_range" in sig.parameters
        else A.GaussNoise(var_limit=((sigma * 255.0) ** 2, (sigma * 255.0) ** 2), mean=0, p=1.0)
    )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            noise,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def _downsample_transform(image_size: int, scale: float, mean, std) -> A.Compose:
    low_res = max(16, int(image_size * scale))
    return A.Compose(
        [
            A.Resize(low_res, low_res, interpolation=cv2.INTER_AREA),
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: DeepfakeFrameDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    use_tta: bool,
    desc: str,
) -> Tuple[Dict[str, object], Dict[str, object], List[str], List[float], List[float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    run_predict = predict_tta if use_tta else predict
    image_paths, labels, probs = run_predict(model, loader, device, desc=desc)
    frame_metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    video_metrics, _, _, _ = aggregate_video_predictions(image_paths, labels, probs, threshold=threshold)
    return frame_metrics, video_metrics, image_paths, labels, probs


def compute_ffpp_per_manipulation(
    image_paths: List[str],
    labels: List[float],
    probs: List[float],
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Evaluate Original (real) vs each FF++ manipulation subset individually."""
    real_indices = [i for i, lbl in enumerate(labels) if int(round(lbl)) == 0]
    breakdown: Dict[str, Dict[str, object]] = {}
    for manip in FFPP_MANIPULATIONS:
        fake_indices = [
            i for i, (p, lbl) in enumerate(zip(image_paths, labels))
            if int(round(lbl)) == 1 and manip in p.lower()
        ]
        if not fake_indices or not real_indices:
            continue
        sel = real_indices + fake_indices
        sel_paths = [image_paths[i] for i in sel]
        sel_labels = [labels[i] for i in sel]
        sel_probs = [probs[i] for i in sel]
        frame_m = compute_binary_metrics(sel_labels, sel_probs, threshold=threshold)
        video_m, _, _, _ = aggregate_video_predictions(sel_paths, sel_labels, sel_probs, threshold=threshold)
        breakdown[manip] = {"frame": frame_m, "video": video_m}
    return breakdown


def profile_efficiency(model: torch.nn.Module, image_size: int, device: torch.device) -> Dict[str, object]:
    trainable, total = count_model_parameters(model)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    flops_g: Optional[float] = None
    try:
        from thop import profile as thop_profile  # type: ignore

        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        flops_g = float(macs) / 1e9
    except Exception:
        flops_g = None

    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        iters = 30
        for _ in range(iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    fps = iters / max(elapsed, 1e-6)
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": round(100.0 * trainable / max(total, 1), 2),
        "gflops": round(flops_g, 3) if flops_g is not None else None,
        "fps_bs1": round(fps, 2),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(config.get("threshold", 0.5))
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 4))
    mean, std = get_normalization_stats(
        backbone=str(config.get("backbone", "favit_cnn")),
        vit_backbone_name=str(config.get("vit_backbone_name", "vit_base_patch16_clip_224")),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config, device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    summary: Dict[str, object] = {
        "backbone": config.get("backbone"),
        "vit_backbone_name": config.get("vit_backbone_name"),
        "checkpoint": args.checkpoint,
        "efficiency": profile_efficiency(model, image_size, device),
        "datasets": {},
    }

    rows_for_csv: List[Dict[str, object]] = []

    def _record(name: str, frame_m: Dict[str, object], video_m: Dict[str, object]) -> None:
        print(f"[{name}] Frame | {format_metrics(frame_m)}")
        print(f"[{name}] Video | {format_metrics(video_m)} | videos: {video_m.get('num_videos')}")
        rows_for_csv.append(
            {
                "benchmark": name,
                "frame_auc": round(float(frame_m["auc"]) * 100, 2),
                "frame_acc": round(float(frame_m["accuracy"]) * 100, 2),
                "video_auc": round(float(video_m["auc"]) * 100, 2),
                "video_eer": round(float(video_m["eer"]) * 100, 2),
                "video_acc": round(float(video_m["accuracy"]) * 100, 2),
                "num_videos": video_m.get("num_videos"),
            }
        )

    # 1. In-domain FF++ test
    if config.get("data_root"):
        ffpp_test = DeepfakeFrameDataset(
            root_dir=config["data_root"],
            split=config.get("test_dir", "test"),
            dataset_type="ffpp",
            eval_transform=get_eval_transform(image_size, mean=mean, std=std),
            original_upsample_factor=0,
            mode="test",
        )
        f_m, v_m, paths, labels, probs = evaluate_dataset(
            model, ffpp_test, batch_size, num_workers, device, threshold, args.tta, "FF++ Test"
        )
        _record("ffpp_in_domain", f_m, v_m)
        per_manip = compute_ffpp_per_manipulation(paths, labels, probs, threshold)
        summary["datasets"]["ffpp_in_domain"] = {"frame": f_m, "video": v_m, "per_manipulation": per_manip}
        for manip, m_dict in per_manip.items():
            _record(f"ffpp_{manip}", m_dict["frame"], m_dict["video"])

    # 2. Cross-dataset (Celeb-DF-v2 + optional extra OOD sets)
    cross_targets: List[Tuple[str, str]] = []
    if config.get("cross_dataset_root"):
        cross_targets.append(("celeb_df_v2", str(config["cross_dataset_root"])))
    for item in args.extra_cross:
        if "=" not in item:
            raise ValueError(f"--extra-cross entries must be name=path, got: {item}")
        name, path = item.split("=", 1)
        cross_targets.append((name, path))

    for name, root in cross_targets:
        ds = DeepfakeFrameDataset(
            root_dir=root,
            split=None,
            dataset_type="cross",
            eval_transform=get_eval_transform(image_size, mean=mean, std=std),
            original_upsample_factor=0,
            mode="test",
        )
        f_m, v_m, _, _, _ = evaluate_dataset(
            model, ds, batch_size, num_workers, device, threshold, args.tta, f"Cross: {name}"
        )
        _record(f"cross_{name}", f_m, v_m)
        summary["datasets"][f"cross_{name}"] = {"frame": f_m, "video": v_m}

    # 3. Perturbation Robustness on FF++ (or Celeb-DF if FF++ absent)
    if not args.skip_robustness and (config.get("data_root") or config.get("cross_dataset_root")):
        robust_root = config["data_root"] if config.get("data_root") else config["cross_dataset_root"]
        robust_split = config.get("test_dir", "test") if config.get("data_root") else None
        robust_type = "ffpp" if config.get("data_root") else "cross"
        perturbations = [
            ("jpeg_q75", _jpeg_transform(image_size, 75, mean, std)),
            ("jpeg_q50", _jpeg_transform(image_size, 50, mean, std)),
            ("jpeg_q35", _jpeg_transform(image_size, 35, mean, std)),
            ("jpeg_q20", _jpeg_transform(image_size, 20, mean, std)),
            ("blur_k3", _blur_transform(image_size, 3, mean, std)),
            ("blur_k5", _blur_transform(image_size, 5, mean, std)),
            ("blur_k7", _blur_transform(image_size, 7, mean, std)),
            ("noise_0.01", _noise_transform(image_size, 0.01, mean, std)),
            ("noise_0.02", _noise_transform(image_size, 0.02, mean, std)),
            ("noise_0.04", _noise_transform(image_size, 0.04, mean, std)),
            ("resize_0.75", _downsample_transform(image_size, 0.75, mean, std)),
            ("resize_0.50", _downsample_transform(image_size, 0.50, mean, std)),
            ("resize_0.25", _downsample_transform(image_size, 0.25, mean, std)),
        ]
        robust_results: Dict[str, object] = {}
        for p_name, p_tf in perturbations:
            ds = DeepfakeFrameDataset(
                root_dir=robust_root,
                split=robust_split,
                dataset_type=robust_type,
                eval_transform=p_tf,
                original_upsample_factor=0,
                mode="test",
            )
            f_m, v_m, _, _, _ = evaluate_dataset(
                model, ds, batch_size, num_workers, device, threshold, False, f"Robust: {p_name}"
            )
            _record(f"robust_{p_name}", f_m, v_m)
            robust_results[p_name] = {"frame": f_m, "video": v_m}
        summary["robustness"] = robust_results

    json_path = output_dir / "eval_summary.json"
    csv_path = output_dir / "eval_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)
    print(f"\nSaved paper evaluation summary to: {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
