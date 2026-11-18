"""Preflight checks for the Windows experiment launcher; does not start training."""

import argparse
import importlib
import sys
from pathlib import Path

import yaml


TRAINING_DIR = Path(__file__).resolve().parent
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures")


def config_path(config, key):
    if not config.get(key):
        raise ValueError(f"Missing config key: {key}")
    path = Path(config[key]).expanduser()
    return path if path.is_absolute() else TRAINING_DIR / path


def check_data_layout(config):
    """Check the supplied video-folder layout and return video-directory counts."""
    root = config_path(config, "data_root")
    folders = []
    for key, default in (("train_dir", "train"), ("val_dir", "val"), ("test_dir", "test")):
        split = root / str(config.get(key, default))
        folders.append(split / "real")
        folders.extend(split / "fake" / method for method in FFPP_METHODS)
    cross = config_path(config, "cross_dataset_root")
    folders.extend((cross / "0", cross / "1"))
    missing = [str(folder) for folder in folders if not folder.is_dir()]
    if missing:
        raise ValueError("Missing data directories:\n  " + "\n  ".join(missing))
    counts = [(folder, sum(child.is_dir() for child in folder.iterdir())) for folder in folders]
    empty = [str(folder) for folder, count in counts if count == 0]
    if empty:
        raise ValueError("No video folders in:\n  " + "\n  ".join(empty))
    return counts


def check_runtime(config, check_cuda=True):
    # Import entrypoints too: syntax checks alone cannot detect missing packages.
    for module in (
        "torchvision", "timm", "albumentations", "cv2", "sklearn", "pandas",
        "numpy", "tqdm", "matplotlib.pyplot", "train", "eval_suite", "visualize_paper",
    ):
        importlib.import_module(module)
    torch = importlib.import_module("torch")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and check_cuda:
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch installation. "
                "Use setup_and_run.ps1 -Reinstall to repair the isolated environment."
            )
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GiB")
        amp_dtype = str(config.get("amp_dtype", "bfloat16")).lower()
        if config.get("use_amp", True) and amp_dtype not in {"fp16", "float16"}:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("The configured GPU does not support bfloat16. Set use_amp: false in your config.")
        # Exercise a CUDA kernel to catch incompatible wheels/architectures early.
        sample = torch.ones((16, 16), device=device)
        if (sample @ sample).sum().item() != 4096.0:
            raise ValueError("CUDA matrix multiplication check failed.")
    elif not check_cuda:
        print("CUDA execution check skipped (scenario listing only).")
    else:
        print(f"Configured device: {device}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-data-check", action="store_true")
    parser.add_argument("--skip-cuda-check", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    for key in ("image_size", "batch_size", "grad_accum_steps"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive.")
    if str(config.get("selection_metric", "video_auc")).strip().lower() not in {"auc", "video_auc"}:
        raise ValueError("selection_metric must be 'auc' or 'video_auc'.")
    if not args.skip_data_check:
        for folder, count in check_data_layout(config):
            print(f"Data: {folder} | video folders: {count}")
    check_runtime(config, check_cuda=not args.skip_cuda_check)
    batch = int(config.get("batch_size", 16))
    accumulation = int(config.get("grad_accum_steps", 1))
    print(f"Batch size: {batch} | accumulation: {accumulation} | effective batch: {batch * accumulation}")
    print(f"Checkpoint metric: val_{config.get('selection_metric', 'video_auc')}")
    print("Preflight checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print(f"SETUP CHECK FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
