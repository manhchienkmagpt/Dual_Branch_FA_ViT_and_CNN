import argparse
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

SPLITS = ("train", "val", "test")
LABELS = ("real", "fake")


@dataclass(frozen=True)
class VideoItem:
    path: Path
    label: str
    source: str


@dataclass(frozen=True)
class ExtractedFrame:
    temp_path: Path
    label: str
    source: str
    video_id: str
    sample_index: int
    frame_index: int


@dataclass
class Stats:
    video_counts: Dict[str, int] = field(default_factory=lambda: {"real": 0, "fake": 0})
    image_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            split: {"real": 0, "fake": 0}
            for split in SPLITS
        }
    )
    source_image_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            split: {}
            for split in SPLITS
        }
    )
    skipped_no_face: int = 0
    failed_videos: int = 0


class SCRFDDetector:
    """Small wrapper around InsightFace SCRFD ONNX inference."""

    def __init__(
        self,
        model_path: str,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        prefer_gpu: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.det_size = det_size
        self.det_thresh = det_thresh

        try:
            import onnxruntime as ort
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise RuntimeError(
                "Missing SCRFD dependencies. Install with: "
                "pip install insightface onnxruntime-gpu opencv-python tqdm numpy"
            ) from exc

        available = set(ort.get_available_providers())
        providers: List[str] = []
        if prefer_gpu and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        if providers[0] != "CUDAExecutionProvider":
            print("[WARN] CUDAExecutionProvider is unavailable. SCRFD will run on CPU.")

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"SCRFD model not found: {model_file}")

        self.model = get_model(str(model_file), providers=providers)
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        self.model.prepare(ctx_id=ctx_id, input_size=self.det_size, det_thresh=self.det_thresh)
        print(f"[INFO] Loaded SCRFD model with provider: {providers[0]}")

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return face boxes as Nx5 arrays: x1, y1, x2, y2, score."""
        bboxes, _ = self.model.detect(image_bgr, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(bboxes, dtype=np.float32)

    def largest_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        bboxes = self.detect(image_bgr)
        if len(bboxes) == 0:
            return None

        widths = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0])
        heights = np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
        areas = widths * heights
        return bboxes[int(np.argmax(areas))]


class MTCNNDetector:
    """Face detector wrapper with the same largest_face API as SCRFDDetector."""

    def __init__(
        self,
        min_face_size: int = 20,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        prefer_gpu: bool = True,
    ) -> None:
        try:
            import torch
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise RuntimeError(
                "Missing MTCNN dependencies. Install with: "
                "pip install facenet-pytorch torch opencv-python tqdm numpy"
            ) from exc

        use_cuda = prefer_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if prefer_gpu and not use_cuda:
            print("[WARN] CUDA is unavailable. MTCNN will run on CPU.")

        self.model = MTCNN(
            keep_all=True,
            min_face_size=min_face_size,
            thresholds=thresholds,
            device=self.device,
        )
        print(f"[INFO] Loaded MTCNN detector on device: {self.device}")

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return face boxes as Nx5 arrays: x1, y1, x2, y2, score."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.model.detect(image_rgb)
        if boxes is None or probs is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        probs = np.asarray(probs, dtype=np.float32).reshape(-1, 1)
        boxes = np.asarray(boxes, dtype=np.float32)
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(probs[:, 0])
        if not np.any(valid):
            return np.empty((0, 5), dtype=np.float32)
        return np.concatenate([boxes[valid], probs[valid]], axis=1)

    def largest_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        bboxes = self.detect(image_bgr)
        if len(bboxes) == 0:
            return None

        widths = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0])
        heights = np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
        areas = widths * heights
        return bboxes[int(np.argmax(areas))]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_root", required=True, type=Path, help="Root folder of the source videos")
    parser.add_argument("--output_root", required=True, type=Path, help="Folder to write cropped face images")
    parser.add_argument("--scrfd_model", required=True, type=Path, help="Path to SCRFD ONNX model")
    parser.add_argument("--img_size", default=380, type=int, help="Output face crop size in pixels")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for dataset split")
    return parser


def build_mtcnn_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_root", required=True, type=Path, help="Root folder of the source videos")
    parser.add_argument("--output_root", required=True, type=Path, help="Folder to write cropped face images")
    parser.add_argument("--img_size", default=380, type=int, help="Output face crop size in pixels")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for dataset split")
    return parser


def find_videos(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"[WARN] Missing folder, skipped: {folder}")
        return []

    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=natural_path_key,
    )


def natural_path_key(path: Path) -> Tuple[object, ...]:
    parts: List[object] = []
    for part in path.parts:
        for token in re.split(r"(\d+)", part.lower()):
            if token.isdigit():
                parts.append(int(token))
            elif token:
                parts.append(token)
    return tuple(parts)


def collect_videos(input_root: Path, real_dirs: Sequence[str], fake_dirs: Sequence[str]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for dirname in real_dirs:
        for path in find_videos(input_root / dirname):
            items.append(VideoItem(path=path, label="real", source=dirname))

    for dirname in fake_dirs:
        for path in find_videos(input_root / dirname):
            items.append(VideoItem(path=path, label="fake", source=dirname))

    return items


def ensure_output_dirs(output_root: Path) -> None:
    for split in SPLITS:
        for label in LABELS:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)


def ensure_label_output_dirs(output_root: Path, splits: Sequence[str]) -> None:
    for split in splits:
        for label in LABELS:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)


def ensure_source_output_dirs(output_root: Path, sources: Sequence[str]) -> None:
    for split in SPLITS:
        for source in sources:
            (output_root / split / source).mkdir(parents=True, exist_ok=True)


def evenly_spaced_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0 or num_frames <= 0:
        return []

    # Endpoint sampling spreads frames over the full video instead of taking a leading burst.
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)
    return [int(idx) for idx in indices]


def crop_with_margin(image_bgr: np.ndarray, bbox: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(float)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 1 or box_h <= 1:
        return None

    pad_x = box_w * margin
    pad_y = box_h * margin
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]


def safe_video_id(video_path: Path, input_root: Path) -> str:
    try:
        rel = video_path.relative_to(input_root)
    except ValueError:
        rel = video_path.name

    rel_no_suffix = str(Path(rel).with_suffix(""))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", rel_no_suffix).strip("_")


def write_face(
    face_bgr: np.ndarray,
    output_path: Path,
    img_size: int,
) -> bool:
    resized = cv2.resize(face_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(output_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def final_frame_path(
    output_root: Path,
    dataset_name: str,
    split: str,
    label: str,
    video_id: str,
    sample_index: int,
    frame_index: int,
) -> Path:
    filename = (
        f"{dataset_name}_{split}_{label}_{video_id}_"
        f"sample{sample_index:06d}_frame{frame_index:06d}.jpg"
    )
    return output_root / split / label / filename


def process_video(
    item: VideoItem,
    input_root: Path,
    temp_dir: Path,
    num_frames: int,
    detector: SCRFDDetector,
    img_size: int,
    stats: Stats,
) -> List[ExtractedFrame]:
    extracted: List[ExtractedFrame] = []
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {item.path}")
        stats.failed_videos += 1
        return extracted

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = evenly_spaced_frame_indices(total_frames, num_frames)
    if not frame_indices:
        print(f"[WARN] Empty or unreadable video: {item.path}")
        stats.failed_videos += 1
        cap.release()
        return extracted

    video_id = safe_video_id(item.path, input_root)
    for sample_index, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            stats.skipped_no_face += 1
            continue

        bbox = detector.largest_face(frame)
        if bbox is None:
            stats.skipped_no_face += 1
            continue

        face = crop_with_margin(frame, bbox, margin=0.2)
        if face is None:
            stats.skipped_no_face += 1
            continue

        temp_filename = (
            f"{item.label}_{video_id}_sample{sample_index:06d}_"
            f"frame{frame_index:06d}.jpg"
        )
        temp_path = temp_dir / temp_filename
        if write_face(face, temp_path, img_size):
            extracted.append(
                ExtractedFrame(
                    temp_path=temp_path,
                    label=item.label,
                    source=item.source,
                    video_id=video_id,
                    sample_index=sample_index,
                    frame_index=frame_index,
                )
            )

    cap.release()
    return extracted


def split_extracted_frames(frames: Sequence[ExtractedFrame], seed: int) -> Dict[str, List[ExtractedFrame]]:
    rng = random.Random(seed)
    shuffled = sorted(frames, key=lambda frame: str(frame.temp_path))
    rng.shuffle(shuffled)

    n_frames = len(shuffled)
    n_train = int(n_frames * 0.8)
    n_val = int(n_frames * 0.1)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def split_videos_by_source(
    items: Sequence[VideoItem],
    counts: Tuple[int, int, int] = (720, 140, 140),
) -> Dict[str, List[VideoItem]]:
    by_source: Dict[str, List[VideoItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)

    if any(count < 0 for count in counts):
        raise ValueError("Split counts must be non-negative")

    splits: Dict[str, List[VideoItem]] = {split: [] for split in SPLITS}
    for source, source_items in sorted(by_source.items()):
        ordered = sorted(source_items, key=lambda item: natural_path_key(item.path))
        n_train = min(counts[0], len(ordered))
        n_val = min(counts[1], max(0, len(ordered) - n_train))

        splits["train"].extend(ordered[:n_train])
        splits["val"].extend(ordered[n_train:n_train + n_val])
        splits["test"].extend(ordered[n_train + n_val:])

    for split in SPLITS:
        splits[split] = sorted(splits[split], key=lambda item: (item.source, natural_path_key(item.path)))

    return splits


def parse_celebdf_test_list(test_list_path: Path, input_root: Path) -> List[VideoItem]:
    if not test_list_path.is_file():
        raise FileNotFoundError(f"CelebDF test list not found: {test_list_path}")

    items: List[VideoItem] = []
    with test_list_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid CelebDF test list line {line_number}: expected '<label> <relative_video_path>'"
                )

            raw_label, relative_video = parts
            relative_path = Path(relative_video.replace("\\", "/"))
            if raw_label == "1":
                label = "real"
            elif raw_label == "0":
                label = "fake"
            else:
                raise ValueError(
                    f"Invalid CelebDF test label on line {line_number}: {raw_label!r}. "
                    "Use 1 for real and 0 for fake."
                )

            video_path = input_root / relative_path
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"CelebDF test video from line {line_number} not found: {video_path}"
                )
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"CelebDF test path is not a supported video on line {line_number}: {video_path}")

            source = relative_path.parts[0] if relative_path.parts else video_path.parent.name
            items.append(VideoItem(path=video_path, label=label, source=source))

    if not items:
        raise RuntimeError(f"CelebDF test list is empty: {test_list_path}")

    return items


def split_train_val_videos(
    items: Sequence[VideoItem],
    seed: int,
    train_ratio: float = 0.8,
) -> Dict[str, List[VideoItem]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = random.Random(seed)
    splits: Dict[str, List[VideoItem]] = {"train": [], "val": [], "test": []}
    for label in LABELS:
        label_items = sorted((item for item in items if item.label == label), key=lambda item: natural_path_key(item.path))
        rng.shuffle(label_items)
        n_train = int(len(label_items) * train_ratio)
        splits["train"].extend(label_items[:n_train])
        splits["val"].extend(label_items[n_train:])

    for split in ("train", "val"):
        splits[split] = sorted(splits[split], key=lambda item: (item.label, natural_path_key(item.path)))

    return splits


def move_frames_to_splits(
    frames_by_split: Dict[str, List[ExtractedFrame]],
    output_root: Path,
    dataset_name: str,
    stats: Stats,
) -> None:
    for split in SPLITS:
        for frame in tqdm(frames_by_split[split], desc=f"{dataset_name} move {split}", unit="frame"):
            output_path = final_frame_path(
                output_root=output_root,
                dataset_name=dataset_name,
                split=split,
                label=frame.label,
                video_id=frame.video_id,
                sample_index=frame.sample_index,
                frame_index=frame.frame_index,
            )
            shutil.move(str(frame.temp_path), str(output_path))
            stats.image_counts[split][frame.label] += 1


def final_source_frame_path(
    output_root: Path,
    dataset_name: str,
    split: str,
    source: str,
    video_id: str,
    sample_index: int,
    frame_index: int,
) -> Path:
    filename = (
        f"{dataset_name}_{split}_{source}_{video_id}_"
        f"sample{sample_index:06d}_frame{frame_index:06d}.jpg"
    )
    return output_root / split / source / filename


def process_video_to_source_split(
    item: VideoItem,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    split: str,
    num_frames: int,
    detector: SCRFDDetector,
    img_size: int,
    stats: Stats,
) -> None:
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {item.path}")
        stats.failed_videos += 1
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = evenly_spaced_frame_indices(total_frames, num_frames)
    if not frame_indices:
        print(f"[WARN] Empty or unreadable video: {item.path}")
        stats.failed_videos += 1
        cap.release()
        return

    video_id = safe_video_id(item.path, input_root)
    for sample_index, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            stats.skipped_no_face += 1
            continue

        bbox = detector.largest_face(frame)
        if bbox is None:
            stats.skipped_no_face += 1
            continue

        face = crop_with_margin(frame, bbox, margin=0.2)
        if face is None:
            stats.skipped_no_face += 1
            continue

        output_path = final_source_frame_path(
            output_root=output_root,
            dataset_name=dataset_name,
            split=split,
            source=item.source,
            video_id=video_id,
            sample_index=sample_index,
            frame_index=frame_index,
        )
        if write_face(face, output_path, img_size):
            stats.image_counts[split][item.label] += 1
            stats.source_image_counts[split][item.source] = (
                stats.source_image_counts[split].get(item.source, 0) + 1
            )

    cap.release()


def process_video_to_label_split(
    item: VideoItem,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    split: str,
    num_frames: int,
    detector: SCRFDDetector,
    img_size: int,
    stats: Stats,
) -> None:
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {item.path}")
        stats.failed_videos += 1
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = evenly_spaced_frame_indices(total_frames, num_frames)
    if not frame_indices:
        print(f"[WARN] Empty or unreadable video: {item.path}")
        stats.failed_videos += 1
        cap.release()
        return

    video_id = safe_video_id(item.path, input_root)
    for sample_index, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            stats.skipped_no_face += 1
            continue

        bbox = detector.largest_face(frame)
        if bbox is None:
            stats.skipped_no_face += 1
            continue

        face = crop_with_margin(frame, bbox, margin=0.2)
        if face is None:
            stats.skipped_no_face += 1
            continue

        output_path = final_frame_path(
            output_root=output_root,
            dataset_name=dataset_name,
            split=split,
            label=item.label,
            video_id=video_id,
            sample_index=sample_index,
            frame_index=frame_index,
        )
        if write_face(face, output_path, img_size):
            stats.image_counts[split][item.label] += 1

    cap.release()


def resolve_celebdf_test_list(input_root: Path, test_list: Optional[str]) -> Path:
    test_list_path = Path(test_list) if test_list else input_root / "List_of_testing_videos.txt"
    if not test_list_path.is_absolute():
        test_list_path = input_root / test_list_path
    return test_list_path


def run_preprocess_celebdf_train_val(
    input_root: Path,
    output_root: Path,
    scrfd_model: Path,
    img_size: int,
    seed: int,
    test_list: Optional[str] = None,
) -> None:
    input_root = input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if img_size <= 0:
        raise ValueError("--img_size must be a positive integer")

    real_dirs = ("Celeb-real", "YouTube-real")
    fake_dirs = ("Celeb-synthesis",)
    frames_per_video = 32

    all_items = collect_videos(input_root, real_dirs, fake_dirs)
    if not all_items:
        raise RuntimeError(f"No videos found under: {input_root}")

    test_list_path = resolve_celebdf_test_list(input_root, test_list)
    test_items = parse_celebdf_test_list(test_list_path, input_root)
    test_paths = {str(item.path.resolve()).lower() for item in test_items}
    train_val_items = [
        item
        for item in all_items
        if str(item.path.resolve()).lower() not in test_paths
    ]
    if not train_val_items:
        raise RuntimeError("No CelebDF videos remain for train/val after excluding the test list")

    video_splits = split_train_val_videos(train_val_items, seed=seed, train_ratio=0.8)

    stats = Stats()
    stats.video_counts["real"] = sum(1 for item in train_val_items if item.label == "real")
    stats.video_counts["fake"] = sum(1 for item in train_val_items if item.label == "fake")

    ensure_label_output_dirs(output_root, ("train", "val"))
    detector = SCRFDDetector(str(scrfd_model))

    for split in ("train", "val"):
        desc = f"celebdf extract {split}"
        for item in tqdm(video_splits[split], desc=desc, unit="video"):
            process_video_to_label_split(
                item=item,
                input_root=input_root,
                output_root=output_root,
                dataset_name="celebdf",
                split=split,
                num_frames=frames_per_video,
                detector=detector,
                img_size=img_size,
                stats=stats,
            )

    print_summary(stats)


def run_preprocess_celebdf_test(
    input_root: Path,
    output_root: Path,
    scrfd_model: Path,
    img_size: int,
    test_list: Optional[str] = None,
) -> None:
    input_root = input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if img_size <= 0:
        raise ValueError("--img_size must be a positive integer")

    frames_per_video = 32
    test_list_path = resolve_celebdf_test_list(input_root, test_list)
    test_items = parse_celebdf_test_list(test_list_path, input_root)
    test_items = sorted(test_items, key=lambda item: (item.label, natural_path_key(item.path)))

    stats = Stats()
    stats.video_counts["real"] = sum(1 for item in test_items if item.label == "real")
    stats.video_counts["fake"] = sum(1 for item in test_items if item.label == "fake")

    ensure_label_output_dirs(output_root, ("test",))
    detector = SCRFDDetector(str(scrfd_model))

    for item in tqdm(test_items, desc="celebdf extract test", unit="video"):
        process_video_to_label_split(
            item=item,
            input_root=input_root,
            output_root=output_root,
            dataset_name="celebdf",
            split="test",
            num_frames=frames_per_video,
            detector=detector,
            img_size=img_size,
            stats=stats,
        )

    print_summary(stats)


def run_preprocess_celebdf_test_mtcnn(
    input_root: Path,
    output_root: Path,
    img_size: int,
    test_list: Optional[str] = None,
    frames_per_video: int = 50,
) -> None:
    input_root = input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if img_size <= 0:
        raise ValueError("--img_size must be a positive integer")

    test_list_path = resolve_celebdf_test_list(input_root, test_list)
    test_items = parse_celebdf_test_list(test_list_path, input_root)
    test_items = sorted(test_items, key=lambda item: (item.label, natural_path_key(item.path)))

    stats = Stats()
    stats.video_counts["real"] = sum(1 for item in test_items if item.label == "real")
    stats.video_counts["fake"] = sum(1 for item in test_items if item.label == "fake")

    ensure_label_output_dirs(output_root, ("test",))
    detector = MTCNNDetector()

    for item in tqdm(test_items, desc="celebdf extract test", unit="video"):
        process_video_to_label_split(
            item=item,
            input_root=input_root,
            output_root=output_root,
            dataset_name="celebdf",
            split="test",
            num_frames=frames_per_video,
            detector=detector,
            img_size=img_size,
            stats=stats,
        )

    print_summary(stats)


def run_preprocess_celebdf(
    input_root: Path,
    output_root: Path,
    scrfd_model: Path,
    img_size: int,
    seed: int,
    test_list: Optional[str] = None,
) -> None:
    run_preprocess_celebdf_train_val(
        input_root=input_root,
        output_root=output_root,
        scrfd_model=scrfd_model,
        img_size=img_size,
        seed=seed,
        test_list=test_list,
    )
    run_preprocess_celebdf_test(
        input_root=input_root,
        output_root=output_root,
        scrfd_model=scrfd_model,
        img_size=img_size,
        test_list=test_list,
    )


def run_preprocess(
    dataset_name: str,
    input_root: Path,
    output_root: Path,
    scrfd_model: Path,
    img_size: int,
    seed: int,
    real_dirs: Sequence[str],
    fake_dirs: Sequence[str],
    frames_per_real_video: int,
    frames_per_fake_video: int,
    split_by_video_source: bool = False,
    video_split_counts: Tuple[int, int, int] = (720, 140, 140),
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if img_size <= 0:
        raise ValueError("--img_size must be a positive integer")

    items = collect_videos(input_root, real_dirs, fake_dirs)
    if not items:
        raise RuntimeError(f"No videos found under: {input_root}")

    stats = Stats()
    stats.video_counts["real"] = sum(1 for item in items if item.label == "real")
    stats.video_counts["fake"] = sum(1 for item in items if item.label == "fake")

    detector = SCRFDDetector(str(scrfd_model))

    if split_by_video_source:
        ensure_source_output_dirs(output_root, (*real_dirs, *fake_dirs))
        video_splits = split_videos_by_source(items, counts=video_split_counts)
        for split in SPLITS:
            desc = f"{dataset_name} extract {split}"
            for item in tqdm(video_splits[split], desc=desc, unit="video"):
                frames_per_video = frames_per_real_video if item.label == "real" else frames_per_fake_video
                process_video_to_source_split(
                    item=item,
                    input_root=input_root,
                    output_root=output_root,
                    dataset_name=dataset_name,
                    split=split,
                    num_frames=frames_per_video,
                    detector=detector,
                    img_size=img_size,
                    stats=stats,
                )

        print_summary(stats)
        return

    ensure_output_dirs(output_root)
    temp_dir = Path(tempfile.mkdtemp(prefix=".frame_split_", dir=str(output_root)))
    extracted_frames: List[ExtractedFrame] = []

    try:
        for label in LABELS:
            label_items = sorted((item for item in items if item.label == label), key=lambda item: str(item.path))
            frames_per_video = frames_per_real_video if label == "real" else frames_per_fake_video
            desc = f"{dataset_name} extract {label}"
            for item in tqdm(label_items, desc=desc, unit="video"):
                extracted_frames.extend(
                    process_video(
                        item=item,
                        input_root=input_root,
                        temp_dir=temp_dir,
                        num_frames=frames_per_video,
                        detector=detector,
                        img_size=img_size,
                        stats=stats,
                    )
                )

        frame_splits = split_extracted_frames(extracted_frames, seed=seed)
        move_frames_to_splits(frame_splits, output_root, dataset_name, stats)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print_summary(stats)


def run_preprocess_ffpp_mtcnn(
    input_root: Path,
    output_root: Path,
    img_size: int,
    seed: int,
    real_dirs: Sequence[str] = ("original",),
    fake_dirs: Sequence[str] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"),
    train_frames_per_video: int = 20,
    eval_frames_per_video: int = 50,
    video_split_counts: Tuple[int, int, int] = (720, 140, 140),
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if img_size <= 0:
        raise ValueError("--img_size must be a positive integer")

    items = collect_videos(input_root, real_dirs, fake_dirs)
    if not items:
        raise RuntimeError(f"No videos found under: {input_root}")

    stats = Stats()
    stats.video_counts["real"] = sum(1 for item in items if item.label == "real")
    stats.video_counts["fake"] = sum(1 for item in items if item.label == "fake")

    ensure_source_output_dirs(output_root, (*real_dirs, *fake_dirs))
    detector = MTCNNDetector()
    video_splits = split_videos_by_source(items, counts=video_split_counts)

    for split in SPLITS:
        desc = f"ffpp extract {split}"
        frames_per_video = train_frames_per_video if split == "train" else eval_frames_per_video
        for item in tqdm(video_splits[split], desc=desc, unit="video"):
            process_video_to_source_split(
                item=item,
                input_root=input_root,
                output_root=output_root,
                dataset_name="ffpp",
                split=split,
                num_frames=frames_per_video,
                detector=detector,
                img_size=img_size,
                stats=stats,
            )

    print_summary(stats)


def print_summary(stats: Stats) -> None:
    print("\n========== Preprocess summary ==========")
    print(f"Videos real: {stats.video_counts['real']}")
    print(f"Videos fake: {stats.video_counts['fake']}")

    print("\nSaved face images:")
    for split in SPLITS:
        print(
            f"  {split}: real={stats.image_counts[split]['real']}, "
            f"fake={stats.image_counts[split]['fake']}"
        )
        if stats.source_image_counts[split]:
            source_counts = ", ".join(
                f"{source}={count}"
                for source, count in sorted(stats.source_image_counts[split].items())
            )
            print(f"    by source: {source_counts}")

    print(f"\nSkipped frames without detected face/readable crop: {stats.skipped_no_face}")
    print(f"Failed videos: {stats.failed_videos}")
    print("========================================")
