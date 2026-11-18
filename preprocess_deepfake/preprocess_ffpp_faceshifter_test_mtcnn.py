import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from utils import (
    MTCNNDetector,
    crop_with_margin,
    evenly_spaced_frame_indices,
    find_videos,
    write_face,
)


SOURCE = "FaceShifter"
FFPP_C23_FACES = Path("E:/Deepfake_Data_Chien/ffpp_celebdf_data/processed/faces/ffpp/c23")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MTCNN face crops for FaceForensics++ FaceShifter test videos. "
            "Output layout: <output_dir>/<video_name>/000.jpg ... 049.jpg"
        )
    )
    parser.add_argument(
        "--input_root",
        default=Path("F:/DeepFakedata/FaceForensics++_C23"),
        type=Path,
        help="FF++ root folder that contains the FaceShifter folder",
    )
    parser.add_argument(
        "--output_dir",
        default=FFPP_C23_FACES / "test" / "fake" / SOURCE,
        type=Path,
        help="Folder to write FaceShifter face crops",
    )
    parser.add_argument(
        "--split_json",
        default=None,
        type=Path,
        help="Official FF++ split file (e.g. splits/test.json). If set, overrides --reference_dir",
    )
    parser.add_argument(
        "--reference_dir",
        default=FFPP_C23_FACES / "test" / "fake" / "NeuralTextures",
        type=Path,
        help="Already-processed fake method folder; its video subfolder names define the test list",
    )
    parser.add_argument("--img_size", default=224, type=int, help="Output face crop size in pixels (existing FF++ c23 data is 224)")
    parser.add_argument("--frames_per_video", default=50, type=int, help="Evenly spaced frames per video")
    return parser


def load_video_names(split_json: Path, reference_dir: Path) -> List[str]:
    if split_json is not None:
        # Official FF++ split: list of [a, b] pairs -> fake videos a_b and b_a.
        pairs = json.loads(split_json.read_text(encoding="utf-8"))
        names = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
        print(f"[INFO] Loaded {len(names)} video names from {split_json}")
    else:
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference folder not found: {reference_dir}")
        names = {path.name for path in reference_dir.iterdir() if path.is_dir()}
        print(f"[INFO] Loaded {len(names)} video names from {reference_dir}")

    if not names:
        raise RuntimeError("Empty video list")
    return sorted(names)


def resolve_source_dir(input_root: Path) -> Path:
    candidates = (
        input_root / SOURCE,
        input_root / "manipulated_sequences" / SOURCE,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"FaceShifter folder not found. Tried: {', '.join(str(c) for c in candidates)}")


def process_video(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
    detector: MTCNNDetector,
    img_size: int,
) -> Dict[str, int]:
    counts = {"saved": 0, "existing": 0, "no_face": 0}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return counts

    frame_indices = evenly_spaced_frame_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_frames)
    if not frame_indices:
        print(f"[WARN] Empty or unreadable video: {video_path}")
        cap.release()
        return counts

    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_index, frame_index in enumerate(frame_indices):
        output_path = output_dir / f"{sample_index:03d}.jpg"
        if output_path.exists():
            counts["existing"] += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            counts["no_face"] += 1
            continue

        bbox = detector.largest_face(frame)
        face = crop_with_margin(frame, bbox, margin=0.2) if bbox is not None else None
        if face is None:
            counts["no_face"] += 1
            continue

        if write_face(face, output_path, img_size):
            counts["saved"] += 1

    cap.release()
    return counts


def main() -> None:
    args = build_parser().parse_args()
    if args.img_size <= 0:
        raise ValueError("--img_size must be a positive integer")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    names = load_video_names(args.split_json, args.reference_dir)
    source_dir = resolve_source_dir(args.input_root)
    videos_by_name = {path.stem: path for path in find_videos(source_dir)}

    missing = [name for name in names if name not in videos_by_name]
    selected = [name for name in names if name in videos_by_name]
    print(f"[INFO] {SOURCE}: {len(videos_by_name)} videos found, {len(selected)} selected, {len(missing)} missing")
    if missing:
        print(f"[WARN] Missing videos: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")

    detector = MTCNNDetector()
    totals = {"saved": 0, "existing": 0, "no_face": 0}
    for name in tqdm(selected, desc=f"{SOURCE} extract test", unit="video"):
        counts = process_video(
            video_path=videos_by_name[name],
            output_dir=args.output_dir / name,
            num_frames=args.frames_per_video,
            detector=detector,
            img_size=args.img_size,
        )
        for key in totals:
            totals[key] += counts[key]

    print("\n========== FaceShifter test summary ==========")
    print(f"Videos processed: {len(selected)} (missing: {len(missing)})")
    print(f"Images saved: {totals['saved']}, already existed: {totals['existing']}")
    print(f"Skipped frames without detected face/readable crop: {totals['no_face']}")
    print(f"Output: {args.output_dir}")
    print("==============================================")


if __name__ == "__main__":
    main()
