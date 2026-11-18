import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from preprocess_ffpp_faceshifter_test_mtcnn import (
    FFPP_C23_FACES,
    SOURCE,
    load_video_names,
    process_video,
    resolve_source_dir,
)
from utils import MTCNNDetector, find_videos


FRAMES_PER_SPLIT = {"train": 20, "val": 50}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MTCNN face crops for FF++ c23 FaceShifter train and val videos."
    )
    parser.add_argument("--split", choices=FRAMES_PER_SPLIT, help="Process only this split")
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("F:/DeepFakedata/FaceForensics++_C23"),
        help="FF++ root containing FaceShifter",
    )
    parser.add_argument(
        "--reference_root", type=Path, default=FFPP_C23_FACES,
        help="Root containing <split>/fake/NeuralTextures video folders",
    )
    parser.add_argument(
        "--output_root", type=Path, default=FFPP_C23_FACES,
        help="Root for <split>/fake/FaceShifter/<video_id> crops",
    )
    return parser


def extract_split(
    split: str,
    names: List[str],
    videos_by_name: Dict[str, Path],
    output_dir: Path,
    detector: MTCNNDetector,
) -> None:
    selected = [name for name in names if name in videos_by_name]
    missing = [name for name in names if name not in videos_by_name]
    if missing:
        preview = ", ".join(missing[:20])
        print(f"[WARN] {split}: Missing videos: {preview}{' ...' if len(missing) > 20 else ''}")

    totals = {"saved": 0, "existing": 0, "no_face": 0}
    for name in tqdm(selected, desc=f"{SOURCE} extract {split}", unit="video"):
        counts = process_video(
            video_path=videos_by_name[name],
            output_dir=output_dir / name,
            num_frames=FRAMES_PER_SPLIT[split],
            detector=detector,
            img_size=224,
        )
        for key in totals:
            totals[key] += counts[key]

    print(f"\n========== FaceShifter {split} summary ==========")
    print(f"Videos processed: {len(selected)} (missing: {len(missing)})")
    print(f"Images saved: {totals['saved']}, already existed: {totals['existing']}")
    print(f"Skipped frames without detected face/readable crop: {totals['no_face']}")
    print(f"Output: {output_dir}")


def main() -> None:
    args = build_parser().parse_args()
    splits = (args.split,) if args.split else FRAMES_PER_SPLIT
    names_by_split = {
        split: load_video_names(None, args.reference_root / split / "fake" / "NeuralTextures")
        for split in splits
    }
    source_dir = resolve_source_dir(args.input_root)
    videos_by_name = {path.stem: path for path in find_videos(source_dir)}
    detector = MTCNNDetector()
    for split, names in names_by_split.items():
        extract_split(split, names, videos_by_name, args.output_root / split / "fake" / SOURCE, detector)


if __name__ == "__main__":
    main()
