from utils import build_arg_parser, run_preprocess


def main() -> None:
    parser = build_arg_parser("Preprocess FaceForensics++ videos with SCRFD face detection")
    args = parser.parse_args()

    run_preprocess(
        dataset_name="ffpp",
        input_root=args.input_root,
        output_root=args.output_root,
        scrfd_model=args.scrfd_model,
        img_size=args.img_size,
        seed=args.seed,
        real_dirs=("original",),
        fake_dirs=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"),
        frames_per_real_video=32,
        frames_per_fake_video=32,
        split_by_video_source=True,
        video_split_counts=(720, 140, 140),
    )


if __name__ == "__main__":
    main()
