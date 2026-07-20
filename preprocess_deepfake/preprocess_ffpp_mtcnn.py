from utils import build_mtcnn_arg_parser, run_preprocess_ffpp_mtcnn


def main() -> None:
    parser = build_mtcnn_arg_parser("Preprocess FaceForensics++ videos with MTCNN face detection")
    args = parser.parse_args()

    run_preprocess_ffpp_mtcnn(
        input_root=args.input_root,
        output_root=args.output_root,
        img_size=args.img_size,
        seed=args.seed,
        train_frames_per_video=20,
        eval_frames_per_video=50,
    )


if __name__ == "__main__":
    main()
