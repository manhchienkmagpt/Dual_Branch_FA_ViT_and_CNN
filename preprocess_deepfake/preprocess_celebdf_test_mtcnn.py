from utils import build_mtcnn_arg_parser, run_preprocess_celebdf_test_mtcnn


def main() -> None:
    parser = build_mtcnn_arg_parser("Preprocess CelebDF-v2 test videos with MTCNN face detection")
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="CelebDF test list txt. Defaults to input_root/List_of_testing_videos.txt",
    )
    args = parser.parse_args()

    run_preprocess_celebdf_test_mtcnn(
        input_root=args.input_root,
        output_root=args.output_root,
        img_size=args.img_size,
        test_list=args.test_list,
        frames_per_video=50,
    )


if __name__ == "__main__":
    main()
