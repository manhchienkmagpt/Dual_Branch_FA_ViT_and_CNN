# Deepfake Preprocessing with SCRFD or MTCNN

This project extracts evenly distributed video frames, detects the largest face in each frame with SCRFD or MTCNN, crops the face with a 20% margin, and saves flat `.jpg` face crops for Deepfake Detection training.

## Install

```bash
pip install -r requirements.txt
```

On Kaggle, choose a GPU runtime before installing. On Windows, install CUDA/cuDNN versions compatible with your `onnxruntime-gpu` and PyTorch packages.

MTCNN uses `facenet-pytorch` and PyTorch. If PyTorch is not installed automatically in your environment, install the PyTorch build that matches your CUDA/CPU setup before running the MTCNN scripts.

## Download SCRFD ONNX

Only SCRFD scripts need an ONNX model. Download an SCRFD ONNX model such as `scrfd_10g_bnkps.onnx` from the InsightFace model zoo, then pass its local path with `--scrfd_model`.

Example path:

```text
models/scrfd_10g_bnkps.onnx
```

MTCNN scripts do not use `--scrfd_model`.

## FaceForensics++ Preprocess

Expected input:

```text
root_folder/
|__ original/
|__ Deepfakes/
|__ Face2Face/
|__ FaceSwap/
|__ NeuralTextures/
|__ FaceShifter/
```

Run with SCRFD:

```bash
python preprocess_ffpp_scrfd.py ^
  --input_root "D:/datasets/FF++" ^
  --output_root "D:/datasets/ffpp_faces_scrfd" ^
  --scrfd_model "models/scrfd_10g_bnkps.onnx" ^
  --img_size 224 ^
  --seed 42
```

Run with MTCNN:

```bash
python preprocess_ffpp_mtcnn.py ^
  --input_root "D:/datasets/FF++" ^
  --output_root "D:/datasets/ffpp_faces_mtcnn" ^
  --img_size 380 ^
  --seed 42
```

Linux/Kaggle:

```bash
python preprocess_ffpp_scrfd.py \
  --input_root "/kaggle/input/ffpp" \
  --output_root "/kaggle/working/ffpp_faces_scrfd" \
  --scrfd_model "models/scrfd_10g_bnkps.onnx" \
  --img_size 224 \
  --seed 42
```

```bash
python preprocess_ffpp_mtcnn.py \
  --input_root "/kaggle/input/ffpp" \
  --output_root "/kaggle/working/ffpp_faces_mtcnn" \
  --img_size 380 \
  --seed 42
```

FF++ settings:

- SCRFD script: `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, and `NeuralTextures`, 32 sampled frames per video.
- MTCNN script: `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, and `FaceShifter`; 20 sampled frames per train video and 50 sampled frames per val/test video.
- Videos are sorted inside each source folder and split before frame extraction: the first 720 videos go to train, the next 140 go to val, and the remaining videos go to test. This keeps matching FF++ videos aligned across source folders.

## CelebDF-v2 Preprocess

Expected input:

```text
celeb_root_folder/
|__ Celeb-real/
|__ Celeb-synthesis/
|__ YouTube-real/
|__ List_of_testing_videos.txt
```

Run test with SCRFD:

```bash
python preprocess_celebdf_test_scrfd.py ^
  --input_root "D:/datasets/CelebDF-v2" ^
  --output_root "D:/datasets/celebdf_faces_scrfd" ^
  --scrfd_model "models/scrfd_10g_bnkps.onnx" ^
  --img_size 224 ^
  --test_list "D:/datasets/CelebDF-v2/List_of_testing_videos.txt"
```

Run test with MTCNN:

```bash
python preprocess_celebdf_test_mtcnn.py ^
  --input_root "D:/datasets/CelebDF-v2" ^
  --output_root "D:/datasets/celebdf_faces_mtcnn" ^
  --img_size 380 ^
  --test_list "D:/datasets/CelebDF-v2/List_of_testing_videos.txt"
```

Linux/Kaggle:

```bash
python preprocess_celebdf_test_scrfd.py \
  --input_root "/kaggle/input/celebdf-v2" \
  --output_root "/kaggle/working/celebdf_faces_scrfd" \
  --scrfd_model "models/scrfd_10g_bnkps.onnx" \
  --img_size 224 \
  --test_list "/kaggle/input/celebdf-v2/List_of_testing_videos.txt"
```

```bash
python preprocess_celebdf_test_mtcnn.py \
  --input_root "/kaggle/input/celebdf-v2" \
  --output_root "/kaggle/working/celebdf_faces_mtcnn" \
  --img_size 380 \
  --test_list "/kaggle/input/celebdf-v2/List_of_testing_videos.txt"
```

CelebDF-v2 settings:

- SCRFD test script: test videos are read from the txt file, 32 sampled frames per video.
- MTCNN test script: test videos are read from the txt file, 50 sampled frames per video.
- For CelebDF test list labels, label `1` is saved under `test/real`; label `0` is saved under `test/fake`.

## Output Structure

For FaceForensics++, output is grouped by split first and then by original source folder:

```text
output_root/
|__ train/
|   |__ original/
|   |__ Deepfakes/
|   |__ Face2Face/
|   |__ FaceSwap/
|   |__ NeuralTextures/
|   |__ FaceShifter/
|__ val/
|   |__ original/
|   |__ Deepfakes/
|   |__ Face2Face/
|   |__ FaceSwap/
|   |__ NeuralTextures/
|   |__ FaceShifter/
|__ test/
    |__ original/
    |__ Deepfakes/
    |__ Face2Face/
    |__ FaceSwap/
    |__ NeuralTextures/
    |__ FaceShifter/
```

For CelebDF-v2 test preprocessing, output is grouped by split and label:

```text
output_root/
|__ test/
    |__ real/
    |__ fake/
```

The output folders contain only face crop images, without per-video subfolders. File names include dataset name, split, source or label, source video id, and frame index.

Example:

```text
ffpp_train_Deepfakes_Deepfakes_001_002_sample000005_frame000128.jpg
```

At the end, each script prints:

- Number of real/fake videos.
- Number of saved real/fake face images in each frame-level split.
- Number of skipped frames where no face/readable crop was detected.
- Number of failed videos.
