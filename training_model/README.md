# Deepfake Detection Backbones

PyTorch project for binary deepfake frame detection with configurable ImageNet backbones. EfficientNet-B4, ResNet-50, Swin-Tiny, and Swin-Small use the backbone feature vector followed by a dropout classifier. The model outputs one raw logit. Training uses `BCEWithLogitsLoss`; inference converts logits with `torch.sigmoid`.

Labels:

- `real`: `0`
- `fake`: `1`

## Project Structure

```text
deepfake_efficientnetb4/
|-- configs/config.yaml
|-- datasets/deepfake_dataset.py
|-- models/backbones.py
|-- utils/
|-- train.py
|-- train_cross.py
|-- train_stage_2.py
|-- train_cross_stage_2.py
|-- train_with_gan.py
|-- test_origin_dataset.py
|-- test_cross_dataset.py
|-- requirements.txt
`-- README.md
```

## Install

From the repository root on Windows, double-click `setup_and_run.bat` or run:

```powershell
.\setup_and_run.bat
```

The launcher requires 64-bit Python 3.10-3.14, an NVIDIA driver, and internet for
initial package/pretrained-weight downloads. It creates `.venv-training`, installs
PyTorch 2.11.0 + torchvision 0.26.0 CUDA 12.8 wheels and this project's requirements,
checks imports/CUDA/data directories, then runs all experiment scenarios. The wheel
pair follows the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2110).
It reuses installed dependencies and experiment progress on subsequent launches.
Installed package versions are recorded in `.venv-training/installed-packages.txt`.

The default config is `configs/unified_favit_rtx5060ti.yaml`; edit its data paths
and training settings for your machine. The launcher keeps these settings and
checks video-folder counts; it does not audit split manifests or source identities.

```powershell
# Show the setup plan without installing packages or starting training:
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -Preview
# Install/check the environment without requiring data or starting training:
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -SetupOnly
# After setup, list all scenario states without checking data or running CUDA kernels:
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -DryRun
# Use another Python executable/config:
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -PythonExe C:\Python312\python.exe -ConfigPath training_model\configs\unified_favit_rtx5060ti.yaml
```

Run the same launcher again to resume. Use `-Reinstall` if the isolated environment
needs repair. Normal runs check all FF++ train/val/test real and fake method folders
and Celeb-DF `test/0`, `test/1` before launching training.

After each epoch, the console prints train/validation frame loss, accuracy,
balanced accuracy, F1, precision, recall, ROC-AUC, EER, and AP. It also prints the
selected validation AUC (`val_video_auc` by default), best AUC, current LR,
epochs without improvement, whether the best checkpoint was saved, and whether
LR was reduced. `last_best_model.pth` is saved after every completed epoch.
Training loss includes enabled auxiliary losses; validation loss is BCE only.
FF++ test and Celeb-DF test are evaluated after training each scenario completes.

```bash
cd deepfake_efficientnetb4
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your machine if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Data Layout

FF++ train, validation, and test data are expected under `data_root`:

```text
root/
|-- train/
|   |-- original/
|   |-- Deepfakes/
|   |-- Face2Face/
|   |-- FaceShifter/
|   |-- FaceSwap/
|   `-- NeuralTextures/
|-- val/
|   |-- original/
|   |-- Deepfakes/
|   |-- Face2Face/
|   |-- FaceShifter/
|   |-- FaceSwap/
|   `-- NeuralTextures/
`-- test/
    |-- original/
    |-- Deepfakes/
    |-- Face2Face/
    |-- FaceShifter/
    |-- FaceSwap/
    `-- NeuralTextures/
```

`original` is label `0`. All manipulation folders are label `1`.

The cross-dataset test root is configured with `cross_dataset_root` and should contain:

```text
cross_dataset_root/
|-- real/
`-- fake/
```

[WildDeepfake](https://www.kaggle.com/datasets/maysuni/wild-deepfake)
stage-two training reuses `data_root`, `train_dir`, `val_dir`, and `test_dir`
for this normalized layout:

```text
data_root/
|-- train/
|   |-- real/
|   `-- fake/
|-- val/
|   |-- real/
|   `-- fake/
`-- test/
    |-- real/
    `-- fake/
```

Image discovery below every class folder is recursive, so nested video or
sequence directories are included. WildDeepfake keeps the same binary labels:
`real = 0` and `fake = 1`.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

GAN training data can be configured with `gan_fake_dir` and `gan_real_dir` in `configs/config.yaml`.
Images under `gan_fake_dir` use label `1`; images under `gan_real_dir` use label `0`.

```yaml
gan_fake_dir: "path/to/gan_fake"
gan_real_dir: "path/to/gan_real"
```

## Train

Edit `configs/config.yaml` if needed, then run:

```bash
python train.py --config configs/config.yaml
```

Choose a backbone in `configs/config.yaml`:

```yaml
backbone: "efficientnetb4"  # options: efficientnetb4, resnet50, swin_tiny, swin_small, redesigned_favit, favit_cnn
```

Available values:

- `efficientnetb4`: EfficientNet-B4
- `resnet50`: ResNet-50
- `swin_tiny`: Swin Transformer Tiny
- `swin_small`: Swin Transformer Small
- `redesigned_favit`: frozen ViT-B/16 backbone with trainable GAM, LAM, CNN local branch, and classifier
- `favit_cnn`: `redesigned_favit` style RGB branch plus a lightweight CNN feature extractor branch

`redesigned_favit` and `favit_cnn` return raw logits plus features internally.
`favit_cnn` returns classifier features after LayerNorm and before dropout so
FALoss and the final Linear classifier operate in the same feature space.
Use `lambda_fal` to enable the optional forgery-aware loss:

```yaml
backbone: "favit_cnn"
freq_in_channels: 3
freq_dim: 128
use_freq: true
lambda_fal: 0.1
fal_margin: 0.25
fal_scale: 32
```

`favit_cnn` accepts an optional auxiliary tensor in its forward pass. If `freq_x` is omitted, it uses the RGB input as the CNN branch input.

Resume from a checkpoint:

```bash
python train.py --config configs/config.yaml --resume checkpoints/best_model.pth
```

Or set `resume_from` in `configs/config.yaml`.

The best checkpoint is saved when validation ROC-AUC strictly improves, to:

```text
save_dir/checkpoint_name
```

Configure the normal training checkpoint path with:

```yaml
save_dir: "checkpoints_efficientnetb4"
checkpoint_name: "best_model.pth"
```

Training uses:

- `BCEWithLogitsLoss`
- `AdamW`
- `ReduceLROnPlateau(mode="max")` tracking the selected validation AUC
- Early stopping tracking the same validation AUC
- Strong augmentation for all train samples, including real, fake, and upsampled real samples

For `train.py` and its GAN/stage-two wrappers, choose the validation metric with:

```yaml
selection_metric: "video_auc"  # default; use "auc" for frame-level val_auc
```

Each video's frames must be directly inside its own video folder. Grouping uses
the full parent directory path, including dataset, split, class, and method;
frame filenames and numeric folder names such as `0` and `1` do not affect grouping.
Video scores are mean frame probabilities within each folder. Accuracy remains a
reported metric. Best and last checkpoints store the scheduler state after its
epoch update so resuming keeps the same AUC schedule.

Resuming with a different `selection_metric` raises an error. Keep the original
metric or start a new run. Legacy checkpoints without metric metadata restore
model/optimizer state but reset best AUC, scheduler history, and the early-stopping
counter. Paper visualization uses the same normalization and matching RGB
denormalization as evaluation: CLIP statistics for CLIP and mean/std `0.5` for
the `redesigned_favit` backbone.

The `ablation_1_vit_gam` scenario is CLIP ViT with trainable LayerNorm and GAM,
without the CNN/frequency branch. It is not a plain ViT baseline.

Set `original_upsample_factor: null` to disable upsampling. Set it to `N` to keep all real samples and add `N` extra augmented copies for each real training image.

Set `train_real_percent` to control how many real images from `train_dir/original` are used during training:

```yaml
train_real_percent: 50  # use 50% of train/original
```

This only affects the origin training split. Validation, test, and GAN real data are unchanged.

## Train Checkpoint Cross

Use `train_cross.py` when you want the same training behavior as `train.py`, but with validation metrics computed on the cross dataset instead of the FF++ validation split.

First configure the origin training data and cross-dataset root in `configs/config.yaml`:

```yaml
data_root: "path/to/ffpp_faces"
train_dir: "train"
cross_dataset_root: "path/to/cross_dataset/test"
save_dir: "checkpoints_cross"
checkpoint_name: "best_cross.pth"
```

The cross dataset must use this layout:

```text
cross_dataset_root/
|-- real/
`-- fake/
```

Then run:

```bash
python train_cross.py --config configs/config.yaml
```

Resume from a checkpoint:

```bash
python train_cross.py --config configs/config.yaml --resume checkpoints_cross/best_cross.pth
```

This script:

- trains on `data_root/train_dir`
- evaluates on `cross_dataset_root` after every epoch
- saves a checkpoint when cross-dataset AUC improves
- uses `ReduceLROnPlateau(mode="max")` and early stopping based on cross-dataset AUC

## WildDeepfake Stage-Two Training

Stage-two training supports only `redesigned_favit` and `favit_cnn`.
Point the existing dataset keys at the normalized WildDeepfake splits, then
configure the separate stage-two output paths:

```yaml
data_root: "path/to/wilddeepfake_root"
train_dir: "train"
val_dir: "val"
test_dir: "test"

stage_2_save_dir: "checkpoints_stage_2"
stage_2_checkpoint_name: "best_stage_2.pth"
cross_stage_2_save_dir: "checkpoints_cross_stage_2"
cross_stage_2_checkpoint_name: "best_cross_stage_2.pth"
```

`train_stage_2.py` starts from a stage-one checkpoint produced by `train.py`.
It trains on `data_root/train_dir` and validates on `data_root/val_dir`.

```bash
python train_stage_2.py --config configs/config.yaml \
  --resume checkpoints/best_model.pth
```

`train_cross_stage_2.py` starts from a stage-one checkpoint produced by
`train_cross.py`. It trains on the WildDeepfake training split but continues to
validate on `cross_dataset_root`; it does not use the WildDeepfake validation
split.

```bash
python train_cross_stage_2.py --config configs/config.yaml \
  --resume checkpoints_cross/best_cross.pth
```

The two scripts use the same resume behavior as `train.py` and
`train_cross.py`: `--resume` overrides `resume_from` in the configuration and
restores the model, optimizer, scheduler, saved epoch, best metrics, and
early-stopping counter. Stage-two training requires one of those resume paths.
To continue a later stage-two run, pass its matching stage-two checkpoint:

```bash
python train_stage_2.py --config configs/config.yaml \
  --resume path/to/stage_2_checkpoint.pth

python train_cross_stage_2.py --config configs/config.yaml \
  --resume path/to/cross_stage_2_checkpoint.pth
```

Both scripts verify checkpoint `config.backbone` metadata before constructing
the model, then load model weights strictly. Keep `backbone`, `dropout`,
`image_size`, `lambda_fal`, `fal_margin`, and `fal_scale` consistent with the
stage-one run. For `favit_cnn`, also keep `freq_in_channels`, `freq_dim`,
and `use_freq` consistent.

## Train With GAN Data

Edit `gan_fake_dir` and `gan_real_dir` in `configs/config.yaml`, then run:

```bash
python train_with_gan.py --config configs/config.yaml
```

Resume from a checkpoint:

```bash
python train_with_gan.py --config configs/config.yaml --resume checkpoints/best_model_with_gan.pth
```

This script trains on origin train data plus GAN train data and keeps origin validation unchanged. The best checkpoint defaults to:

```text
checkpoints/best_model_with_gan.pth
```

## Test Origin Dataset

```bash
python test_origin_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional CSV path:

```bash
python test_origin_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output-csv outputs/origin_predictions.csv
```

## Test Cross Dataset

```bash
python test_cross_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional CSV path:

```bash
python test_cross_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output-csv outputs/cross_predictions.csv
```

Both test scripts print Accuracy, F1, Precision, Recall, AUC, and the confusion matrix, and save per-image predictions with:

- `image_path`
- `label`
- `probability`
- `prediction`

## Best model arcitect

### FA_VIT
![Model Architecture](assets\FA_VIT.png)

### FA_VIT_Plus
![Model Architecture](assets\FA_VIT_plus.png)
