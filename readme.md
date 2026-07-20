# FA-ViT-CNN cho phát hiện Deepfake

Project xây dựng pipeline phát hiện deepfake ở mức **ảnh khuôn mặt (frame-level)**. Video đầu vào được lấy mẫu theo thời gian, phát hiện và cắt khuôn mặt; các ảnh sau tiền xử lý được dùng để huấn luyện mô hình nhị phân **FA-ViT-CNN** nhằm dự đoán `real` hoặc `fake`.

## Tổng quan pipeline

```text
Video FF++ / Celeb-DF-v2
        |
        v
Lấy mẫu frame cách đều
        |
        v
MTCNN phát hiện khuôn mặt lớn nhất
        |
        v
Cắt khuôn mặt + margin 20% + resize
        |
        v
Ảnh JPG theo split/lớp
        |
        v
FA-ViT-CNN -> logit -> sigmoid -> real/fake
```

Mã nguồn được chia thành hai phần:

- `preprocess_deepfake/`: tiền xử lý FaceForensics++ và Celeb-DF-v2.
- `training_model/`: dataset loader, mô hình, huấn luyện và đánh giá.

## 1. Tiền xử lý dữ liệu

### Quy trình chung

Các script hiện có sử dụng **MTCNN** (`facenet-pytorch`) và thực hiện:

1. Tìm video đệ quy trong thư mục dữ liệu.
2. Chọn các frame cách đều trên toàn bộ video bằng `numpy.linspace`.
3. Phát hiện tất cả khuôn mặt và giữ khuôn mặt có bounding box lớn nhất.
4. Mở rộng bounding box thêm 20% theo mỗi chiều.
5. Resize ảnh về kích thước vuông `img_size`.
6. Lưu dưới dạng JPEG, chất lượng 95; frame không đọc được hoặc không tìm thấy mặt sẽ bị bỏ qua.

MTCNN tự dùng CUDA khi khả dụng, nếu không sẽ chạy trên CPU. Thiết lập detector mặc định là `min_face_size=20` và thresholds `(0.6, 0.7, 0.7)`.

### FaceForensics++ (FF++)

Cấu trúc video đầu vào:

```text
FF++/
|-- original/          # real
|-- Deepfakes/         # fake
|-- Face2Face/         # fake
|-- FaceSwap/          # fake
|-- NeuralTextures/    # fake
`-- FaceShifter/       # fake
```

Video trong từng nguồn được sắp xếp và chia trước khi trích frame:

- 720 video đầu: `train`, lấy 20 frame/video.
- 140 video tiếp theo: `val`, lấy 50 frame/video.
- Các video còn lại: `test`, lấy 50 frame/video.

Cách chia theo video giúp tránh đưa frame của cùng một video vào nhiều split và giữ các video tương ứng giữa các phương pháp giả mạo được căn chỉnh.

Chạy tiền xử lý:

```bash
cd preprocess_deepfake
pip install -r requirements.txt
python preprocess_ffpp_mtcnn.py \
  --input_root "/path/to/FF++" \
  --output_root "/path/to/ffpp_faces" \
  --img_size 224 \
  --seed 42
```

Kết quả:

```text
ffpp_faces/
|-- train/
|   |-- original/
|   |-- Deepfakes/
|   |-- Face2Face/
|   |-- FaceSwap/
|   |-- NeuralTextures/
|   `-- FaceShifter/
|-- val/
`-- test/
```

### Celeb-DF-v2

Script Celeb-DF-v2 xử lý tập test được khai báo trong `List_of_testing_videos.txt` và lấy 50 frame/video.

```text
CelebDF-v2/
|-- Celeb-real/
|-- Celeb-synthesis/
|-- YouTube-real/
`-- List_of_testing_videos.txt
```

Theo định dạng danh sách test của Celeb-DF-v2, nhãn `1` được chuyển thành `real`, nhãn `0` thành `fake`.

```bash
cd preprocess_deepfake
python preprocess_celebdf_test_mtcnn.py \
  --input_root "/path/to/CelebDF-v2" \
  --output_root "/path/to/celebdf_faces" \
  --img_size 224 \
  --test_list "/path/to/CelebDF-v2/List_of_testing_videos.txt"
```

Kết quả có dạng:

```text
celebdf_faces/
`-- test/
    |-- real/
    `-- fake/
```

Tên mỗi ảnh chứa dataset, split, nguồn/lớp, định danh video, thứ tự mẫu và chỉ số frame. Cuối quá trình, script in số video/ảnh real-fake, số frame bị bỏ qua và số video lỗi.

## 2. Dataset và augmentation

`DeepfakeFrameDataset` đọc ảnh đệ quy với các phần mở rộng `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Quy ước nhãn dùng trong huấn luyện là:

- `original` hoặc `real`: **0**.
- `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter` hoặc `fake`: **1**.

Ảnh validation/test chỉ được resize và chuẩn hóa theo ImageNet. Ảnh train có thêm horizontal flip, rotation, affine scale/translation, biến đổi brightness/contrast, HSV/RGB, gamma, Gaussian noise, blur và JPEG compression. Có thể giảm tỷ lệ ảnh real bằng `train_real_percent` hoặc nhân bản có augmentation bằng `original_upsample_factor`.

## 3. Kiến trúc FA-ViT-CNN

Mô hình được cài đặt tại `training_model/models/favit_cnn.py`, gồm hai nhánh đặc trưng.

### Nhánh RGB FA-ViT

- Backbone mặc định là ViT-B/16 (`vit_base_patch16_224`) pretrained từ `timm`.
- Toàn bộ tham số backbone ViT được đóng băng.
- **GAM (Global Adaptive Module)** được chèn vào patch token trước mỗi block ViT. GAM dùng chuỗi convolution `1x1 -> 3x3 -> 1x1` theo dạng residual để học điều chỉnh đặc trưng toàn cục. Lớp cuối được khởi tạo bằng 0 nên ban đầu module gần với ánh xạ đồng nhất.
- Một CNN cục bộ trích xuất feature map từ ảnh RGB.
- **LAM (Local Adaptive Module)** tại các block `0`, `3`, `6` dùng cross-attention: patch token làm query, token từ CNN làm key/value. Hệ số residual `beta` được học và khởi tạo bằng 0.
- CLS token cuối cùng tạo vector đặc trưng RGB (768 chiều đối với ViT-B/16).

### Nhánh CNN phụ

Nhánh `CNN_feature_extractor_branch` gồm bốn convolution, BatchNorm, GELU, pooling và projection để tạo vector `freq_dim` (mặc định 128 chiều). Nhánh này được thiết kế cho dữ liệu tần số/nhiễu FFT hoặc SRM thông qua đối số `freq_x`.

Trong pipeline huấn luyện hiện tại, model được gọi bằng `model(images)` nên khi `freq_x` không được truyền, nhánh phụ **dùng lại chính ảnh RGB đã chuẩn hóa**. Vì vậy project chưa tự tính FFT/SRM trong dataset loader. Đặt `use_freq: false` để tắt nhánh này.

### Hợp nhất và phân loại

Đặc trưng CLS của FA-ViT được nối với đặc trưng CNN, sau đó đi qua:

```text
LayerNorm -> Dropout(0.3) -> Linear -> 1 logit
```

`forward()` trả về `(logits, fused_features)`. Logit chưa qua sigmoid; xác suất fake được tính bằng `sigmoid(logit)` và so với `threshold` (mặc định 0.5).

## 4. Huấn luyện FA-ViT-CNN

Cài đặt thư viện và chỉnh `training_model/configs/config.yaml`:

```bash
cd training_model
pip install -r requirements.txt
```

Cấu hình tối thiểu:

```yaml
data_root: "/path/to/ffpp_faces"
train_dir: "train"
val_dir: "val"
test_dir: "test"

image_size: 224
backbone: "favit_cnn"
pretrained: true
freq_in_channels: 3
freq_dim: 128
use_freq: true

batch_size: 16
epochs: 30
lr: 1e-4
weight_decay: 1e-4
device: "cuda"
```

Chạy train:

```bash
python train.py --config configs/config.yaml
```

Tiếp tục từ checkpoint:

```bash
python train.py --config configs/config.yaml --resume /path/to/checkpoint.pth
```

Quá trình huấn luyện sử dụng `BCEWithLogitsLoss`, optimizer AdamW, label smoothing tùy chọn, `ReduceLROnPlateau` và early stopping theo validation accuracy. Có thể bật **Forgery-Aware Loss** bổ sung bằng:

```yaml
lambda_fal: 0.1
fal_margin: 0.25
fal_scale: 32
```

FALoss khuyến khích đặc trưng real gần prototype lấy từ trọng số classifier hơn đặc trưng fake. Nếu batch không có đủ cả hai lớp, thành phần loss này bằng 0.

## 5. Đánh giá

Đánh giá trên test split của dữ liệu gốc:

```bash
python test_origin_dataset.py \
  --config configs/config.yaml \
  --checkpoint /path/to/best_model.pth
```

Đánh giá chéo trên dataset cấu hình bởi `cross_dataset_root`:

```bash
python test_cross_dataset.py \
  --config configs/config.yaml \
  --checkpoint /path/to/best_model.pth
```

Các script báo cáo Accuracy, F1, Precision, Recall, AUC và confusion matrix; có thể dùng `--output-csv` để lưu dự đoán từng ảnh.

## Lưu ý

- `image_size` khi train nên trùng với kích thước crop khi preprocess; cấu hình mặc định phù hợp ViT-B/16 là 224.
- Pretrained ViT có thể cần tải trọng số ở lần chạy đầu tiên.
- Đây là dự đoán ở mức frame. Nếu cần kết quả ở mức video, cần gom xác suất của nhiều frame (ví dụ mean/median) ở bước inference.
- Đường dẫn dữ liệu trong `configs/config.yaml` hiện là đường dẫn máy cục bộ và cần được thay bằng đường dẫn của môi trường chạy.
