# Hệ thống Điểm danh Tự động bằng Nhận diện Khuôn mặt

> Báo cáo cuối kì môn Nhập môn AI — Nhóm 8  
> Khoa Vật lý, Trường ĐH Khoa học Tự nhiên, ĐHQGHN

## Giới thiệu

Hệ thống điểm danh tự động sử dụng nhận diện khuôn mặt dựa trên kiến trúc **ResNet50** kết hợp **ArcFace Loss**. Mô hình được huấn luyện trên tập dữ liệu MS1M-ArcFace, đạt độ chính xác **94.74%** trên 1.000 người chưa gặp (open-set recognition).

## Kết quả

| Phương pháp | Accuracy |
|-------------|----------|
| CNN + Triplet Loss | 36.71% |
| **ResNet50 + ArcFace Loss** | **94.74%** |

## Kiến trúc hệ thống

```
Ảnh khuôn mặt (112×112×3)
        ↓
ResNet50 Backbone (50 layers)
        ↓
Embedding Head: BN → FC(2048→512) → BN → L2 Norm
        ↓
Vector 512D (Unit Hypersphere)
       / \
      /   \
ArcFace   SVM Classifier
(Train)   (Inference)
```

## Cấu trúc thư mục

```
face-attendance-arcface/
├── app.py                    # Flask web application
├── face_model.py             # ResNet50 + Embedding Head model
├── database.py               # SQLite database handler
├── face-prject (1).ipynb     # Kaggle training notebook
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   ├── history.html
│   └── students.html
└── static/                   # CSS & JavaScript
    ├── css/style.css
    └── js/main.js
```

> **Lưu ý:** Model weights (`.pth`) không được lưu trong repo do giới hạn kích thước.  
> Tải model tại: [Google Drive](https://drive.google.com)

## Công nghệ sử dụng

- **Model**: PyTorch, ResNet50, ArcFace Loss, SVM (scikit-learn)
- **Web**: Flask, SQLite, Bootstrap 5
- **Face Detection**: OpenCV Haar Cascade
- **Training**: Kaggle Tesla T4 GPU

## Cài đặt & Chạy

```bash
# 1. Clone repo
git clone https://github.com/qockhah253/face-attendance-arcface.git
cd face-attendance-arcface

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Tải model weights vào thư mục models/
#    arcface_epoch20.pth
#    arcface_head_epoch20.pth

# 4. Chạy ứng dụng
python app.py
```

Mở trình duyệt tại `http://localhost:5000`

## Tính năng web app

- **Đăng ký sinh viên**: Nhập thông tin + chụp ảnh khuôn mặt qua webcam
- **Điểm danh tự động**: Nhận diện mỗi 2 giây, ghi trạng thái Đúng giờ / Muộn
- **Lịch sử điểm danh**: Lọc theo ngày và mã sinh viên
- **Quản lý sinh viên**: Xem danh sách, xóa sinh viên

## Thành viên nhóm

| Họ tên | MSSV |
|--------|------|
| Triệu Quốc Khánh | 23001615 |
| Nguyễn Thị Thu Hà | 23001603 |
| Lê Văn Đạt | 2300.... |
| Triệu Đình Dũng | 2300.... |
