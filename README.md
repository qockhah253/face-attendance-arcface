# Face Attendance — ArcFace

> Hệ thống điểm danh tự động bằng nhận diện khuôn mặt  
> Báo cáo cuối kì môn **Nhập môn AI**
> Khoa Vật lý · Trường ĐH Khoa học Tự nhiên, ĐHQGHN

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## Tổng quan

Hệ thống nhận diện khuôn mặt open-set, cho phép điểm danh tự động không cần re-train khi thêm sinh viên mới. Kiến trúc sử dụng **ResNet50** làm backbone kết hợp **ArcFace Loss** để học face embedding trên không gian hypersphere, đạt độ phân biệt cao hơn đáng kể so với Softmax hay Triplet Loss thông thường.

### Kết quả thực nghiệm

| Phương pháp | Accuracy (open-set, 1.000 người) |
|---|---|
| CNN + Triplet Loss | 36.71% |
| **ResNet50 + ArcFace Loss** | **94.74%** ✅ |

Training dataset: **MS1M-ArcFace** · GPU: Kaggle Tesla T4

---

## Kiến trúc mô hình

```
Input: Ảnh khuôn mặt (112 × 112 × 3)
           │
    ┌──────▼──────┐
    │  ResNet50   │  ← Backbone (50 layers, pre-trained weights = None)
    │  Backbone   │
    └──────┬──────┘
           │  2048-dim feature vector
    ┌──────▼──────────────────┐
    │  Embedding Head         │
    │  BN → FC(2048→512) → BN │
    │  → L2 Normalize         │
    └──────┬──────────────────┘
           │  512-dim unit vector
      ┌────┴────┐
      │         │
  ArcFace    Cosine
  (Train)   Similarity
             (Inference)
```

Tại inference, mỗi khuôn mặt được ánh xạ thành một vector 512 chiều trên hypersphere đơn vị. Nhận diện bằng **cosine similarity** — không cần classifier cố định — nên thêm sinh viên mới chỉ cần đăng ký embedding, không cần re-train.

---

## Tính năng

- **Đăng ký sinh viên** — chụp ảnh qua webcam, lưu embedding vào SQLite
- **Điểm danh realtime** — nhận diện mỗi 2 giây, tự động ghi trạng thái *Đúng giờ / Muộn*
- **Lịch sử điểm danh** — lọc theo ngày hoặc mã sinh viên
- **Quản lý sinh viên** — xem danh sách, xóa sinh viên
- **REST API** — toàn bộ logic được expose qua JSON API, tách biệt với frontend

---

## Cấu trúc thư mục

```
face-attendance-arcface/
├── app.py              # Flask app + REST API routes
├── face_model.py       # ArcFaceModel (ResNet50 + Embedding Head) & FaceRecognizer
├── database.py         # SQLite handler (students, attendance)
├── requirements.txt
├── models/             # ← Đặt file .pth vào đây (xem phần Cài đặt)
│   ├── arcface_epoch20.pth
│   └── arcface_head_epoch20.pth
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   ├── history.html
│   └── students.html
└── static/
    ├── css/style.css
    └── js/main.js
```

> **Lưu ý:** File model weights (`.pth`) không được lưu trong repo do giới hạn kích thước GitHub.

---

## Cài đặt & Chạy

### Yêu cầu
- Python 3.9+
- Webcam (để đăng ký & điểm danh)

### Các bước

```bash
# 1. Clone repo
git clone https://github.com/qockhah253/face-attendance-arcface.git
cd face-attendance-arcface

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Tải model weights
#    → Tải arcface_epoch20.pth từ Google Drive (link bên dưới)
#    → Đặt vào thư mục models/
mkdir -p models
# Tải tại: https://drive.google.com/...

# 4. Chạy
python app.py
```

Mở trình duyệt tại **http://localhost:5000**

---

## API Reference

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/register` | Đăng ký sinh viên mới (name, student_id, class_name, image base64) |
| `POST` | `/api/attendance` | Nhận diện và ghi điểm danh (image base64, late flag) |
| `GET` | `/api/history` | Lịch sử điểm danh (query: `date`, `student_id`) |
| `GET` | `/api/students` | Danh sách sinh viên |
| `DELETE` | `/api/students/<id>` | Xóa sinh viên |

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Deep Learning | PyTorch, ResNet50, ArcFace Loss |
| Face Detection | OpenCV Haar Cascade |
| Web Backend | Flask |
| Database | SQLite |
| Frontend | Bootstrap 5, Vanilla JS |
| Training | Kaggle (Tesla T4 GPU) |

---

