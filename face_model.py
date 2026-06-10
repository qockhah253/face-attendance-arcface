import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np

class ArcFaceModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.bn  = nn.BatchNorm1d(2048)
        self.fc  = nn.Linear(2048, embedding_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x).squeeze(-1).squeeze(-1)
        x = self.bn(x)
        x = self.fc(x)
        x = self.bn2(x)
        return F.normalize(x, dim=1)

class FaceRecognizer:
    def __init__(self, model_path, threshold=0.45):
        self.device    = torch.device('cpu')
        self.threshold = threshold
        self.model     = ArcFaceModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        print("✅ FaceRecognizer loaded!")

    def detect_face(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * max(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(image.shape[1], x + w + pad)
        y2  = min(image.shape[0], y + h + pad)
        return image[y1:y2, x1:x2], (x1, y1, x2, y2)

    def get_embedding(self, face_img):
        img_rgb    = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img_tensor).cpu().numpy()[0]
        return emb

    def recognize(self, image, gallery):
        face, bbox = self.detect_face(image)
        if face is None:
            return None, 0.0, None, "Không phát hiện khuôn mặt"
        emb = self.get_embedding(face)
        if not gallery:
            return None, 0.0, emb, "Chưa có sinh viên đăng ký"
        best_student, best_score = None, -1
        for student in gallery:
            score = float(np.dot(emb, student['embedding']))
            if score > best_score:
                best_score  = score
                best_student = student
        if best_score >= self.threshold:
            return best_student, best_score, emb, None
        return None, best_score, emb, f"Không nhận ra (độ tin cậy: {best_score:.2f})"
