# license_plate_recognition.py
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==================== 1. Load 2 model ====================
# Model phát hiện biển số
detector = YOLO("runs/detect/license_plate_v1/weights/best.pt")  # YOLOv8

# Model nhận diện ký tự (CNN bạn đã train)
class CharacterCNN(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model OCR + mapping ký tự
ocr_model = CharacterCNN(num_classes=31)
checkpoint = torch.load("character_recognition_best.pth", map_location="cpu")
ocr_model.load_state_dict(checkpoint['model_state_dict'])
ocr_model.eval()

# Danh sách ký tự biển số VN (phải đúng thứ tự khi train!)
ALL_CHARS = 'ABCDĐEFGHIKLMNPQRSTUVXYZ0123456789'  # 31 ký tự
# Nếu bạn dùng thứ tự khác thì sửa lại cho khớp với lúc train nhé!

# ==================== 2. Hàm tiền xử lý + tách ký tự (copy từ notebook 03) ====================
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_h = 64
    new_w = int(w * new_h / h)
    gray = cv2.resize(gray, (new_w, new_h))
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary, gray

def segment_characters(binary, gray):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    h, w = binary.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch
        if 0.15 < aspect < 1.0 and ch > h*0.35 and cw > 6:
            char = gray[max(0,y-3):y+ch+3, max(0,x-3):x+cw+3]
            if char.size > 0:
                char = cv2.resize(char, (28, 28))
                chars.append((x, char))
    chars = sorted(chars, key=lambda x: x[0])  # sort left → right
    return [c[1] for c in chars]

# ==================== 3. Hàm End-to-End ====================
def recognize_plate(image_path, conf_threshold=0.4):
    img = cv2.imread(image_path)
    if img is None:
        return "Không đọc được ảnh"
    
    # Bước 1: Detect biển số
    results = detector(img, conf=conf_threshold, verbose=False)
    
    plate_text = ""
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        plate_img = img[y1:y2, x1:x2]
        
        # Bước 2: Tách + nhận diện ký tự
        binary, gray = preprocess_plate(plate_img)
        chars = segment_characters(binary, gray)
        
        if len(chars) >= 6:  # ít nhất 6 ký tự mới hợp lệ
            with torch.no_grad():
                for char_img in chars:
                    tensor = torch.from_numpy(char_img).float().unsqueeze(0).unsqueeze(0) / 255.0
                    tensor = (tensor - 0.5) / 0.5  # chuẩn hóa giống lúc train
                    output = ocr_model(tensor)
                    pred = output.argmax(1).item()
                    plate_text += ALL_CHARS[pred]
    
    return plate_text if plate_text else "Không tìm thấy biển số"

# ==================== 4. Test nhanh ====================
if __name__ == "__main__":
    img_path = "test_images/001.jpg"  # đổi thành ảnh của bạn
    result = recognize_plate(img_path)
    print(f"Biển số: {result}")