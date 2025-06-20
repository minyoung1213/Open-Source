import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import json
import sys
import os
sys.path.append(os.path.abspath("./Train"))
from Train.pose_classifier import PoseClassifier

# 설정
model_path = r"C:\Open-Source\Pose\pose_model.pt"
label_map_path = r"C:\Open-Source\Pose\Data\label_map.json"

# 라벨 맵 로드
with open(label_map_path, "r") as f:
    label_map = json.load(f)
label_map_rev = {v: k for k, v in label_map.items()}

# 모델 로드
model = PoseClassifier(input_dim=99, num_classes=len(label_map))
model.load_state_dict(torch.load(model_path))
model.eval()

# MediaPipe 포즈
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    label = "No Person"
    alert = False

    if results.pose_landmarks:
        pose_vector = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
        pose_tensor = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(pose_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = probs.argmax().item()
            label = label_map_rev[pred_class]

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if label == "dumping":
            alert = True
            # 테두리
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
            # 강조 텍스트
            cv2.putText(frame, "!!! DUMPING DETECTED !!!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # 일반 예측 텍스트
    cv2.putText(frame, f"Prediction: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if not alert else (0, 0, 255), 2)

    cv2.imshow("Pose Prediction - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
