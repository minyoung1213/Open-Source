#파일 확장자로 이미지,영상 자동 구분해서 추출, 판별하도록 수정 + tts 추가함
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import json
import pyttsx3
from pose_classifier import PoseClassifier
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 그 아래에서 경로 문제 없이 import 가능
from detector.warning_output import WarningOutput
import mediapipe as mp

warner = WarningOutput()  # ← TTS 객체 선언
# 설정
model_path = "Data/pose_model.pt"
label_map_path = "Data/label_map.json"
input_path = "Data/test1.MOV"   # ← 여기만 바꾸면 됨 (사진 or 영상 파일)

# 라벨맵 & 모델 로드 
with open(label_map_path, "r") as f:
    raw_label_map = json.load(f)
label_map = {v: k for k, v in raw_label_map.items()}

model = PoseClassifier(input_dim=99, num_classes=len(label_map))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# MediaPipe 초기화 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

# 확장자 기반으로 이미지 vs 영상 구분 
ext = os.path.splitext(input_path)[1].lower()
is_image = ext in ['.jpg', '.jpeg', '.png']

if is_image:
    # 이미지 처리 
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {input_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        print("⚠️ 포즈 감지 실패: 사람이 인식되지 않았습니다.")
    else:
        pose_vector = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
        pose_tensor = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(pose_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = probs.argmax().item()
            pred_label = label_map[pred_class] 

        print(f"\n🧠 예측 결과: {pred_label}")
        print("📊 클래스별 확률:")
        for i, p in enumerate(probs):
            print(f"  {label_map[i]:10s}: {p.item():.2%}")

        annotated = image.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(annotated, f"Prediction: {pred_label}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Pose Prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    # 영상 처리
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 영상을 열 수 없습니다: {input_path}")

    pose = mp_pose.Pose(static_image_mode=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 회전 + 크기 조절 (추가)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 영상 눕혀서 나올 때 회전
        frame = cv2.resize(frame, (480, 640))  # 창 너무 클 때 리사이즈

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            pose_vector = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
            if len(pose_vector) == 99:
                pose_tensor = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(pose_tensor)
                    probs = F.softmax(output, dim=1)[0]
                    pred_class = probs.argmax().item()
                    pred_label = label_map[pred_class] 

                cv2.putText(frame, f"Prediction: {pred_label}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255) if pred_label == "dumping" else (0, 255, 0), 3)
                if pred_label == "dumping":
                    warner.speak_once()
                else:
                    warner.triggered = False

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
