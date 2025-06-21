# main.py
import cv2
import numpy as np
import mediapipe as mp
import torch
import json
import pathlib
import os

from detector.trash_checker import TrashChecker
from detector.warning_output import WarningOutput
from Pose.Train.pose_classifier import PoseClassifier

# 설정값
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
TRASH_BOX = [0.0, 0.0, 1.0, 1.0]

# 초기화
checker = TrashChecker(trash_box=TRASH_BOX)
warner = WarningOutput()

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Path 호환성 처리
_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLO 모델 로드
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    yolo_model.conf = 0.5
except Exception as e:
    from ultralytics import YOLO
    yolo_model = YOLO('best.pt')
finally:
    pathlib.PosixPath = _temp

# PoseClassifier state_dict 방식 로드
def load_pose_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    pose_model_path = os.path.join(base_path, "Pose", "Train", "pose_model_new.pt")
    label_map_path = os.path.join(base_path, "Pose", "Data", "label_map.json")

    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    label_map_rev = {v: k for k, v in label_map.items()}

    model = PoseClassifier(input_dim=99, num_classes=len(label_map))
    model.load_state_dict(torch.load(pose_model_path, map_location="cpu"))
    model.eval()
    return model, label_map_rev

pose_model, label_map_rev = load_pose_model()

# YOLO 객체 인식 함수
def detect_trash_bags(frame):
    results = yolo_model(frame)
    detections = results.pandas().xyxy[0]
    trash_boxes = []
    for _, detection in detections.iterrows():
        if detection['confidence'] > 0.5:
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            trash_boxes.append([x1, y1, x2, y2])
    return trash_boxes

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠 열기 실패")
    exit()

cv2.namedWindow("AI DUMPING DETECTOR", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
    trash_boxes = detect_trash_bags(frame)

    for x1, y1, x2, y2 in trash_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Trash Bag', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    dumping_condition = False
    alert = False
    pred_label = "No Person"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        pose_vector = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]

        if len(pose_vector) == 99:
            tensor = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = pose_model(tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_class = probs.argmax().item()
                pred_label = label_map_rev[pred_class]

        # 관절 기반 조건 확인
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        wrist_knee = rw.y <= rk.y  # 손목이 무릎 아래
        hip_bent = lh.y >= ls.y    # 허리 굽힘
        wrist_x, wrist_y = rw.x, rw.y

        # 추가 모션기반 감지
        checker_detected = checker.update(wrist_x, wrist_y, trash_boxes, FRAME_WIDTH, FRAME_HEIGHT)

        gesture_suspected = wrist_knee or hip_bent or checker_detected

        # dumping 조건 판단
        dumping_condition = (pred_label == "dumping" or gesture_suspected) and len(trash_boxes) > 0

    if dumping_condition:
        alert = True
        frame = warner.show_on_frame(frame)

        if not warner.triggered:
            warner.trigger_warning()

        pred_label = "dumping"  # 빨간 글씨용
    else:
        alert = False
        warner.reset()
        pred_label = "normal"  # 초록 글씨용

    # --- 결과 출력 ---
    # 확률 추출
    confidence_percent = probs[pred_class].item() * 100
    cv2.putText(frame, f"Pose: {pred_label} ({confidence_percent:.1f}%)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if alert else (0, 255, 0), 2)

    # 다른 클래스 확률 표시 (dumping, normal 모두 표시)
    y_offset = 80
    for i, prob in enumerate(probs):
        label = label_map_rev[i]
        percent = prob.item() * 100
        cv2.putText(frame, f"{label}: {percent:.1f}%", (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        y_offset += 30

    
    cv2.imshow("AI 쓰레기 감지", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
