#초기 테스트용
import cv2
import numpy as np
from collections import deque
import pyttsx3
import mediapipe as mp
import torch  
from detector.trash_checker import TrashChecker
from detector.warning_output import WarningOutput

#설정값
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
TRASH_BOX = [0.0, 0.0, 1.0, 1.0]  #비율 기준 [x1, y1, x2, y2]

# 모듈 초기화
checker = TrashChecker(trash_box=TRASH_BOX)
warner = WarningOutput()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# YOLO 모델 로드 
import os
import pathlib

# Windows PosixPath 문제 해결
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    yolo_model.conf = 0.5  
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('best.pt')
        print("ultralytics YOLO 모델로 대체 로드 성공")
    except ImportError:
        print("ultralytics 패키지가 설치되지 않았습니다. 'pip install ultralytics' 실행하세요.")
        exit()
finally:
    pathlib.PosixPath = temp

# 쓰레기봉투 검출 함수 
def detect_trash_bags(frame):
    results = yolo_model(frame)
    detections = results.pandas().xyxy[0]
    trash_boxes = []
    for _, detection in detections.iterrows():
        if detection['confidence'] > 0.5:
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            trash_boxes.append([x1, y1, x2, y2])
    return trash_boxes

# 손이 쓰레기 근처에 있는지 확인 함수 
def is_hand_near_trash(wrist_x, wrist_y, trash_boxes, frame_width, frame_height, threshold=50):
    hand_pixel_x = int(wrist_x * frame_width)
    hand_pixel_y = int(wrist_y * frame_height)
    
    for x1, y1, x2, y2 in trash_boxes:
        if (x1 - threshold) <= hand_pixel_x <= (x2 + threshold) and (y1 - threshold) <= hand_pixel_y <= (y2 + threshold):
            return True
    return False

# 웹캠 불러오기
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("❌ 웹캠 열기 실패")
    exit()

# 웹캠 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("AI 쓰레기 감지", cv2.WINDOW_NORMAL)
cv2.moveWindow("AI 쓰레기 감지", 100, 100)

# 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("웹캠에서 프레임을 읽을 수 없습니다")
        break

    FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

    # YOLO로 쓰레기봉투 검출 
    trash_boxes = detect_trash_bags(frame)
    
    # 쓰레기봉투 박스 그리기 
    for x1, y1, x2, y2 in trash_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 박스
        cv2.putText(frame, 'Trash Bag', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    #MediaPipe 처리 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        wrist_x, wrist_y = right_wrist.x, right_wrist.y

        # 기존 trash_box 체크 또는 새로운 쓰레기봉투 근처 체크 
        if checker.update(wrist_x, wrist_y) or is_hand_near_trash(wrist_x, wrist_y, trash_boxes, FRAME_WIDTH, FRAME_HEIGHT):
            warner.show_on_frame(frame)
            warner.speak_once()

    #영상에 쓰레기 영역 시각화 (예시 박스) - 기존 코드 유지
    x1 = int(TRASH_BOX[0] * FRAME_WIDTH)
    y1 = int(TRASH_BOX[1] * FRAME_HEIGHT)
    x2 = int(TRASH_BOX[2] * FRAME_WIDTH)
    y2 = int(TRASH_BOX[3] * FRAME_HEIGHT)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("AI 쓰레기 감지", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()