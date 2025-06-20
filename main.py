#초기 테스트용
import cv2
import numpy as np
from collections import deque
import pyttsx3
import mediapipe as mp
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

# 영상 불러오기
cap = cv2.VideoCapture("data/test1.MOV")
if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()

cv2.namedWindow("AI 쓰레기 감지", cv2.WINDOW_NORMAL)
cv2.moveWindow("AI 쓰레기 감지", 100, 100)

# 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("인식됨")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

    #MediaPipe 처리 #모델학습 전 테스트용으로 임의설정해둠!
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        wrist_x, wrist_y = right_wrist.x, right_wrist.y

        if checker.update(wrist_x, wrist_y):
            warner.show_on_frame(frame)
            warner.speak_once()

    #영상에 쓰레기 영역 시각화 (예시 박스)
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
