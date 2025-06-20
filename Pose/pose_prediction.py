import cv2
import torch
import torch.nn.functional as F
import numpy as np
import json
from pose_classifier import PoseClassifier
import mediapipe as mp

# 설정
image_path = r"C:\Users\jiyunae\OneDrive\Desktop\Sookmyung\test\1.jpg"
model_path = "../pose_model.pt"
label_map_path = "../label_map.json"

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
pose = mp_pose.Pose(static_image_mode=True)

# 이미지 로드
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)

if not results.pose_landmarks:
    print("⚠️ 포즈 감지 실패: 사람이 인식되지 않았습니다.")
else:
    # 좌표 → 모델 입력
    pose_vector = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
    pose_tensor = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)

    # 모델 예측
    with torch.no_grad():
        output = model(pose_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        pred_label = label_map_rev[pred_class]

    # 출력
    print(f"\n🧠 예측 결과: {pred_label}")
    print("📊 클래스별 확률:")
    for i, p in enumerate(probs):
        print(f"  {label_map_rev[i]:10s}: {p.item():.2%}")

    # 시각화 표시만
    annotated = image.copy()
    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.putText(annotated, f"Prediction: {pred_label}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 표시 (저장 X)
    cv2.imshow("Pose Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
