import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import json
import os
import sys

# 경로 설정
# test_dir은 테스트 이미지 들어있는 폴더
test_dir = r"C:\Users\jiyunae\OneDrive\Desktop\Sookmyung\test"
model_path = r"C:\Users\jiyunae\OneDrive\Desktop\Sookmyung\Open-Source\Pose\pose_model.pt"
label_map_path = r"C:\Users\jiyunae\OneDrive\Desktop\Sookmyung\Open-Source\Pose\Data\label_map.json"

# pose_classifier 임포트 경로 설정
sys.path.append(os.path.abspath("./Train"))
from pose_classifier import PoseClassifier

# 라벨 맵 로드
with open(label_map_path, "r") as f:
    label_map = json.load(f)
label_map_rev = {v: k for k, v in label_map.items()}

# 모델 로드
model = PoseClassifier(input_dim=99, num_classes=len(label_map))
model.load_state_dict(torch.load(model_path))
model.eval()

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 통계용 변수
total = 0
dumping_count = 0
fail_count = 0
failed_images = []

# 이미지 순회
for filename in os.listdir(test_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(test_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {filename}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        print(f"❌ 포즈 감지 실패: {filename}")
        fail_count += 1
        failed_images.append(filename)
        continue

    pose_vector = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
    pose_tensor = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(pose_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        pred_label = label_map_rev[pred_class]

    print(f"\n파일명: {filename}")
    print(f"- 예측 결과: {pred_label}")
    for i, p in enumerate(probs):
        print(f"  {label_map_rev[i]:10s}: {p.item():.2%}")

    if pred_label == "dumping":
        dumping_count += 1
    total += 1

# 결과 통계 출력
print("\n최종 통계")
print(f"총 이미지 수: {total + fail_count}")
print(f"예측 성공: {total}")
print(f"❌ 포즈 인식 실패: {fail_count}")
print(f"- 실패한 이미지 목록: {failed_images}")
print(f"- dumping 분류 수: {dumping_count}")
if total > 0:
    print(f"- dumping 비율: {dumping_count / total:.2%}")
