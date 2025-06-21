import torch
import numpy as np
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from train_pose_classifier import PoseClassifier  # 학습 시 정의한 모델 클래스

# 1. 라벨 매핑 로드
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_map_rev = {v: k for k, v in label_map.items()}  # 숫자 → 문자열 라벨

# 2. 데이터 로드
X = np.load("X.npy")
y = np.load("y.npy")

# 3. 모델 정의 및 로딩
model = PoseClassifier(input_dim=99, num_classes=len(label_map))
model.load_state_dict(torch.load("pose_model_new.pt"))
model.eval()

# 4. 예측
with torch.no_grad():
    inputs = torch.tensor(X, dtype=torch.float32)
    outputs = model(inputs)
    preds = outputs.argmax(dim=1).numpy()

# 5. 결과 출력
print("전체 데이터 예측 완료")
print(f"Accuracy: {accuracy_score(y, preds):.2%}\n")

print("Classification Report:")
print(classification_report(
    y, preds, target_names=[label_map_rev[i] for i in sorted(label_map_rev)]
))

print("Confusion Matrix:")
print(confusion_matrix(y, preds))
