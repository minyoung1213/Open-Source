import pandas as pd
import numpy as np
import json

# CSV 파일 읽기
df = pd.read_csv("pose_data1.csv")

# 라벨 인코딩 (문자 → 숫자)
labels = df['label'].unique()
label_map = {name: idx for idx, name in enumerate(labels)}
print("라벨 매핑:", label_map)

# 데이터 분리
X = df.drop(columns=["label"]).to_numpy().astype(np.float32)
y = df['label'].map(label_map).to_numpy().astype(np.int64)

# 저장
np.save("X.npy", X)
np.save("y.npy", y)

# 라벨 맵 저장
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("변환 완료 → X.npy, y.npy, label_map.json")
