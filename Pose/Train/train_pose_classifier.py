# train_pose_classifier.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pose_classifier import PoseClassifier
import os

# 경로 설정
base_path = os.path.dirname(os.path.abspath(__file__))
X_path = os.path.join(base_path, "..", "Data", "X.npy")
y_path = os.path.join(base_path, "..", "Data", "y.npy")
save_path = os.path.join(base_path, "..", "pose_model_new.pt")


# 데이터 불러오기
X = np.load(X_path)
y = np.load(y_path)

# Tensor 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train/Val 분할
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 모델 정의
model = PoseClassifier(input_dim=99, num_classes=len(np.unique(y)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습 루프
epochs = 30
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 검증
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Val Acc: {acc:.2%}")

# 모델 저장 (state_dict 방식)
torch.save(model.state_dict(), save_path)
print(f"모델 학습 완료 및 저장: {save_path}")

