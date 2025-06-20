# pose_classifier.py

import torch
import torch.nn as nn

class PoseClassifier(nn.Module):
    def __init__(self, input_dim=99, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
