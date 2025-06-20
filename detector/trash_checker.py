import numpy as np
from collections import deque

class TrashChecker:
    def __init__(self, trash_box):
        self.trash_box = trash_box
        self.detected = False

    def is_touching(self, wrist_x, wrist_y):
        x1, y1, x2, y2 = self.trash_box
        touching = x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2
        print(f"[DEBUG] wrist=({wrist_x:.3f},{wrist_y:.3f}) / box=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) / touching={touching}")
        return touching

    def update(self, wrist_x, wrist_y):
        print("[DEBUG] update() called")

        if self.detected:
            print("[DEBUG] 이미 감지됨 → True 반환")
            return True

        if self.is_touching(wrist_x, wrist_y):
            print("박스 안에 손이 들어옴 → 감지 성공!")
            self.detected = True
        else:
            print("박스 밖 → 감지 불가")

        return self.detected