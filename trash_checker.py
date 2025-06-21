from collections import deque

class TrashChecker:
    def __init__(self, trash_box, stability_frames=5, reset_frames=30):
        self.trash_box = trash_box
        self.detected = False
        self.stability_frames = stability_frames
        self.reset_frames = reset_frames
        
        self.touch_count = 0
        self.no_touch_count = 0
        self.position_history = deque(maxlen=10)

    def is_touching(self, wrist_x, wrist_y):
        x1, y1, x2, y2 = self.trash_box
        touching = x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2
        self.position_history.append((wrist_x, wrist_y, touching))
        return touching

    def get_average_position(self):
        if not self.position_history:
            return None, None, False

        x_avg = sum(pos[0] for pos in self.position_history) / len(self.position_history)
        y_avg = sum(pos[1] for pos in self.position_history) / len(self.position_history)
        touch_ratio = sum(pos[2] for pos in self.position_history) / len(self.position_history)
        return x_avg, y_avg, touch_ratio > 0.6

    def update(self, wrist_x, wrist_y):
        is_touching_now = self.is_touching(wrist_x, wrist_y)
        avg_x, avg_y, is_stable_touch = self.get_average_position()

        if self.detected:
            if not is_touching_now:
                self.no_touch_count += 1
                if self.no_touch_count >= self.reset_frames:
                    print(f"[DEBUG] {self.reset_frames}프레임 동안 터치 없음 → 자동 리셋")
                    self.reset()
            else:
                self.no_touch_count = 0
            return True

        if is_stable_touch and is_touching_now:
            self.touch_count += 1
            if self.touch_count >= self.stability_frames:
                print(f"[DEBUG] {self.stability_frames}프레임 동안 안정적 터치 → 감지 성공!")
                self.detected = True
                return True
        else:
            self.touch_count = 0

        return False

    def reset(self):
        self.detected = False
        self.touch_count = 0
        self.no_touch_count = 0
        self.position_history.clear()
