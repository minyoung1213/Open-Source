from collections import deque

class TrashChecker:
    def __init__(self, trash_box, stability_frames=5, reset_frames=30):
        self.trash_box = trash_box  # 기본 box 비율 (x1, y1, x2, y2) → 비율
        self.detected = False
        self.stability_frames = stability_frames
        self.reset_frames = reset_frames
        
        self.touch_count = 0
        self.no_touch_count = 0
        self.position_history = deque(maxlen=10)

    def is_touching(self, wrist_x, wrist_y, trash_boxes, frame_width, frame_height, threshold=50):
        # 비율형 기본 박스 (전체 화면 중 특정 영역)도 계산
        tb_x1 = int(self.trash_box[0] * frame_width)
        tb_y1 = int(self.trash_box[1] * frame_height)
        tb_x2 = int(self.trash_box[2] * frame_width)
        tb_y2 = int(self.trash_box[3] * frame_height)

        # 손 위치 (pixel 단위)
        hand_x = int(wrist_x * frame_width)
        hand_y = int(wrist_y * frame_height)

        # 1. 고정 박스 내부에 있는지
        if tb_x1 <= hand_x <= tb_x2 and tb_y1 <= hand_y <= tb_y2:
            self.position_history.append((hand_x, hand_y, True))
            return True

        # 2. YOLO 박스들 중 하나라도 근처에 있는지
        for x1, y1, x2, y2 in trash_boxes:
            if (x1 - threshold) <= hand_x <= (x2 + threshold) and (y1 - threshold) <= hand_y <= (y2 + threshold):
                self.position_history.append((hand_x, hand_y, True))
                return True

        # 아닌 경우
        self.position_history.append((hand_x, hand_y, False))
        return False

    def get_average_position(self):
        if not self.position_history:
            return None, None, False

        x_avg = sum(pos[0] for pos in self.position_history) / len(self.position_history)
        y_avg = sum(pos[1] for pos in self.position_history) / len(self.position_history)
        touch_ratio = sum(pos[2] for pos in self.position_history) / len(self.position_history)
        return x_avg, y_avg, touch_ratio > 0.6

    def update(self, wrist_x, wrist_y, trash_boxes, frame_width, frame_height):
        is_touching_now = self.is_touching(wrist_x, wrist_y, trash_boxes, frame_width, frame_height)
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
