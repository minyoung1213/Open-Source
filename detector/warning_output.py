import cv2
import pyttsx3
import threading

class WarningOutput:
    def __init__(self):
        self.triggered = False
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 1.0)

    def show_on_frame(self, frame):
        cv2.putText(frame, "WARNING", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        return frame

    def trigger_warning(self):
        if not self.triggered:
            self.triggered = True
            threading.Thread(target=self._speak_warning).start()

    def _speak_warning(self):
        self.engine.say("경고. 무단투기 행위가 감지되었습니다.")
        self.engine.runAndWait()

    def reset(self):
        self.triggered = False
