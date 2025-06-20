import pyttsx3
import cv2

class WarningOutput:
    def __init__(self):
        self.triggered = False
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def show_on_frame(self, frame):
        cv2.putText(frame, "Illegal Dumping Detected!", (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 5)

    def speak_once(self):
        if not self.triggered:
            self.engine.say("쓰레기 버리지 마세요.")
            self.engine.runAndWait()
            self.triggered = True