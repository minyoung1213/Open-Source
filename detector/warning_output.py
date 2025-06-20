import pyttsx3
import cv2

class WarningOutput:
    def __init__(self):
        self.triggered = False
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def speak_once(self):
        if not self.triggered:
            self.engine.say("쓰레기 버리지 마세요.")
            self.engine.runAndWait()
            self.triggered = True
