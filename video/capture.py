import cv2 as cv

class Capture:
    def __init__(self, source=0):
        self.capture = cv.VideoCapture(source)

    def read(self):
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Failed to read from capture source.")
        return frame

    def release(self):
        self.capture.release()