import cv2

class VideoLoader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("op error")

    def frames(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, f = self.cap.read()
            if not ret or f is None:
                break
            yield f
            
    def get_frames(self, frame_id):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
        
