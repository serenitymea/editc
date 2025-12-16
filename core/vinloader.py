import cv2

class VideoLoader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("Video open error")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self, start=0, end=None):
        if end is None:
            end = self.frame_count

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start

        while i < end:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield i, frame
            i += 1

    def get_frame(self, frame_id):
        if frame_id < 0 or frame_id >= self.frame_count:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
