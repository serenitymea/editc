import cv2
import numpy as np
from typing import Generator, Tuple, Optional


class VideoLoader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise IOError(f"can't open video: {path}")

        self.fps = self._get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

        print(f"[VL] open fps={self.fps} frames={self.frame_count}")

    def _get(self, prop: int) -> float:
        value = self.cap.get(prop)
        return float(value) if value > 0 else 0.0

    def frames(self, start: int = 0, end: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        print(f"[VL] frames start={start} end={end}")

        if self.frame_count <= 0:
            print("[VL] no frames")
            return

        start = max(0, start)
        end = min(end if end else self.frame_count, self.frame_count)

        if start >= end:
            print("[VL] empty range")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for index in range(start, end):
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.size == 0:
                print(f"[VL] read failed at {index}")
                break

            yield index, frame

        print("[VL] frames done")

    def release(self) -> None:
        if self.cap:
            print("[VL] release")
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()