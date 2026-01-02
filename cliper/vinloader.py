import cv2
import numpy as np
from typing import Generator, Tuple, Optional


class VideoLoader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise IOError(f"can't open video: {path}")

        self.fps: float = self._get(cv2.CAP_PROP_FPS)
        self.frame_count: int = int(self._get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.duration: float = (
            self.frame_count / self.fps
            if self.fps > 0 else 0.0
        )

        print(f"[VL] open fps={self.fps} frames={self.frame_count}")

    def _get(self, prop: int) -> float:
        value = self.cap.get(prop)
        return float(value) if value > 0 else 0.0

    def frames(
        self,
        start: int = 0,
        end: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:

        print(f"[VL] frames start={start} end={end}")

        if self.frame_count <= 0:
            print("[VL] no frames")
            return

        if start < 0:
            start = 0

        if end is None or end > self.frame_count:
            end = self.frame_count

        if start >= end:
            print("[VL] empty range")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        index = start
        while index < end:
            ret, frame = self.cap.read()

            if not ret:
                print(f"[VL] read failed at {index}")
                break

            if frame is None or frame.size == 0:
                print(f"[VL] bad frame at {index}")
                break

            yield index, frame
            index += 1

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