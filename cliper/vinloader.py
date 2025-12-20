import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Any


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

    def _get(self, prop: int) -> float:
        value = self.cap.get(prop)
        return float(value) if value > 0 else 0.0

    def frames(
        self,
        start: int = 0,
        end: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:

        if start < 0:
            start = 0

        if end is None or end > self.frame_count:
            end = self.frame_count

        if start >= end:
            return iter(())

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for index in range(start, end):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield index, frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
