import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class Clip:
    start_frame: int
    end_frame: int
    fps: float
    score: float

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def duration(self):
        return self.end_time - self.start_time


class EpicDetector:
    def __init__(self, loader, audio_analyzer):
        self.loader = loader
        self.audio = audio_analyzer
        self.cap = loader.cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def detect(self, visual_thresh=60, audio_thresh=0.01, context=15, min_len=10):
        prev_gray = None
        mask = np.zeros(self.frames_count, dtype=bool)
        scores = np.zeros(self.frames_count, dtype=np.float32)

        for i, frame in self.loader.frames():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                v = cv2.absdiff(prev_gray, gray).mean()
                a = self.audio.get_energy(i, self.frames_count)

                if v > visual_thresh and a > audio_thresh:
                    start = max(0, i - context)
                    end = min(self.frames_count - 1, i + context)
                    mask[start:end + 1] = True
                    scores[start:end + 1] = np.maximum(scores[start:end + 1], v * a)

            prev_gray = gray

        return self._build_clips(mask, scores, min_len)

    def _build_clips(self, mask, scores, min_len):
        clips = []
        start = None

        for i, active in enumerate(mask):
            if active and start is None:
                start = i
            elif not active and start is not None:
                if i - start >= min_len:
                    clips.append(self._clip(start, i - 1, scores[start:i]))
                start = None

        if start is not None and self.frames_count - start >= min_len:
            clips.append(self._clip(start, self.frames_count - 1, scores[start:]))

        return clips

    def _clip(self, start, end, scores):
        score = float(np.max(scores))
        return Clip(start, end, self.fps, score)
