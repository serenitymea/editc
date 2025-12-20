import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional


@dataclass(frozen=True)
class Clip:
    start_frame: int
    end_frame: int
    fps: float
    score: float

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_time(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class EpicDetector:
    MOTION_WEIGHT = 0.6
    AUDIO_WEIGHT = 0.4
    AUDIO_GAIN = 40
    SMOOTH_SIGMA = 3

    def __init__(self, loader, audio_analyzer):
        self.loader = loader
        self.audio = audio_analyzer

        self.cap = loader.cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frames_count / self.fps

    def detect_for_edit(
        self,
        clip_duration: float = 0.5,
        target_duration: Optional[float] = None,
    ) -> List[Clip]:

        sync_points = self._get_sync_points()
        scores = self._compute_scores()
        clip_frames = int(clip_duration * self.fps)

        selected: List[Clip] = []
        total_duration = 0.0

        for beat_time in sync_points:
            if target_duration is not None and total_duration >= target_duration:
                break

            clip = self._find_best_clip_at_beat(beat_time, clip_frames, scores)
            if clip and not self._overlaps(clip, selected):
                selected.append(clip)
                total_duration += clip.duration

        return sorted(selected, key=lambda c: c.start_time)

    def _get_sync_points(self) -> np.ndarray:
        beats = self.audio.beat_times()
        peaks = self.audio.peak_times()
        points = np.unique(np.concatenate([beats, peaks]))
        return points[points < self.duration]

    def _compute_scores(self) -> np.ndarray:
        scores = np.zeros(self.frames_count, dtype=np.float32)
        audio_energy = np.array([
            self.audio.audio_energy(i, self.frames_count)
            for i in range(self.frames_count)
        ])

        prev_gray = None
        for i, frame in self.loader.frames():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                motion = cv2.absdiff(prev_gray, gray).mean()
                scores[i] = (
                    motion * self.MOTION_WEIGHT +
                    audio_energy[i] * self.AUDIO_GAIN * self.AUDIO_WEIGHT
                )
            prev_gray = gray

        return gaussian_filter1d(scores, sigma=self.SMOOTH_SIGMA)

    def _find_best_clip_at_beat(
        self,
        beat_time: float,
        clip_frames: int,
        scores: np.ndarray,
        search_forward_sec: float = 1.0
    ) -> Optional[Clip]:

        target_frame = int(beat_time * self.fps)
        window = int(search_forward_sec * self.fps)
        start = target_frame
        end = min(self.frames_count - clip_frames, target_frame + window)

        if start >= end:
            return None

        best_start = start
        best_score = -np.inf

        for s in range(start, end):
            segment_score = scores[s:s + clip_frames].mean()
            if segment_score > best_score:
                best_score = segment_score
                best_start = s

        return Clip(
            start_frame=best_start,
            end_frame=best_start + clip_frames - 1,
            fps=self.fps,
            score=best_score
        )

    @staticmethod
    def _overlaps(clip: Clip, clips: List[Clip]) -> bool:
        return any(
            not (clip.end_frame < c.start_frame or clip.start_frame > c.end_frame)
            for c in clips
        )
