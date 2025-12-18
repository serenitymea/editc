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
        self.duration = self.frames_count / self.fps

    def detect_for_edit(
        self,
        target_duration: float = 15.0,
        clip_duration: float = 0.5,
        motion_thresh: float = 30,
        keep_top_percent: int = 80,
    ):

        beat_times = self.audio.beat_times()
        peak_times = self.audio.peak_times()

        sync_points = self._prepare_sync_points(
            beat_times, peak_times, target_duration
        )

        scores = self._analyze_video(motion_thresh)
        clip_frames = int(clip_duration * self.fps)

        clips = [
            clip
            for t in sync_points
            if (clip := self._find_best_clip(t, clip_frames, scores))
        ]

        return self._filter_clips(clips, keep_top_percent)

    def _prepare_sync_points(self, beats, peaks, duration):
        points = np.concatenate([beats, peaks])
        points = points[points < duration]
        return np.unique(np.round(points, 2))

    def _analyze_video(self, motion_thresh):
        scores = np.zeros(self.frames_count, dtype=np.float32)
        prev_gray = None

        for i, frame in self.loader.frames():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                motion = cv2.absdiff(prev_gray, gray).mean()
                audio_energy = self.audio.audio_energy(i, self.frames_count)

                scores[i] = (
                    motion * 0.6 +
                    audio_energy * 40 * 0.4
                )

            prev_gray = gray

        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(scores, sigma=3)

    def _find_best_clip(self, target_time, clip_frames, scores):
        search_window_sec = 2.0

        target_frame = int(target_time * self.fps)
        window = int(search_window_sec * self.fps)

        start = max(0, target_frame - window)
        end = min(self.frames_count - clip_frames, target_frame + window)

        if start >= end:
            return None

        best_score = -np.inf
        best_start = None

        for s in range(start, end):
            segment = scores[s : s + clip_frames]
            segment_score = segment.mean()

            time_offset = abs(s - target_frame) / self.fps
            proximity = max(0.0, 1.0 - time_offset / search_window_sec)

            score = segment_score * (0.7 + 0.3 * proximity)

            if score > best_score:
                best_score = score
                best_start = s

        if best_start is None:
            return None

        return Clip(
            start_frame=best_start,
            end_frame=best_start + clip_frames - 1,
            fps=self.fps,
            score=best_score,
        )

    def _filter_clips(self, clips, keep_top_percent):
        if not clips:
            return []

        clips = sorted(clips, key=lambda c: c.score, reverse=True)
        keep = max(1, int(len(clips) * keep_top_percent / 100))

        selected = clips[:keep]
        return sorted(selected, key=lambda c: c.start_time)
