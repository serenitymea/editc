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
    song_start_time: float
    features: Optional[dict] = None

    @property
    def start(self) -> float:
        return self.start_frame / self.fps

    @property
    def end(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration(self) -> float:
        return self.end - self.start


class TrainingDetector:

    MOTION_WEIGHT = 0.7
    AUDIO_WEIGHT = 0.3
    AUDIO_GAIN = 1.0
    SMOOTH_SIGMA = 1.5

    def __init__(self, loader, audio_analyzer, model=None, audio_loops: int = 1):
        self.loader = loader
        self.audio = audio_analyzer
        self.audio_loops = max(1, audio_loops)

        self.cap = loader.cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frames_count / self.fps
        self.model = model

        self._base_beat_intervals = self.audio.get_beat_intervals() or []
        self._base_audio_duration = (
            max(b[0] + b[1] for b in self._base_beat_intervals)
            if self._base_beat_intervals else 0.0
        )

    def detect_perfect_clips(self, max_clips=None) -> List[Clip]:

        print("\n[ED]strt...")

        scores, motion, audio = self._compute_epicness_scores()

        beat_intervals = self._loop_beat_intervals()

        if not beat_intervals:
            print("[ED]not found")
            return []

        print(f" [ED]found inter: {len(beat_intervals)}")

        clips = self._find_clips_for_all_beats(
            scores, motion, audio, beat_intervals, max_clips
        )

        return clips

    def _loop_beat_intervals(self) -> List[tuple]:
        if not self._base_beat_intervals or self.audio_loops == 1:
            return self._base_beat_intervals

        result = []
        for i in range(self.audio_loops):
            offset = i * self._base_audio_duration
            for start, dur in self._base_beat_intervals:
                result.append((start + offset, dur))
        return result

    def _compute_epicness_scores(self):

        scores = np.zeros(self.frames_count, dtype=np.float32)

        audio_energy = np.array(
            [
                self.audio.audio_energy(
                    i % self.frames_count,
                    self.frames_count
                )
                for i in range(self.frames_count)
            ],
            dtype=np.float32
        )

        print(" [ED]move analyz...")
        motion_scores = np.zeros(self.frames_count, dtype=np.float32)
        prev_gray = None
        FLOW_STEP = 2

        for i, frame in self.loader.frames():
            if i >= self.frames_count:
                break

            if i % 1000 == 0:
                print(f" [ED]prog: {i}/{self.frames_count} frames", end="\r")

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray_full, (320, 180))

            if prev_gray is not None and i % FLOW_STEP == 0:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion_scores[i] = np.mean(
                    np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                )
            elif i > 0:
                motion_scores[i] = motion_scores[i - 1]

            prev_gray = gray

        print()

        if motion_scores.max() > 0:
            motion_scores /= motion_scores.max()
        if audio_energy.max() > 0:
            audio_energy /= audio_energy.max()

        scores = (
            motion_scores * self.MOTION_WEIGHT +
            audio_energy * self.AUDIO_GAIN * self.AUDIO_WEIGHT
        )

        scores = gaussian_filter1d(scores, self.SMOOTH_SIGMA)

        return scores, motion_scores, audio_energy

    def _find_clips_for_all_beats(
        self,
        scores,
        motion,
        audio,
        beat_intervals,
        max_clips=None
    ):

        clips = []
        total_intervals = len(beat_intervals)

        if max_clips:
            total_intervals = min(total_intervals, max_clips)

        print(f"\n   [ED]f clips {total_intervals} beats...")

        for i in range(total_intervals):
            song_start, target_duration = beat_intervals[i]

            best_clip = self._find_best_segment(
                scores,
                motion,
                audio,
                target_duration,
                song_start,
                excluded_clips=clips
            )

            if best_clip:
                clips.append(best_clip)
                print(
                    f"      [ED]beat {i+1}: clip "
                    f"{best_clip.start:.2f}s-{best_clip.end:.2f}s "
                    f"(len: {best_clip.duration:.2f}s, "
                    f"score: {best_clip.score:.3f})"
                )

        return clips

    def _find_best_segment(
        self,
        scores,
        motion,
        audio,
        target_duration,
        song_start_time,
        excluded_clips
    ):

        clip_frames = int(target_duration * self.fps)

        if clip_frames >= self.frames_count:
            return None

        best_score = -1
        best_start = 0
        best_features = None

        step = max(1, int(self.fps * 0.1))

        for start_frame in range(0, self.frames_count - clip_frames, step):
            end_frame = start_frame + clip_frames

            if self._overlaps_with_any(start_frame, end_frame, excluded_clips):
                continue

            features = self._extract_features(
                start_frame, end_frame, motion, audio
            )
            heuristic = scores[start_frame:end_frame].mean()

            if self.model is not None:
                ml_score = self.model.predict_proba(
                    [list(features.values())]
                )[0][1]
                score = 0.5 * heuristic + 0.5 * ml_score
            else:
                score = heuristic

            if score > best_score:
                best_score = score
                best_start = start_frame
                best_features = features

        if best_score < 0:
            return None

        return Clip(
            start_frame=best_start,
            end_frame=best_start + clip_frames,
            fps=self.fps,
            score=best_score,
            song_start_time=song_start_time,
            features=best_features
        )

    def _extract_features(self, start, end, motion, audio):
        return {
            "motion_mean": float(motion[start:end].mean()),
            "motion_max": float(motion[start:end].max()),
            "motion_std": float(motion[start:end].std()),
            "audio_mean": float(audio[start:end].mean()),
            "audio_max": float(audio[start:end].max()),
            "audio_std": float(audio[start:end].std()),
            "duration": (end - start) / self.fps
        }

    def _overlaps_with_any(self, start_frame, end_frame, clips):

        for clip in clips:
            if not (end_frame < clip.start_frame or start_frame > clip.end_frame):
                return True
        return False
