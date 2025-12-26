import os
import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from typing import List

@dataclass(frozen=True)
class Clip:
    start_frame: int
    end_frame: int
    fps: float
    score: float
    song_start_time: float

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

    MOTION_WEIGHT = 0.7
    AUDIO_WEIGHT = 0.3
    AUDIO_GAIN = 1.0
    SMOOTH_SIGMA = 1.5
    
    def __init__(self, loader, audio_analyzer):
        self.loader = loader
        self.audio = audio_analyzer
        
        self.cap = loader.cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frames_count / self.fps
    
    def detect_perfect_clips(self, max_clips=None) -> List[Clip]:

        print("\n[ED]strt...")

        scores = self._compute_epicness_scores()

        beat_intervals = self.audio.get_beat_intervals()
        
        if not beat_intervals:
            print("[ED]not found")
            return []
        
        print(f" [ED]found inter: {len(beat_intervals)}")

        clips = self._find_clips_for_all_beats(scores, beat_intervals, max_clips)
        
        return clips
    
    def _compute_epicness_scores(self) -> np.ndarray:
        
        scores = np.zeros(self.frames_count, dtype=np.float32)

        audio_energy = np.array(
            [self.audio.audio_energy(i, self.frames_count)
             for i in range(self.frames_count)],
            dtype=np.float32
        )

        print(" [ED]move analyz...")
        motion_scores = np.zeros(self.frames_count, dtype=np.float32)
        prev_gray = None
        FLOW_STEP = 2
        
        for i, frame in self.loader.frames():
            if i >= self.frames_count:
                break
            
            if i % 100 == 0:
                print(f" [ED]prog: {i}/{self.frames_count} frames", end='\r')
            
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
        
        return scores
    
    def _find_clips_for_all_beats(
        self, 
        scores: np.ndarray, 
        beat_intervals: List[tuple],
        max_clips: int = None
    ) -> List[Clip]:

        clips = []
        total_intervals = len(beat_intervals)
        
        if max_clips:
            total_intervals = min(total_intervals, max_clips)
        
        print(f"\n   [ED]f clips {total_intervals} beats...")

        for i in range(total_intervals):
            song_start, target_duration = beat_intervals[i]

            best_clip = self._find_best_segment(
                scores, 
                target_duration, 
                song_start,
                excluded_clips=clips
            )
            
            if best_clip:
                clips.append(best_clip)
                print(f"      [ED]beat {i+1}: clip {best_clip.start_time:.2f}s-{best_clip.end_time:.2f}s "
                      f"(len: {best_clip.duration:.2f}s, score: {best_clip.score:.3f})")
        
        return clips
    
    def _find_best_segment(
        self, 
        scores: np.ndarray, 
        target_duration: float,
        song_start_time: float,
        excluded_clips: List[Clip]
    ) -> Clip:

        clip_frames = int(target_duration * self.fps)
        
        if clip_frames >= self.frames_count:
            return None
        
        best_score = -1
        best_start = 0

        for start_frame in range(0, self.frames_count - clip_frames, max(1, int(self.fps * 0.1))):
            end_frame = start_frame + clip_frames

            if self._overlaps_with_any(start_frame, end_frame, excluded_clips):
                continue

            segment_score = scores[start_frame:end_frame].mean()
            
            if segment_score > best_score:
                best_score = segment_score
                best_start = start_frame
        
        if best_score < 0:
            return None
        
        return Clip(
            start_frame=best_start,
            end_frame=best_start + clip_frames,
            fps=self.fps,
            score=best_score,
            song_start_time=song_start_time
        )
    
    def _overlaps_with_any(
        self, 
        start_frame: int, 
        end_frame: int, 
        clips: List[Clip]
    ) -> bool:

        for clip in clips:
            if not (end_frame < clip.start_frame or start_frame > clip.end_frame):
                return True
        return False