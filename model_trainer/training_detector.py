import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional
import numba


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


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_motion_score(mag):
    return 0.6 * np.percentile(mag, 90) + 0.4 * mag.mean()


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_overlap_penalty(start, end, clip_starts, clip_ends, clip_frames):
    penalty = 1.0
    for i in range(len(clip_starts)):
        overlap = max(0, min(end, clip_ends[i]) - max(start, clip_starts[i]))
        if overlap > 0:
            penalty *= np.exp(-overlap / (0.3 * clip_frames))
    return penalty


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_heuristic(segment):
    peak = segment.max()
    mean = segment.mean()
    contrast = peak - mean
    peak_pos = np.argmax(segment) / len(segment)
    return 0.5 * peak + 0.3 * contrast + 0.2 * (1 - abs(peak_pos - 0.5))


class VisualFeatureExtractor:
    """OPTIMIZED: Extract visual features with minimal overhead"""
    
    def __init__(self, enable_face_detection=True):
        self.use_face_detection = False
        if enable_face_detection:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_face_detection = True
            except:
                pass
    
    def extract(self, frame) -> dict:
        """extraction"""

        h, w = frame.shape[:2]
        if h > 360:
            scale = 360 / h
            frame = cv2.resize(frame, (int(w * scale), 360))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 / (1.0 + laplacian_var / 100.0)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_variance = float(hsv[:, :, 1].std() / 255.0)

        avg_brightness = float(gray.mean() / 255.0)
        contrast_mean = float(gray.std() / 255.0)

        face_present = 0.0
        face_size_ratio = 0.0
        if self.use_face_detection:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                face_present = 1.0
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                face_area = largest_face[2] * largest_face[3]
                frame_area = frame.shape[0] * frame.shape[1]
                face_size_ratio = float(face_area / frame_area)

        symmetry = self._compute_symmetry_fast(gray)

        
        return {
            "blur_score": blur_score,
            "edge_density": edge_density,
            "color_variance": color_variance,
            "avg_brightness": avg_brightness,
            "contrast_mean": contrast_mean,
            "face_present": face_present,
            "face_size_ratio": face_size_ratio,
            "symmetry": symmetry,
            "rule_of_thirds": 0.0,  # Skipped
            "text_presence": 0.0,    # Skipped
        }
    
    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _compute_symmetry_numba(left, right):
        """Numba-optimized symmetry computation"""
        diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean()
        return max(0.0, 1.0 - diff / 255.0)
    
    def _compute_symmetry_fast(self, gray):
        """FAST symmetry - downsampled"""
        h, w = gray.shape
        if w > 100:
            gray = cv2.resize(gray, (100, int(100 * h / w)))
            h, w = gray.shape
        
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        return float(self._compute_symmetry_numba(left, right))


class TrainingDetector:
    MOTION_WEIGHT = 0.40
    AUDIO_WEIGHT = 0.35
    VISUAL_WEIGHT = 0.10
    CONTEXT_WEIGHT = 0.15
    SMOOTH_SIGMA = 1.5

    def __init__(self, loader, audio_analyzer, model=None, enable_face_detection=False, audio_loops: int = 1):
        self.loader = loader
        self.audio = audio_analyzer
        self.audio_loops = max(1, audio_loops)
        self.cap = loader.cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frames_count / self.fps
        self.model = model

        self.visual_extractor = VisualFeatureExtractor(enable_face_detection)
        
        self.use_ml = model is not None
        self.heuristic_weight = 0.0 if self.use_ml else 1.0

        self._base_beat_intervals = self.audio.get_beat_intervals() or []
        self._base_audio_duration = (
            max(b[0] + b[1] for b in self._base_beat_intervals)
            if self._base_beat_intervals else 0.0
        )

    def detect_perfect_clips(self, max_clips=None) -> List[Clip]:
        print("\n[TD]strt...")

        all_features = self._compute_all_features_optimized()
        
        beat_intervals = self._loop_beat_intervals()

        if not beat_intervals:
            print("[TD]not found")
            return []

        print(f" [TD]found inter: {len(beat_intervals)}")
        
        clips = self._find_clips_for_all_beats(all_features, beat_intervals, max_clips)
        
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

    def _compute_all_features_optimized(self):
        """Faster feature computation"""
        print(" [TD]computing features...")
        
        motion_scores = np.zeros(self.frames_count, dtype=np.float32)
        visual_features = []
        
        prev_gray = None

        FLOW_STEP = max(1, int(self.fps / 10))
        VISUAL_STEP = max(1, int(self.fps / 5))

        for i, frame in self.loader.frames():
            if i >= self.frames_count:
                break
            
            if i % 1000 == 0:
                print(f" [TD]prog: {i}/{self.frames_count} frames", end='\r')

            if i % FLOW_STEP == 0:

                gray = cv2.cvtColor(cv2.resize(frame, (256, 144)), cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 
                        0.5, 2, 10, 2, 5, 1.1, 0
                    )
                    mag = np.linalg.norm(flow, axis=2)
                    motion_scores[i] = compute_motion_score(mag)
                
                prev_gray = gray

            if i % VISUAL_STEP == 0:
                vis_feat = self.visual_extractor.extract(frame)
                visual_features.append((i, vis_feat))

        print()

        non_zero = motion_scores > 0
        if non_zero.any():
            motion_scores = np.interp(
                np.arange(self.frames_count),
                np.where(non_zero)[0],
                motion_scores[non_zero]
            )
        
        motion_scores = gaussian_filter1d(motion_scores, sigma=self.SMOOTH_SIGMA)

        all_features = []
        for idx in range(self.frames_count):
            
            audio_feat = self.audio.get_advanced_audio_features(idx, self.frames_count)
            motion_feat = self._extract_motion_features_fast(idx, motion_scores)
            visual_feat = self._get_closest_visual_features(idx, visual_features)
            context_feat = self._extract_temporal_context_fast(idx, motion_scores, audio_feat)
            
            combined = {**motion_feat, **audio_feat, **visual_feat, **context_feat}
            all_features.append(combined)

        return all_features
    
    def _get_closest_visual_features(self, frame_idx, visual_features):
        if not visual_features:
            return self._empty_visual_features()

        closest = min(visual_features, key=lambda x: abs(x[0] - frame_idx))
        return closest[1]
    
    @staticmethod
    def _empty_visual_features():
        return {
            "blur_score": 0.5,
            "edge_density": 0.0,
            "color_variance": 0.0,
            "avg_brightness": 0.5,
            "contrast_mean": 0.0,
            "face_present": 0.0,
            "face_size_ratio": 0.0,
            "symmetry": 0.5,
            "rule_of_thirds": 0.0,
            "text_presence": 0.0,
        }
    
    def _extract_motion_features_fast(self, idx, motion_scores):
        """FAST motion feature extraction with numpy slicing"""
        window_size = int(self.fps * 2)
        start = max(0, idx - window_size)
        end = min(len(motion_scores), idx + window_size)
        
        segment = motion_scores[start:end]
        
        if len(segment) == 0:
            segment = np.array([0.0])

        mean_val = segment.mean()
        max_val = segment.max()
        
        return {
            "motion_mean": float(mean_val),
            "motion_max": float(max_val),
            "motion_p90": float(np.percentile(segment, 90)),
            "motion_std": float(segment.std()),
            "motion_peak_ratio": float(max_val / (mean_val + 1e-6)),
        }
    
    def _extract_temporal_context_fast(self, idx, motion_scores, audio_feat):
        """temporal context"""
        window = 5
        start = max(0, idx - window)
        end = min(len(motion_scores), idx + window)
        
        motion_window = motion_scores[start:end]
        
        if len(motion_window) > 1:

            motion_momentum = float(motion_window[-1] - motion_window[0]) / len(motion_window)
        else:
            motion_momentum = 0.0
        
        audio_momentum = audio_feat.get("rms_contrast", 0.0)
        combined_buildup = (motion_momentum + audio_momentum) * 0.5
        relative_position = float(idx / self.frames_count)
        
        if idx > 0 and idx < len(motion_scores) - 1:
            motion_derivative = float(motion_scores[idx] - motion_scores[idx-1])
        else:
            motion_derivative = 0.0
        
        return {
            "motion_momentum": motion_momentum,
            "audio_momentum": audio_momentum,
            "combined_buildup": combined_buildup,
            "relative_position": relative_position,
            "motion_derivative": max(0, motion_derivative),
        }

    def _find_clips_for_all_beats(self, all_features, beat_intervals, max_clips=None) -> List[Clip]:
        clips = []
        total_intervals = min(len(beat_intervals), max_clips) if max_clips else len(beat_intervals)

        print(f"\n   [TD]f clips {total_intervals} beats...")

        for i in range(total_intervals):
            song_start, target_duration = beat_intervals[i]
            best_clip = self._find_best_segment(all_features, target_duration, song_start, clips)

            if best_clip:
                clips.append(best_clip)
                print(f"      [TD]beat {i + 1}: clip {best_clip.start:.2f}s-{best_clip.end:.2f}s "
                      f"(len: {best_clip.duration:.2f}s, score: {best_clip.score:.3f})")

        return clips

    def _find_best_segment(self, all_features, target_duration, song_start_time, excluded_clips) -> Optional[Clip]:
        clip_frames = int(target_duration * self.fps)
        if clip_frames >= self.frames_count:
            return None

        best_score = -1.0
        best_start = 0
        best_features = None

        step = max(1, int(self.fps * 0.3))

        clip_starts = np.array([c.start_frame for c in excluded_clips], dtype=np.int32)
        clip_ends = np.array([c.end_frame for c in excluded_clips], dtype=np.int32)

        for start_frame in range(0, self.frames_count - clip_frames, step):
            end_frame = start_frame + clip_frames

            segment_features = self._aggregate_segment_features_fast(
                all_features, start_frame, end_frame
            )

            start_time_sec = start_frame / self.fps
            segment_features["beat_alignment_score"] = self.audio.get_beat_alignment_score(
                start_time_sec
            )

            if self.use_ml:
                score = self._ml_score(segment_features)
            else:
                score = self._heuristic_score(segment_features)

            overlap_penalty = compute_overlap_penalty(
                start_frame, end_frame, clip_starts, clip_ends, clip_frames
            )
            score *= overlap_penalty

            if score > best_score:
                best_score = score
                best_start = start_frame
                best_features = segment_features

        if best_score < 0:
            return None

        return Clip(
            best_start,
            best_start + clip_frames,
            self.fps,
            best_score,
            song_start_time,
            best_features
        )
    
    def _aggregate_segment_features_fast(self, all_features, start, end):
        """aggregation using numpy"""
        segment = all_features[start:end]
        
        if not segment:
            return all_features[start] if start < len(all_features) else {}

        aggregated = {}
        keys = segment[0].keys()
        
        for key in keys:
            values = np.array([f[key] for f in segment], dtype=np.float32)
            aggregated[key] = float(values.mean())
        
        aggregated["duration"] = (end - start) / self.fps
        
        return aggregated
    
    def _ml_score(self, features):
        try:
            feature_vector = self._features_to_vector(features)
            ml_score = self.model.predict_proba([feature_vector])[0][1]
            
            if self.heuristic_weight > 0:
                heur_score = self._heuristic_score(features)
                return self.heuristic_weight * heur_score + (1 - self.heuristic_weight) * ml_score
            
            return float(ml_score)
        except:
            return self._heuristic_score(features)
    
    def _heuristic_score(self, features):
        """FAST heuristic - minimal operations"""
        score = (
            0.25 * features.get("motion_p90", 0) +
            0.15 * features.get("motion_peak_ratio", 0) * 0.1 +
            0.15 * features.get("beat_alignment_score", 0) +
            0.15 * features.get("bass_energy", 0) +
            0.10 * (1 - features.get("blur_score", 0.5)) +
            0.10 * features.get("combined_buildup", 0) +
            0.10 * features.get("rms_peak", 0)
        )
        
        return float(max(0, min(1, score)))
    
    def _features_to_vector(self, features):
        feature_order = [
            "motion_mean", "motion_max", "motion_p90", "motion_std", "motion_peak_ratio",
            "rms_mean", "rms_peak", "rms_contrast",
            "spectral_centroid", "spectral_rolloff", "mfcc_variance",
            "bass_energy", "vocal_probability", "onset_density",
            "beat_alignment_score",
            "blur_score", "edge_density", "color_variance",
            "avg_brightness", "contrast_mean",
            "face_present", "face_size_ratio",
            "symmetry", "rule_of_thirds", "text_presence",
            "motion_momentum", "audio_momentum", "combined_buildup",
            "relative_position", "motion_derivative",
            "duration"
        ]
        
        return [features.get(key, 0.0) for key in feature_order]
    
    def _compute_diversity_penalty(self, features, excluded_clips):
        """OPTIONAL: Skip this for speed if quality is already good"""
        if not excluded_clips:
            return 1.0
        
        feature_vector = np.array(self._features_to_vector(features))
        
        penalty = 1.0
        for clip in excluded_clips:
            if clip.features is None:
                continue
            
            clip_vector = np.array(self._features_to_vector(clip.features))
            
            norm_prod = np.linalg.norm(feature_vector) * np.linalg.norm(clip_vector)
            if norm_prod > 0:
                similarity = np.dot(feature_vector, clip_vector) / norm_prod
                penalty *= (1 - 0.3 * max(0, similarity))
        
        return float(penalty)