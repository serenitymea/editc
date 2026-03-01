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
def compute_overlap_penalty(start, end, clip_starts, clip_ends, clip_frames, decay_ratio):
    penalty = 1.0
    for i in range(len(clip_starts)):
        overlap = max(0, min(end, clip_ends[i]) - max(start, clip_starts[i]))
        if overlap > 0:
            penalty *= np.exp(-overlap / (decay_ratio * clip_frames))
    return penalty


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_heuristic(segment):
    peak = segment.max()
    mean = segment.mean()
    contrast = peak - mean
    peak_pos = np.argmax(segment) / len(segment)
    return 0.5 * peak + 0.3 * contrast + 0.2 * (1 - abs(peak_pos - 0.5))


class VisualFeatureExtractor:

    def __init__(self, config):

        self.config = config
        self.use_face_detection = False
        
        if self.config["face_detection"]["enabled"]:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_face_detection = True
            except:
                pass
    
    def extract(self, frame) -> dict:

        h, w = frame.shape[:2]
        resize_height = self.config["video_processing"]["resize_height"]
        
        if h > resize_height:
            scale = resize_height / h
            frame = cv2.resize(frame, (int(w * scale), resize_height))
        
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
            scale_factor = self.config["face_detection"]["scale_factor"]
            min_neighbors = self.config["face_detection"]["min_neighbors"]
            
            faces = self.face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
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

        diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean()
        return max(0.0, 1.0 - diff / 255.0)
    
    def _compute_symmetry_fast(self, gray):

        h, w = gray.shape
        symmetry_max_width = self.config["video_processing"]["symmetry_max_width"]
        
        if w > symmetry_max_width:
            gray = cv2.resize(gray, (symmetry_max_width, int(symmetry_max_width * h / w)))
            h, w = gray.shape
        
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        return float(self._compute_symmetry_numba(left, right))


class TrainingDetector:
    SMOOTH_SIGMA = 1.5

    def __init__(self, config, loader, audio_analyzer, model=None, audio_loops: Optional[int] = None):

        self.config = config
        self.loader = loader
        self.audio = audio_analyzer
        
        if audio_loops is None:
            audio_loops = self.config["audio_loops"]["default"]
        self.audio_loops = max(1, audio_loops)
        
        self.cap = loader.cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frames_count / self.fps
        self.model = model

        self.visual_extractor = VisualFeatureExtractor(config)
        
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

        print(" [TD]computing features...")
        
        motion_scores = np.zeros(self.frames_count, dtype=np.float32)
        visual_features = []
        
        prev_gray = None

        flow_step = max(1, int(self.fps / self.config["windows"]["flow_sample_fps_divider"]))
        visual_step = max(1, int(self.fps / self.config["windows"]["visual_sample_fps_divider"]))

        for i, frame in self.loader.frames():
            if i >= self.frames_count:
                break
            
            if i % 1000 == 0:
                print(f" [TD]prog: {i}/{self.frames_count} frames", end='\r')

            if i % flow_step == 0:
                flow_res = self.config["video_processing"]["flow_resolution"]
                gray = cv2.cvtColor(cv2.resize(frame, tuple(flow_res)), cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    opt_flow_params = self.config["optical_flow"]
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 
                        opt_flow_params["pyr_scale"],
                        opt_flow_params["levels"],
                        opt_flow_params["winsize"],
                        opt_flow_params["iterations"],
                        opt_flow_params["poly_n"],
                        opt_flow_params["poly_sigma"],
                        opt_flow_params["flags"]
                    )
                    mag = np.linalg.norm(flow, axis=2)
                    motion_scores[i] = compute_motion_score(mag)
                
                prev_gray = gray

            if i % visual_step == 0:
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
        
        motion_scores = gaussian_filter1d(motion_scores, self.SMOOTH_SIGMA)
        
        if motion_scores.max() > 0:
            motion_scores /= motion_scores.max()

        print(" [TD]building feature vectors...")
        all_features = self._build_feature_vectors_fast(
            motion_scores, visual_features
        )
        
        return all_features
    
    def _build_feature_vectors_fast(self, motion_scores, visual_features):

        all_features = []

        audio_features_cache = {}
        
        for frame_idx in range(self.frames_count):

            vis_feat = self._get_visual_features_at_frame(frame_idx, visual_features)

            if frame_idx not in audio_features_cache:
                audio_feat = self.audio.get_advanced_audio_features(frame_idx, self.frames_count)
                audio_features_cache[frame_idx] = audio_feat
            else:
                audio_feat = audio_features_cache[frame_idx]

            motion_feat = self._extract_motion_features_fast(frame_idx, motion_scores)

            context_feat = self._extract_temporal_context_fast(frame_idx, motion_scores, audio_feat)
            
            combined = {
                **motion_feat,
                **audio_feat,
                **vis_feat,
                **context_feat,
            }
            
            all_features.append(combined)
        
        return all_features
    
    def _get_visual_features_at_frame(self, frame_idx, visual_features):

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

        window_size = int(self.fps * self.config["windows"]["motion_window_seconds"])
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

        window = self.config["windows"]["temporal_window_frames"]
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

        step = max(1, int(self.fps * self.config["windows"]["clip_search_step_seconds"]))

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

            decay_ratio = self.config["clip_selection"]["overlap_decay_ratio"]
            overlap_penalty = compute_overlap_penalty(
                start_frame, end_frame, clip_starts, clip_ends, clip_frames, decay_ratio
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

        weights = self.config["scoring"]["heuristic_weights"]
        
        score = (
            weights["motion_p90"] * features.get("motion_p90", 0) +
            weights["motion_peak_ratio"] * features.get("motion_peak_ratio", 0) * 0.1 +
            weights["beat_alignment"] * features.get("beat_alignment_score", 0) +
            weights["bass_energy"] * features.get("bass_energy", 0) +
            weights["blur_inverse"] * (1 - features.get("blur_score", 0.5)) +
            weights["buildup"] * features.get("combined_buildup", 0) +
            weights["rms_peak"] * features.get("rms_peak", 0)
        )
        
        return float(max(0, min(1, score)))
    
    def _features_to_vector(self, features):
        feature_order = self.config["ml"]["feature_order"]
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