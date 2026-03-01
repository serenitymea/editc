from __future__ import annotations

import bisect
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import numba
from scipy.ndimage import gaussian_filter1d

from .gpu_kernels import (
    get_backend,
    gpu_aggregate_segments,
    gpu_diversity_penalties,
    gpu_batch_motion_features,
    gpu_batch_temporal_context,
    gpu_batch_heuristic_scores,
    gpu_batch_overlap_penalties,
)
from .pipeline import ParallelFeaturePipeline, optimal_num_workers

@dataclass(frozen=True)
class Clip:
    start_frame: int
    end_frame:   int
    fps:         float
    score:       float
    song_start_time: float
    features:    Optional[dict] = None

    @property
    def start(self) -> float:    return self.start_frame / self.fps
    @property
    def end(self) -> float:      return self.end_frame   / self.fps
    @property
    def duration(self) -> float: return self.end - self.start


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_motion_score(mag: np.ndarray) -> float:
    flat = mag.ravel()
    p90  = np.sort(flat)[int(0.9 * len(flat))]
    return 0.6 * p90 + 0.4 * flat.mean()


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_overlap_penalty(
    start: int, end: int,
    clip_starts: np.ndarray, clip_ends: np.ndarray,
    clip_frames: int, decay_ratio: float,
) -> float:
    penalty   = 1.0
    inv_decay = 1.0 / (decay_ratio * clip_frames)
    for i in range(len(clip_starts)):
        a = clip_ends[i]   if end   > clip_ends[i]   else end
        b = clip_starts[i] if start < clip_starts[i] else start
        overlap = a - b
        if overlap > 0:
            penalty *= np.exp(-overlap * inv_decay)
    return penalty


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_heuristic(segment: np.ndarray) -> float:
    peak     = segment.max()
    mean     = segment.mean()
    contrast = peak - mean
    peak_pos = np.argmax(segment) / len(segment)
    return 0.5 * peak + 0.3 * contrast + 0.2 * (1.0 - abs(peak_pos - 0.5))

class VisualFeatureExtractor:
    """Extract visual features with minimal overhead."""

    def __init__(self, config):
        self.config = config
        self.use_face_detection = False
        if self.config["face_detection"]["enabled"]:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self.use_face_detection = True
            except Exception:
                pass

    def extract(self, frame) -> dict:
        h, w = frame.shape[:2]
        rh   = self.config["video_processing"]["resize_height"]
        if h > rh:
            frame = cv2.resize(frame, (int(w * rh / h), rh))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lv          = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score  = 1.0 / (1.0 + lv / 100.0)
        edges       = cv2.Canny(gray, 50, 150, apertureSize=3)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        hsv          = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_var    = float(hsv[:, :, 1].std() / 255.0)
        brightness   = float(gray.mean() / 255.0)
        contrast     = float(gray.std()  / 255.0)

        face_present    = 0.0
        face_size_ratio = 0.0
        if self.use_face_detection:
            faces = self.face_cascade.detectMultiScale(
                gray,
                self.config["face_detection"]["scale_factor"],
                self.config["face_detection"]["min_neighbors"],
            )
            if len(faces) > 0:
                face_present    = 1.0
                lf              = max(faces, key=lambda f: f[2] * f[3])
                face_size_ratio = float(lf[2] * lf[3] / (frame.shape[0] * frame.shape[1]))

        symmetry = self._compute_symmetry_fast(gray)
        return {
            "blur_score": blur_score, "edge_density": edge_density,
            "color_variance": color_var, "avg_brightness": brightness,
            "contrast_mean": contrast, "face_present": face_present,
            "face_size_ratio": face_size_ratio, "symmetry": symmetry,
            "rule_of_thirds": 0.0, "text_presence": 0.0,
        }

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _compute_symmetry_numba(left, right):
        diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean()
        return max(0.0, 1.0 - diff / 255.0)

    def _compute_symmetry_fast(self, gray):
        h, w = gray.shape
        smw  = self.config["video_processing"]["symmetry_max_width"]
        if w > smw:
            gray = cv2.resize(gray, (smw, int(smw * h / w)))
            h, w = gray.shape
        left  = gray[:, : w // 2]
        right = cv2.flip(gray[:, w // 2 :], 1)
        mw    = min(left.shape[1], right.shape[1])
        return float(self._compute_symmetry_numba(left[:, :mw], right[:, :mw]))

def _sliding_std(signal: np.ndarray, win: int) -> np.ndarray:
    """Vectorized sliding window std via stride_tricks. No Python loop."""
    n      = len(signal)
    padded = np.pad(signal, win, mode="reflect")
    kernel = 2 * win + 1
    shape   = (n, kernel)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).copy()
    return windows.std(axis=1).astype(np.float32)

class EpicDetector:
    SMOOTH_SIGMA = 1.5

    _FEATURE_KEYS = [
        "motion_mean", "motion_max", "motion_p90", "motion_std", "motion_peak_ratio",
        "rms", "rms_peak", "rms_contrast", "bass_energy", "beat_alignment_score",
        "blur_score", "edge_density", "color_variance", "avg_brightness",
        "contrast_mean", "face_present", "face_size_ratio", "symmetry",
        "rule_of_thirds", "text_presence",
        "motion_momentum", "audio_momentum", "combined_buildup",
        "relative_position", "motion_derivative", "duration",
        "motion_temporal_std",
        "motion_p95_mean_ratio", 
        "audio_rms_std",    
        "edge_contrast_grad", 
    ]
    _KEY_INDEX: dict = {k: i for i, k in enumerate(_FEATURE_KEYS)}

    _AUDIO_KEYS = ["rms", "rms_peak", "rms_contrast", "bass_energy"]

    def __init__(self, config, loader, audio_analyzer, model=None):
        self.config           = config
        self.loader           = loader
        self.audio            = audio_analyzer
        self.cap              = loader.cap
        self.fps              = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames_count     = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration         = self.frames_count / self.fps
        self.model            = model
        self.visual_extractor = VisualFeatureExtractor(config)
        self.use_ml           = model is not None
        self.heuristic_weight = 0.0 if self.use_ml else 1.0
        self._clips_lock      = threading.Lock()
        self._heuristic_vec: Optional[np.ndarray] = None

        self._num_workers = optimal_num_workers()
        print(f"[EpicDetector] GPU backend: {get_backend()} | "
              f"CPU workers: {self._num_workers} / {os.cpu_count()}")

    def detect_perfect_clips(self, max_clips=None) -> List[Clip]:
        print("\n[ED]strt...")
        all_features   = self._compute_all_features_optimized()
        beat_intervals = self.audio.get_beat_intervals()
        if not beat_intervals:
            print("[ED]not found")
            return []
        print(f" [ED]found inter: {len(beat_intervals)}")
        return self._find_clips_for_all_beats(all_features, beat_intervals, max_clips)


    def _compute_all_features_optimized(self) -> np.ndarray:

        print(" [ED]running parallel pipeline...")

        pipeline = ParallelFeaturePipeline(
            config=self.config,
            loader=self.loader,
            audio_analyzer=self.audio,
            visual_extractor=self.visual_extractor,
            frames_count=self.frames_count,
            fps=self.fps,
            num_workers=self._num_workers,
            queue_size=64,
        )

        motion_scores, visual_features, audio_matrix = pipeline.run()

        print(" [ED]building feature matrix (GPU)...")
        return self._build_feature_matrix(motion_scores, visual_features, audio_matrix)

    def _build_feature_matrix(
        self,
        motion_scores:   np.ndarray,
        visual_features: list, 
        audio_matrix:    np.ndarray,
    ) -> np.ndarray:

        n  = self.frames_count
        k  = len(self._FEATURE_KEYS)
        ki = self._KEY_INDEX

        matrix = np.zeros((n, k), dtype=np.float32)

        win = int(self.fps * self.config["windows"]["motion_window_seconds"])
        m_mean, m_max, m_p90, m_std, m_ratio = gpu_batch_motion_features(motion_scores, win)
        matrix[:, ki["motion_mean"]]       = m_mean
        matrix[:, ki["motion_max"]]        = m_max
        matrix[:, ki["motion_p90"]]        = m_p90
        matrix[:, ki["motion_std"]]        = m_std
        matrix[:, ki["motion_peak_ratio"]] = m_ratio

        for ci, ak in enumerate(self._AUDIO_KEYS):
            if ak in ki:
                matrix[:, ki[ak]] = audio_matrix[:, ci]
        matrix[:, ki["beat_alignment_score"]] = audio_matrix[:, len(self._AUDIO_KEYS)]

        tw = self.config["windows"]["temporal_window_frames"]
        rms_contrast = audio_matrix[:, self._AUDIO_KEYS.index("rms_contrast")]
        t_mom, t_buildup, t_deriv = gpu_batch_temporal_context(motion_scores, tw, rms_contrast)
        matrix[:, ki["motion_momentum"]]   = t_mom
        matrix[:, ki["audio_momentum"]]    = rms_contrast
        matrix[:, ki["combined_buildup"]]  = t_buildup
        matrix[:, ki["relative_position"]] = np.arange(n, dtype=np.float32) / n
        matrix[:, ki["motion_derivative"]] = np.maximum(0.0, t_deriv)

        vis_keys = [
            "blur_score", "edge_density", "color_variance", "avg_brightness",
            "contrast_mean", "face_present", "face_size_ratio", "symmetry",
            "rule_of_thirds", "text_presence",
        ]
        if visual_features:
            vis_idx_arr = np.array([x[0] for x in visual_features], dtype=np.int32)
            vis_matrix  = np.array(
                [[vf.get(vk, 0.0) for vk in vis_keys] for _, vf in visual_features],
                dtype=np.float32,
            )
            all_frames = np.arange(n, dtype=np.int32)
            pos        = np.searchsorted(vis_idx_arr, all_frames)
            pos        = np.clip(pos, 0, len(vis_idx_arr) - 1)
            pos_prev   = np.maximum(pos - 1, 0)
            nearest    = np.where(
                np.abs(vis_idx_arr[pos_prev] - all_frames) < np.abs(vis_idx_arr[pos] - all_frames),
                pos_prev, pos,
            )
            for ci, vk in enumerate(vis_keys):
                if vk in ki:
                    matrix[:, ki[vk]] = vis_matrix[nearest, ci]

        if "motion_temporal_std" in ki:
            matrix[:, ki["motion_temporal_std"]] = m_std

        if "motion_p95_mean_ratio" in ki:
            matrix[:, ki["motion_p95_mean_ratio"]] = np.clip(
                m_p90 / (m_mean + 1e-6), 0.0, 10.0
            )

        if "audio_rms_std" in ki:
            rms_signal = matrix[:, ki["rms"]].copy()
            matrix[:, ki["audio_rms_std"]] = _sliding_std(rms_signal, win)

        if "edge_contrast_grad" in ki and "edge_density" in ki:
            edge_col  = matrix[:, ki["edge_density"]].copy()
            edge_grad = np.abs(np.diff(edge_col, prepend=edge_col[0])).astype(np.float32)
            edge_grad = gaussian_filter1d(edge_grad, sigma=3.0).astype(np.float32)
            matrix[:, ki["edge_contrast_grad"]] = edge_grad

        return matrix

    def _find_clips_for_all_beats(
        self,
        all_features: np.ndarray,
        beat_intervals,
        max_clips=None,
    ) -> List[Clip]:
        clips: List[Clip] = []
        total = min(len(beat_intervals), max_clips) if max_clips else len(beat_intervals)
        print(f"\n   [ED]f clips {total} beats...")

        clip_starts = np.empty(total, dtype=np.int32)
        clip_ends   = np.empty(total, dtype=np.int32)
        n_clips     = 0

        for i in range(total):
            song_start, target_duration = beat_intervals[i]
            best_clip = self._find_best_segment(
                all_features, target_duration, song_start,
                clip_starts[:n_clips], clip_ends[:n_clips],
            )
            if best_clip:
                clip_starts[n_clips] = best_clip.start_frame
                clip_ends[n_clips]   = best_clip.end_frame
                n_clips += 1
                clips.append(best_clip)
                print(
                    f"      [ED]beat {i + 1}: clip {best_clip.start:.2f}s-{best_clip.end:.2f}s "
                    f"(len: {best_clip.duration:.2f}s, score: {best_clip.score:.3f})"
                )

        return clips

    def _find_best_segment(
        self,
        all_features: np.ndarray,
        target_duration: float,
        song_start_time: float,
        clip_starts: np.ndarray,
        clip_ends:   np.ndarray,
    ) -> Optional[Clip]:

        clip_frames = int(target_duration * self.fps)
        if clip_frames >= self.frames_count:
            return None

        step        = max(1, int(self.fps * self.config["windows"]["clip_search_step_seconds"]))
        cand_starts = np.arange(0, self.frames_count - clip_frames, step, dtype=np.int32)
        cand_ends   = cand_starts + clip_frames

        if len(cand_starts) == 0:
            return None

        agg = gpu_aggregate_segments(all_features, cand_starts, cand_ends)

        dur_col = self._KEY_INDEX.get("duration", -1)
        if dur_col >= 0:
            agg[:, dur_col] = np.float32(target_duration)

        if self.use_ml:
            scores = np.array([
                self._ml_score(self._row_to_dict(agg[i]))
                for i in range(len(cand_starts))
            ], dtype=np.float32)
        else:
            scores = self._batch_heuristic_scores_gpu(agg)

        if len(clip_starts) > 0:
            decay_ratio = self.config["clip_selection"]["overlap_decay_ratio"]
            penalties   = gpu_batch_overlap_penalties(
                cand_starts, cand_ends,
                clip_starts, clip_ends,
                clip_frames, decay_ratio,
            )
            scores = scores * penalties

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < 0:
            return None

        best_start    = int(cand_starts[best_idx])
        best_features = self._row_to_dict(agg[best_idx])
        best_features["duration"] = target_duration

        return Clip(best_start, best_start + clip_frames, self.fps,
                    best_score, song_start_time, best_features)

    def _get_heuristic_vec(self) -> np.ndarray:

        if self._heuristic_vec is None:
            w = self.config["scoring"]["heuristic_weights"]
            self._heuristic_vec = np.array([
                w.get("motion_p90",        0.28),  # 0
                w.get("motion_peak_ratio", 0.08),  # 1
                w.get("beat_alignment",    0.12),  # 2
                w.get("bass_energy",       0.08),  # 3
                w.get("blur_inverse",      0.06),  # 4
                w.get("buildup",           0.10),  # 5
                w.get("rms_peak",          0.10),  # 6
                w.get("edge_density",      0.06),  # 7
                w.get("color_variance",    0.04),  # 8
                0.0,                               # 9  flat-penalty placeholder
                w.get("p95_mean_ratio",    0.10),  # 10
                0.0,                               # 11 sat-penalty placeholder
                w.get("edge_grad",         0.06),  # 12
                w.get("contrast_mean",     0.06),  # 13
            ], dtype=np.float32)
        return self._heuristic_vec

    def _batch_heuristic_scores_gpu(self, candidates: np.ndarray) -> np.ndarray:
        return gpu_batch_heuristic_scores(
            candidates,
            self._get_heuristic_vec(),
            self._KEY_INDEX,
            self._KEY_INDEX.get("blur_score", -1),
        )

    def _compute_diversity_penalty(self, features: dict, excluded_clips: List[Clip]) -> float:
        if not excluded_clips:
            return 1.0
        fv   = np.array(self._features_to_vector(features), dtype=np.float32)
        if np.linalg.norm(fv) == 0:
            return 1.0
        valid = [c for c in excluded_clips if c.features is not None]
        if not valid:
            return 1.0
        cm      = np.array([self._features_to_vector(c.features) for c in valid], dtype=np.float32)
        factors = gpu_diversity_penalties(fv, cm)
        return float(np.prod(factors))


    def _aggregate_segment_features_fast(
        self, all_features: np.ndarray, start: int, end: int
    ) -> dict:
        agg = gpu_aggregate_segments(
            all_features,
            np.array([start], dtype=np.int32),
            np.array([end],   dtype=np.int32),
        )
        result = self._row_to_dict(agg[0])
        result["duration"] = (end - start) / self.fps
        return result


    def _heuristic_score(self, features: dict) -> float:
        """
        Nonlinear peak-amplifying score (scalar version, mirrors GPU kernel).

        Formula:
          base  = dot(weights, feats)
          peak  = (p90 * peak_ratio)^0.55
          flat  = exp(-m_std*6) * exp(-rms_std*4)
          sat   = (rms_peak>0.6) * exp(-rms_std*3)
          score = tanh(base*2.2) * (1-0.55*flat) * (1-0.30*sat) * (1+0.40*peak)
          score = clip(score, 0, 1)^1.3
        """
        wv = self._get_heuristic_vec()

        m_p90     = features.get("motion_p90", 0.0)
        m_ratio   = features.get("motion_peak_ratio", 0.0)
        beat      = features.get("beat_alignment_score", 0.0)
        bass      = features.get("bass_energy", 0.0)
        blur_inv  = 1.0 - features.get("blur_score", 0.5)
        buildup   = features.get("combined_buildup", 0.0)
        rms_pk    = features.get("rms_peak", 0.0)
        edge      = features.get("edge_density", 0.0)
        col_var   = features.get("color_variance", 0.0)
        m_std     = features.get("motion_temporal_std", features.get("motion_std", 0.0))
        p95_ratio = features.get("motion_p95_mean_ratio", m_ratio)
        rms_std   = features.get("audio_rms_std", 0.0)
        edge_grad = features.get("edge_contrast_grad", 0.0)
        contrast  = features.get("contrast_mean", 0.0)

        feats = np.array([
            m_p90,
            m_ratio  * 0.12,
            beat,
            bass,
            blur_inv,
            buildup,
            rms_pk,
            edge,
            col_var,
            0.0,
            p95_ratio * 0.12,
            0.0,
            edge_grad,
            contrast,
        ], dtype=np.float32)

        base = float(np.dot(wv, feats))

        peak = float(max(0.0, m_p90 * max(m_ratio, 0.0)) ** 0.55)

        flat = float(np.exp(-m_std * 6.0) * np.exp(-rms_std * 4.0))

        sat = float(np.exp(-rms_std * 3.0)) if rms_pk > 0.6 else 0.0

        score = float(np.tanh(base * 2.2))
        score *= (1.0 - 0.55 * flat)
        score *= (1.0 - 0.30 * sat)
        score *= (1.0 + 0.40 * peak)
        score  = float(np.clip(score, 0.0, 1.0)) ** 1.3

        return float(np.clip(score, 0.0, 1.0))

    def _ml_score(self, features: dict) -> float:
        try:
            fv       = self._features_to_vector(features)
            ml_score = self.model.predict_proba([fv])[0][1]
            if self.heuristic_weight > 0:
                heur = self._heuristic_score(features)
                return self.heuristic_weight * heur + (1 - self.heuristic_weight) * ml_score
            return float(ml_score)
        except Exception:
            return self._heuristic_score(features)

    def _features_to_vector(self, features: dict) -> list:
        return [features.get(k, 0.0) for k in self.config["ml"]["feature_order"]]


    def _row_to_dict(self, row: np.ndarray) -> dict:
        return {k: float(row[i]) for i, k in enumerate(self._FEATURE_KEYS)}


    def _get_visual_features_at_frame(self, frame_idx: int, visual_features: list) -> dict:
        if not visual_features:
            return self._empty_visual_features()
        indices = [x[0] for x in visual_features]
        pos     = bisect.bisect_left(indices, frame_idx)
        if pos == 0:                    return visual_features[0][1]
        if pos >= len(visual_features): return visual_features[-1][1]
        b, a = visual_features[pos - 1], visual_features[pos]
        return a[1] if abs(a[0] - frame_idx) < abs(b[0] - frame_idx) else b[1]

    @staticmethod
    def _empty_visual_features() -> dict:
        return {
            "blur_score": 0.5, "edge_density": 0.0, "color_variance": 0.0,
            "avg_brightness": 0.5, "contrast_mean": 0.0, "face_present": 0.0,
            "face_size_ratio": 0.0, "symmetry": 0.5, "rule_of_thirds": 0.0,
            "text_presence": 0.0,
        }

    def _extract_motion_features_fast(self, idx: int, motion_scores: np.ndarray) -> dict:
        win  = int(self.fps * self.config["windows"]["motion_window_seconds"])
        s, e = max(0, idx - win), min(len(motion_scores), idx + win)
        seg  = motion_scores[s:e] if s < e else np.array([0.0], dtype=np.float32)
        mean = float(seg.mean()); mx = float(seg.max())
        return {
            "motion_mean": mean, "motion_max": mx,
            "motion_p90": float(np.percentile(seg, 90)),
            "motion_std": float(seg.std()),
            "motion_peak_ratio": mx / (mean + 1e-6),
        }

    def _extract_temporal_context_fast(
        self, idx: int, motion_scores: np.ndarray, audio_feat: dict
    ) -> dict:
        win  = self.config["windows"]["temporal_window_frames"]
        s, e = max(0, idx - win), min(len(motion_scores), idx + win)
        seg  = motion_scores[s:e]
        mom  = float(seg[-1] - seg[0]) / len(seg) if len(seg) > 1 else 0.0
        arc  = audio_feat.get("rms_contrast", 0.0)
        d    = float(motion_scores[idx] - motion_scores[idx-1]) if 0 < idx < len(motion_scores)-1 else 0.0
        return {
            "motion_momentum": mom, "audio_momentum": arc,
            "combined_buildup": (mom + arc) * 0.5,
            "relative_position": float(idx / self.frames_count),
            "motion_derivative": max(0.0, d),
        }