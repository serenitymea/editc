import os
import subprocess
import numpy as np
import librosa
from typing import List
from functools import lru_cache


class AudioAnalyzer:

    def __init__(self, video_path, music_path):
        os.makedirs("tmp", exist_ok=True)

        print(" [AA]load m...")

        self.music_y, self.music_sr = librosa.load(
            music_path, sr=22050, mono=True
        )

        print("   [AA]a beats...")
        self.tempo, beats_frames = librosa.beat.beat_track(
            y=self.music_y,
            sr=self.music_sr,
            units="frames"
        )

        if isinstance(self.tempo, np.ndarray):
            self.tempo = float(self.tempo[0])

        self.beat_times = librosa.frames_to_time(
            beats_frames, sr=self.music_sr
        )

        tmp_audio = "tmp/video_audio.wav"
        try:
            self._extract_audio_ffmpeg(video_path, tmp_audio)

            self.video_y, self.video_sr = librosa.load(
                tmp_audio, sr=22050, mono=True
            )

            self.rms = librosa.feature.rms(
                y=self.video_y, hop_length=1024
            )[0]

            self._precompute_advanced_features()
            
        finally:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)

        print(f"   [AA]temp: {self.tempo:.1f} BPM")
        print(f"   [AA]f beats: {len(self.beat_times)}")

    def _extract_audio_ffmpeg(self, video_path, out_path):
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "22050",
            "-f", "wav",
            out_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg audio extraction failed")

    def _precompute_advanced_features(self):
        """Precompute spectral and rhythm features"""
        print("   [AA]computing advanced features...")

        hop_length = 1024

        self.spectral_centroid = librosa.feature.spectral_centroid(
            y=self.video_y, sr=self.video_sr, hop_length=hop_length
        )[0]
        
        self.spectral_rolloff = librosa.feature.spectral_rolloff(
            y=self.video_y, sr=self.video_sr, hop_length=hop_length
        )[0]

        mfcc = librosa.feature.mfcc(
            y=self.video_y, sr=self.video_sr, n_mfcc=7, hop_length=hop_length
        )
        self.mfcc_variance = np.var(mfcc, axis=0)

        S = np.abs(librosa.stft(self.video_y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=self.video_sr, n_fft=2048)
        bass_mask = freqs < 200
        self.bass_energy = np.mean(S[bass_mask], axis=0)

        self.onset_env = librosa.onset.onset_strength(
            y=self.video_y, sr=self.video_sr, hop_length=hop_length
        )

        self.harmonicity = self._compute_harmonicity_fast()
        
        print("   [AA]advanced features ready")

    def _compute_harmonicity_fast(self):
        """harmonicity measure - simplified version"""
        hop_length = 1024

        spectral_flatness = librosa.feature.spectral_flatness(
            y=self.video_y, hop_length=hop_length
        )[0]

        harmonicity = 1.0 - spectral_flatness
        
        return harmonicity

    def get_beat_intervals(self) -> List[tuple]:
        return [
            (self.beat_times[i],
             self.beat_times[i + 1] - self.beat_times[i])
            for i in range(len(self.beat_times) - 1)
        ]

    @lru_cache(maxsize=1024)
    def audio_energy(self, frame_idx, total_frames) -> float:
        if total_frames <= 0 or len(self.rms) == 0:
            return 0.0

        idx = int(frame_idx / total_frames * len(self.rms))
        idx = max(0, min(idx, len(self.rms) - 1))

        return float(self.rms[idx] / (self.rms.max() + 1e-6))
    
    def get_advanced_audio_features(self, frame_idx, total_frames) -> dict:
        """Get all advanced audio features for a frame"""
        if total_frames <= 0:
            return self._empty_audio_features()
        
        idx = int(frame_idx / total_frames * len(self.rms))
        idx = max(0, min(idx, len(self.rms) - 1))

        def safe_idx(arr):
            mapped = int(frame_idx / total_frames * len(arr))
            return max(0, min(mapped, len(arr) - 1))

        idx_sc = safe_idx(self.spectral_centroid)
        idx_sr = safe_idx(self.spectral_rolloff)
        idx_mfcc = safe_idx(self.mfcc_variance)
        idx_bass = safe_idx(self.bass_energy)
        idx_harm = safe_idx(self.harmonicity)
        idx_onset = safe_idx(self.onset_env)

        window_start = max(0, idx - 5)
        window_end = min(len(self.rms), idx + 5)
        rms_window = self.rms[window_start:window_end]
        
        return {
            "rms_mean": float(self.rms[idx]),
            "rms_peak": float(rms_window.max() if len(rms_window) > 0 else 0),
            "rms_contrast": float(self.rms[idx] - self.rms[max(0, idx-10):idx].mean()) if idx > 10 else 0.0,
            
            "spectral_centroid": float(self.spectral_centroid[idx_sc]),
            "spectral_rolloff": float(self.spectral_rolloff[idx_sr]),
            "mfcc_variance": float(self.mfcc_variance[idx_mfcc]),
            
            "bass_energy": float(self.bass_energy[idx_bass]),
            "vocal_probability": float(self.harmonicity[idx_harm]),
            
            "onset_density": float(self.onset_env[idx_onset]),
        }
    
    @staticmethod
    def _empty_audio_features():
        return {
            "rms_mean": 0.0,
            "rms_peak": 0.0,
            "rms_contrast": 0.0,
            "spectral_centroid": 0.0,
            "spectral_rolloff": 0.0,
            "mfcc_variance": 0.0,
            "bass_energy": 0.0,
            "vocal_probability": 0.0,
            "onset_density": 0.0,
        }
    
    def get_beat_strength(self, frame_idx, total_frames) -> float:
        """How close is this frame to a beat - OPTIMIZED"""
        if total_frames <= 0 or len(self.beat_times) == 0:
            return 0.0
        
        time_sec = frame_idx / total_frames * (len(self.video_y) / self.video_sr)

        closest_beat_dist = np.abs(self.beat_times - time_sec).min()
 
        beat_strength = np.exp(-closest_beat_dist * 5)
        
        return float(beat_strength)
    
    @lru_cache(maxsize=256)
    def get_beat_alignment_score(self, start_time_sec) -> float:
        """How well aligned is this start time with a beat"""
        if len(self.beat_times) == 0:
            return 0.0

        closest_dist = np.abs(self.beat_times - start_time_sec).min()
        
        alignment = np.exp(-closest_dist * 10)
        
        return float(alignment)