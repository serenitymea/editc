import os
import subprocess
import numpy as np
import librosa
from scipy.signal import find_peaks


class AudioAnalyzer:
    def __init__(self, video_path, external_audio=None):
        os.makedirs("tmp", exist_ok=True)
        tmp_audio = "tmp/audio.wav"

        try:
            self._extract_audio_ffmpeg(video_path, tmp_audio)

            self.y, self.sr = librosa.load(tmp_audio, sr=None, mono=True)
            self.rms = librosa.feature.rms(y=self.y, hop_length=512)[0]

            if external_audio and os.path.exists(external_audio):
                self.target_y, self.target_sr = librosa.load(
                    external_audio, sr=None, mono=True
                )
            else:
                self.target_y, self.target_sr = self.y, self.sr

            self.tempo, self.beats = librosa.beat.beat_track(
                y=self.target_y, sr=self.target_sr
            )

            self.peaks = self._find_energy_peaks(
                self.target_y, self.target_sr
            )

        finally:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)

    def _extract_audio_ffmpeg(self, video_path, out_path):
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "44100",
            "-f", "wav",
            out_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError("ffmpeg audio extraction failed")

    def _find_energy_peaks(self, y, sr):
        hop = 512

        rms = librosa.feature.rms(y=y, hop_length=hop)[0]

        spec = np.abs(librosa.stft(y, hop_length=hop))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))

        onset = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop
        )

        length = min(len(rms), len(flux), len(onset))
        energy = (
            rms[:length] * 0.4 +
            flux[:length] * 0.3 +
            onset[:length] * 0.3
        )

        peaks, _ = find_peaks(
            energy,
            distance=int(sr / hop * 2),
            prominence=np.percentile(energy, 80)
        )

        return librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

    def beat_times(self):
        return librosa.frames_to_time(
            self.beats, sr=self.target_sr
        )

    def peak_times(self):
        return self.peaks

    def audio_energy(self, frame_idx, total_frames):
        if total_frames <= 0 or len(self.rms) == 0:
            return 0.0

        idx = int(frame_idx / total_frames * len(self.rms))
        idx = max(0, min(idx, len(self.rms) - 1))

        return float(self.rms[idx] / (self.rms.max() + 1e-6))
