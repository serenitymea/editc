import os
import subprocess
import numpy as np
import librosa
from typing import List


class AudioAnalyzer:

    def __init__(self, video_path, music_path):
        os.makedirs("tmp", exist_ok=True)

        print(" [AA]load m...")
        self.music_y, self.music_sr = librosa.load(
            music_path, sr=None, mono=True
        )

        print("   [AA]anal beats...")
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
                tmp_audio, sr=None, mono=True
            )
            self.rms = librosa.feature.rms(
                y=self.video_y, hop_length=512
            )[0]
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
            "-ar", "44100",
            "-f", "wav",
            out_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg audio extraction failed")

    def get_beat_intervals(self) -> List[tuple]:

        return [
            (self.beat_times[i],
             self.beat_times[i + 1] - self.beat_times[i])
            for i in range(len(self.beat_times) - 1)
        ]

    def audio_energy(self, frame_idx, total_frames) -> float:

        if total_frames <= 0 or len(self.rms) == 0:
            return 0.0

        idx = int(frame_idx / total_frames * len(self.rms))
        idx = max(0, min(idx, len(self.rms) - 1))

        return float(self.rms[idx] / (self.rms.max() + 1e-6))
