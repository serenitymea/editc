import subprocess
from typing import List, Optional
from .vinloader import VideoLoader
from .audioanalyzer import AudioAnalyzer
from .epicdetector import EpicDetector, Clip


class ClipP:

    def __init__(self, video_path: str, music_path: str):
        self.loader = VideoLoader(video_path)
        self.audio = AudioAnalyzer(video_path, external_audio=music_path)
        self.detector = EpicDetector(self.loader, self.audio)
        self.music_path = music_path

    def run(
        self,
        clip_duration: float = 0.6,
        target_duration: Optional[float] = None,
        keep_top_percent: int = 80,
    ) -> List[Clip]:

        if target_duration is None:
            target_duration = self._get_audio_duration()

        try:
            return self.detector.detect_for_edit(
                target_duration=target_duration,
                clip_duration=clip_duration,
                keep_top_percent=keep_top_percent,
            )
        finally:
            self.loader.release()

    def _get_audio_duration(self) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.music_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"ffprobe failed: {result.stderr.strip()}"
            )

        return float(result.stdout.strip())
