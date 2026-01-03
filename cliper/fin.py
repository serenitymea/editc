import subprocess
from typing import List, Optional
from .vinloader import VideoLoader
from .audioanalyzer import AudioAnalyzer
from .epicdetector import EpicDetector

class ClipP:

    def __init__(self, video_path: str, music_path: str):
        self.video_path = video_path
        self.music_path = music_path

        self.loader = VideoLoader(video_path)
        self.audio = AudioAnalyzer(video_path, music_path)
        self.detector = EpicDetector(self.loader, self.audio)

    def run(
        self,
        target_duration: Optional[float] = None,
        max_clips: Optional[int] = None,
    ) -> List:

        if target_duration is None:
            target_duration = self._get_audio_duration()

        if max_clips is None and target_duration:
            beat_intervals = self.audio.get_beat_intervals()
            cumulative = 0
            for i, (_, duration) in enumerate(beat_intervals):
                cumulative += duration
                if cumulative >= target_duration:
                    max_clips = i + 1
                    break
        
        try:
            
            clips = self.detector.detect_perfect_clips(max_clips=max_clips)
            
            print(f"\nf clips: {len(clips)}")
            if clips:
                total = sum(c.duration for c in clips)
                print(f"   len: {total:.2f}s")
                print(f"   e score: {sum(c.score for c in clips) / len(clips):.3f}")
            
            return clips
            
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
            check=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"ffprobe failed: {result.stderr.strip()}"
            )

        return float(result.stdout.strip())