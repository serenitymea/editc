import subprocess
from pathlib import Path
from .presets import VideoEffects
from .effectsp import EffectProcessor

MIN_SPEED = 0.6
MAX_SPEED = 2.2
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def beats_to_seconds(beats: int, bpm: float) -> float:
    return beats * 60 / bpm


def calc_speed(clip, bpm: float, beats: int) -> float:
    return clip.duration / beats_to_seconds(beats, bpm)


class VideoRenderer:
    def __init__(
        self,
        input_video: str,
        output_video: str,
        bpm: float,
        beats_per_clip: int,
        music_file: str = None,
        resolution: str = "1920:1080",
        fps: int = 30,
        effect_processor: EffectProcessor = None,
    ):
        self.input_video = input_video
        self.output_video = output_video
        self.bpm = bpm
        self.beats_per_clip = beats_per_clip
        self.resolution = resolution
        self.fps = fps
        self.music_file = music_file or self._find_music()
        self.effect_processor = effect_processor

    def _find_music(self) -> str:
        input_dir = Path.cwd() / "input"
        if not input_dir.exists():
            raise FileNotFoundError("input directory not found")
        for file in input_dir.iterdir():
            if file.suffix.lower() in AUDIO_EXTENSIONS:
                return str(file)
        raise FileNotFoundError("no audio file found in input directory")

    def render_edit(self, clips: list):
        
        clips = list(clips)
        if not clips:
            raise ValueError("clips list is empty")

        filters = []
        concat_streams = []
        total_clips = len(clips)

        for i, clip in enumerate(clips):
            speed = calc_speed(clip, self.bpm, self.beats_per_clip)
            speed = max(MIN_SPEED, min(MAX_SPEED, speed))

            effects = VideoEffects(clip, speed, self.effect_processor)
            preset_fn = effects.get_preset(i, total_clips)

            filters.append(preset_fn(i, total_clips))
            concat_streams.append(f"[v{i}]")

        filter_complex = ";".join(filters)
        filter_complex += ";" + "".join(concat_streams)
        filter_complex += f"concat=n={len(clips)}:v=1:a=0[outv]"

        cmd = [
            "ffmpeg", "-y",
            "-i", self.input_video,
            "-i", self.music_file,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "1:a",
            "-r", str(self.fps),
            "-s", self.resolution,
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
            self.output_video,
        ]

        subprocess.run(cmd, check=True)