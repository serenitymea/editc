import subprocess
from pathlib import Path


class AddAudio:
    def __init__(self, video_path, audio_path, output_path):
        self.video = Path(video_path)
        self.audio = Path(audio_path)
        self.output = Path(output_path)

    def merge(self, overwrite=True):
        cmd = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-i", str(self.video),
            "-i", str(self.audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(self.output),
        ]

        subprocess.run(cmd, check=True)

        return self.output
