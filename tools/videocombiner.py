import subprocess
from pathlib import Path


class VideoMerger:
    def __init__(self):
        self.input_dir = Path("input")
        self.output_path = Path("input/g1.mp4")
        self.output_path.parent.mkdir(exist_ok=True)

    def run(self):
        videos = sorted(self.input_dir.glob("*.mp4"))

        if not videos:
            raise ValueError("no .mp4 files found in input folder")

        concat_file = self.output_path.parent / "concat_list.txt"

        with open(concat_file, "w", encoding="utf-8") as f:
            for v in videos:
                f.write(f"file '{v.resolve()}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(self.output_path),
        ]

        subprocess.run(cmd, check=True)

        concat_file.unlink(missing_ok=True)

        return str(self.output_path)
