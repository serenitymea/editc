import subprocess
from pathlib import Path


class VideoMerger:
    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def run(self, video_files, trim_seconds=0):
        if not video_files:
            raise ValueError("list null")

        videos = [Path(v) for v in video_files]

        for v in videos:
            if not v.exists():
                raise FileNotFoundError(f"f not found: {v}")

        temp_dir = Path("tmp")
        temp_dir.mkdir(exist_ok=True)

        trimmed_files = []

        for i, v in enumerate(videos, 1):
            temp_file = temp_dir / f"trimmed_{i}.mp4"
            duration = self.get_video_duration(v)

            end_time = max(duration - trim_seconds, 0)

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(trim_seconds),
                "-i", str(v),
                "-to", str(end_time),
                "-c", "copy",
                str(temp_file)
            ]
            subprocess.run(cmd, check=True)
            trimmed_files.append(temp_file)

        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for tf in trimmed_files:
                f.write(f"file '{tf.resolve()}'\n")

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
        for tf in trimmed_files:
            tf.unlink(missing_ok=True)
        temp_dir.rmdir()

        return str(self.output_path)

    @staticmethod
    def get_video_duration(path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
