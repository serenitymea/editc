import subprocess
from pathlib import Path

class VideoMerger:
    def __init__(self):
        self.input_dir = Path("input")
        self.output_path = Path("input/g1.mp4")
        self.output_path.parent.mkdir(exist_ok=True)

    def run(self, trim_seconds=0):
        videos = sorted(self.input_dir.glob("*.mp4"))

        if not videos:
            raise ValueError("No .mp4 files found in input folder")

        temp_dir = self.output_path.parent / "temp_videos"
        temp_dir.mkdir(exist_ok=True)
        trimmed_files = []

        for i, v in enumerate(videos, 1):
            temp_file = temp_dir / f"trimmed_{i}.mp4"
            duration = self.get_video_duration(v)
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(v),
                "-ss", str(trim_seconds),
                "-to", str(max(duration - trim_seconds, 0)),
                "-c", "copy",
                str(temp_file)
            ]
            subprocess.run(cmd, check=True)
            trimmed_files.append(temp_file)

        concat_file = self.output_path.parent / "concat_list.txt"
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
    def get_video_duration(path):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
