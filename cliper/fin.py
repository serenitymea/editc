from .audioanalyzer import AudioAnalyzer
from .vinloader import VideoLoader
from .epicdetector import EpicDetector
from .aechecker import AestheticChecker
# from .exporter import ClipExporter


class ClipP:
    def __init__(self, video_path: str):
        self.loader = VideoLoader(video_path)
        self.audio = AudioAnalyzer(video_path)
        self.detector = EpicDetector(self.loader, self.audio)
        self.checker = AestheticChecker(self.loader)
        # self.exporter = ClipExporter(video_path)

    def run(self):

        clips = self.detector.detect()

        approved_clips = self.checker.review(clips)

        self.loader.release()

        return approved_clips

        # self.exporter.export(approved_clips, output_path="output/epic_clips.mp4")
        # self.exporter.close()
