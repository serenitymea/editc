from core.audioanalyzer import AudioAnalyzer
from core.clipexporter import ClipExporter
from core.vinloader import VideoLoader
from core.epicdetector import EpicDetector
from checker.aechecker import AestheticChecker

video_path = "input/g1.mp4"

loader = VideoLoader(video_path)
audio = AudioAnalyzer(video_path)

detector = EpicDetector(loader, audio)
clips = detector.detect()

checker = AestheticChecker(loader)
approved_clips = checker.review(clips)

loader.release()

exporter = ClipExporter(video_path)
exporter.export(approved_clips, output_path="output/epic_clips.mp4")
exporter.close()
