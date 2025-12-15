from core.audioanalyzer import AudioAnalyzer
from core.clipexporter import ClipExporter
from core.vinloader import VideoLoader
from core.epicdetector import EpicDetector
from checker.aechecker import AestheticChecker

import cv2

video_path = "input/g1.mp4"

loader = VideoLoader(video_path)
audio = AudioAnalyzer(video_path)

detector = EpicDetector(loader, audio)
epic_frames = detector.detect()

checker = AestheticChecker(loader)
final_frames = checker.review(epic_frames)

fps = loader.cap.get(cv2.CAP_PROP_FPS)
loader.release()

exporter = ClipExporter(video_path)
exporter.export(final_frames, fps)