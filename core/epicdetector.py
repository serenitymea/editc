import cv2
import numpy as np

class EpicDetector:
    def __init__(self, loader, audio_analyzer):
        self.loader = loader
        self.audio_analyzer = audio_analyzer
        self.epic_frames = []

    def detect(self, visual_thresh=60, audio_thresh=0.01, context=15):
        prev_gray = None
        frame_count = int(self.loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        epic_frames = []

        for i, frame in enumerate(self.loader.frames()):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                visual_change = np.mean(diff)
                audio_energy = self.audio_analyzer.get_energy(i, frame_count)

                if visual_change > visual_thresh and audio_energy > audio_thresh:
                    start = max(0, i - context)
                    end = min(frame_count - 1, i + context)
                    epic_frames.extend(range(start, end + 1))

            prev_gray = gray

        epic_frames = sorted(set(epic_frames))
        return epic_frames
