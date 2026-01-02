import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Generator, Tuple
from cliper.epicdetector import EpicDetector, Clip
from cliper.audioanalyzer import AudioAnalyzer

class DummyVideoLoader:
    def __init__(self, frames_count=10, width=64, height=64, fps=24):
        self.frames_count = frames_count
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = self
    def get(self, prop):
        if prop == 5:
            return self.fps
        elif prop == 7:
            return self.frames_count
        return 0
    def frames(self, start=0, end=None) -> Generator[Tuple[int, np.ndarray], None, None]:
        end = self.frames_count if end is None else min(end, self.frames_count)
        for i in range(start, end):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            yield i, frame
    def release(self):
        pass

class DummyAudioAnalyzer:
    def __init__(self, frames_count=10):
        self.frames_count = frames_count
        self.rms = np.linspace(0, 1, frames_count)
    def get_beat_intervals(self):
        return [(0, 0.5), (0.5, 0.5)]
    def audio_energy(self, frame_idx, total_frames):
        return float(self.rms[frame_idx % self.frames_count])

@pytest.fixture
def analyzer():
    return DummyAudioAnalyzer(frames_count=10)

@pytest.fixture
def loader():
    return DummyVideoLoader(frames_count=10)

def test_epic_detector_creation(loader, analyzer):
    detector = EpicDetector(loader, analyzer)
    assert detector.fps > 0
    assert detector.frames_count > 0
    assert detector.duration > 0

def test_detect_perfect_clips(loader, analyzer):
    detector = EpicDetector(loader, analyzer)
    clips = detector.detect_perfect_clips(max_clips=2)
    assert isinstance(clips, list)
    assert all(isinstance(c, Clip) for c in clips)
    for c in clips:
        assert c.start_frame < c.end_frame
        assert 0 <= c.start_time < c.end_time
        assert c.duration > 0
        assert 0 <= c.score <= 1
