import os
import numpy as np
import pytest
import soundfile as sf
import subprocess
from cliper.audioanalyzer import AudioAnalyzer

TEST_DIR = "tests/tmp"
os.makedirs(TEST_DIR, exist_ok=True)

TEST_MUSIC = os.path.join(TEST_DIR, "test_music.wav")
TEST_VIDEO = os.path.join(TEST_DIR, "test_video.mp4")

sr = 44100
y = np.zeros(sr)
for i in range(0, sr, sr//4):
    y[i:i+100] = 1.0
sf.write(TEST_MUSIC, y, sr)

subprocess.run([
    "ffmpeg", "-y",
    "-f", "lavfi", "-i", "color=c=black:s=128x128:d=1",
    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
    "-c:v", "libx264", "-c:a", "aac",
    "-shortest",
    TEST_VIDEO
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_load_music_and_video():
    analyzer = AudioAnalyzer(TEST_VIDEO, TEST_MUSIC)
    assert analyzer.music_y is not None
    assert analyzer.music_sr > 0
    assert analyzer.video_y is not None
    assert analyzer.video_sr > 0


def test_tempo_and_beats():
    analyzer = AudioAnalyzer(TEST_VIDEO, TEST_MUSIC)
    assert analyzer.tempo > 0
    assert isinstance(analyzer.beat_times, np.ndarray)
    assert len(analyzer.beat_times) > 0


def test_get_beat_intervals():
    analyzer = AudioAnalyzer(TEST_VIDEO, TEST_MUSIC)
    intervals = analyzer.get_beat_intervals()
    assert isinstance(intervals, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in intervals)
    assert all(t[1] > 0 for t in intervals)


def test_audio_energy():
    analyzer = AudioAnalyzer(TEST_VIDEO, TEST_MUSIC)
    total_frames = 100
    energies = [analyzer.audio_energy(i, total_frames) for i in range(total_frames)]
    assert all(0.0 <= e <= 1.0 for e in energies)
    assert any(e > 0 for e in energies)


def teardown_module(module):
    if os.path.exists(TEST_MUSIC):
        os.remove(TEST_MUSIC)
    if os.path.exists(TEST_VIDEO):
        os.remove(TEST_VIDEO)
