import librosa
import numpy as np
from moviepy import VideoFileClip
import os

class AudioAnalyzer:
    def __init__(self, path):
        temp_audio = "tmp/temp_audio.wav"
        try:
            clip = VideoFileClip(path)
            clip.audio.write_audiofile(temp_audio, logger=None)
            clip.close()
            
            self.y, self.sr = librosa.load(temp_audio, sr=None, mono=True)
            self.energy = np.array([sum(self.y[i:i+1024]**2) for i in range(0, len(self.y), 1024)])
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def get_energy(self, frame_idx, total_frames):
        if len(self.energy) == 0:
            return 0
        audio_idx = int(frame_idx * len(self.energy) / total_frames)
        audio_idx = min(audio_idx, len(self.energy) - 1)
        return self.energy[audio_idx]