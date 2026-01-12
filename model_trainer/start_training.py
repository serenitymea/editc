from .dataset import DatasetBuilder
from .train_model import ModelTrainer
from .ui import LabelingWidget
from cliper import VideoLoader, AudioAnalyzer
from .training_detector import TrainingDetector
from pathlib import Path
from datetime import datetime

class LabelingController:

    def __init__(self, video_path: str, music_path: str, output_path:str):
        super().__init__()
        
        self.loader = VideoLoader(video_path)
        self.audioanalyzer = AudioAnalyzer(video_path, music_path)
        self.detector = TrainingDetector(self.loader, self.audioanalyzer, model=None, audio_loops=12)
        self.dataset = DatasetBuilder()
        self.trainer = ModelTrainer()
        
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.widget = LabelingWidget(video_path)
        self.clips = self.detector.detect_perfect_clips()
        self.index = 0

        self.widget.good_clicked.connect(lambda: self.rate(1))
        self.widget.bad_clicked.connect(lambda: self.rate(0))
        self.widget.finish_clicked.connect(self.finish)

    def widget_view(self):
        return self.widget

    def start(self):
        if not self.clips:
            return
        self._play_current()

    def _play_current(self):
        clip = self.clips[self.index]
        self.widget.play_clip(clip.start, clip.duration)

    def rate(self, label: int):
        clip = self.clips[self.index]
        self.dataset.add_sample(clip.features, label)

        self.index += 1
        if self.index < len(self.clips):
            self._play_current()

    def finish(self):
        X, y = self.dataset.get_xy()
        self.trainer.train(X, y)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"epic_model_{timestamp}.pkl"

        self.trainer.save(model_path)

        self.widget.close()

        print(f"saved: {model_path}")
    
