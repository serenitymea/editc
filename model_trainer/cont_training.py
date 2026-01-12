from .dataset import DatasetBuilder
from .train_model import ModelTrainer
from .ui import LabelingWidget
from cliper import VideoLoader, AudioAnalyzer
from .training_detector import TrainingDetector
from pathlib import Path
from datetime import datetime

import glob
import joblib

def _get_mlmodel():
    model_files = sorted(glob.glob("model_output/epic_model_*.pkl"))

    if not model_files:
        return None

    latest_model_path = model_files[-1]
    model = joblib.load(latest_model_path)
    print(f"MODELMODELMODELMODEL{latest_model_path}")
    return model

class FineTuneController:

    def __init__(self, video_path: str, music_path: str, output_path: str):

        super().__init__()
        
        self.model_path = _get_mlmodel()
        
        self.loader = VideoLoader(video_path)
        self.audioanalyzer = AudioAnalyzer(video_path, music_path)

        self.trainer = ModelTrainer()
        self.trainer.load(self.model_path)

        self.detector = TrainingDetector(
            self.loader, 
            self.audioanalyzer, 
            model=self.trainer.model, 
            audio_loops=12
        )
        
        self.dataset = DatasetBuilder()
        self.original_model_path = Path(self.model_path)
        
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
            print("No clips detected for fine-tuning")
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
        
        if len(X) == 0:
            print("No samples collected for fine-tuning")
            self.widget.close()
            return

        self.trainer.train(X, y)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"epic_model_finetuned_{timestamp}.pkl"

        self.trainer.save(model_path)

        self.widget.close()

        print(f"tuned model saved: {model_path}")
        print(f"original model: {self.original_model_path}")