from .dataset import DatasetBuilder
from .train_model import ModelTrainer
from .ui import LabelingWidget
from cliper import VideoLoader, AudioAnalyzer
from .training_detector import TrainingDetector
from pathlib import Path
from datetime import datetime
from tools.config_loader import load_default_config

class LabelingController:

    def __init__(self, video_path: str, music_path: str, output_path: str):
        super().__init__()
        
        self.config = load_default_config()
        self.loader = VideoLoader(video_path)
        self.audioanalyzer = AudioAnalyzer(video_path, music_path)
        self.detector = TrainingDetector(
            self.config,
            self.loader, 
            self.audioanalyzer, 
            model=None, 
        )
        self.dataset = DatasetBuilder()
        self.trainer = ModelTrainer(use_catboost=True)
        
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.widget = LabelingWidget(video_path)

        print("\n[LC] Detecting clips for labeling...")
        self.clips = self.detector.detect_perfect_clips()
        self.all_features = self.detector._compute_all_features() if hasattr(self.detector, '_compute_all_features') else []
        
        self.index = 0

        self.widget.good_clicked.connect(lambda: self.rate(1))
        self.widget.bad_clicked.connect(lambda: self.rate(0))
        self.widget.finish_clicked.connect(self.finish)
        
        print(f"[LC] Ready to label {len(self.clips)} clips")

    def widget_view(self):
        return self.widget

    def start(self):
        if not self.clips:
            print("[LC] No clips detected for training")
            return
        self._play_current()

    def _play_current(self):
        clip = self.clips[self.index]
        self.widget.play_clip(clip.start, clip.duration)
        print(f"[LC] Showing clip {self.index + 1}/{len(self.clips)}")

    def rate(self, label: int):
        clip = self.clips[self.index]
        self.dataset.add_sample(clip.features, label)
        
        print(f"[LC] Clip {self.index + 1} labeled as {'GOOD' if label == 1 else 'BAD'}")

        self.index += 1
        if self.index < len(self.clips):
            self._play_current()
        else:
            print("[LC] All clips labeled! Click 'Finish training' to train model.")

    def finish(self):
        X, y = self.dataset.get_xy()
        
        if len(X) == 0:
            print("[LC] No samples collected - cannot train")
            self.widget.close()
            return
        
        print(f"\n[LC] Training model with {len(X)} samples...")

        positive_clips = [c for c, label in zip(self.clips, y) if label == 1]
        
        if positive_clips and self.all_features:
            print("[LC] Generating hard negatives...")
            self.dataset.generate_hard_negatives(
                positive_clips, 
                self.all_features, 
                self.loader.fps
            )
            
            print("[LC] Generating easy negatives...")
            self.dataset.generate_easy_negatives(self.all_features)

        self.dataset.balance_dataset(target_ratio=3.0)

        self.dataset.print_statistics()

        X, y = self.dataset.get_xy()

        (X_train, y_train), (X_val, y_val) = self.dataset.split_train_val(val_ratio=0.2)

        self.trainer.train(X_train, y_train, X_val, y_val)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"epic_model_{timestamp}.pkl"
        self.trainer.save(model_path)

        dataset_path = self.output_dir / f"dataset_{timestamp}.json"
        self.dataset.save(dataset_path)

        self.widget.close()

        print(f"\n[LC] Model saved: {model_path}")
        print(f"[LC] Dataset saved: {dataset_path}")