from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QMessageBox, QFileDialog, QInputDialog
)
import os
import sys
from pathlib import Path
import shutil

from videoeffects import GlitchVideoFX
from editor import VideoPipeline
from tools import VideoMerger, LiveFXTester, AddAudio


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Editor Panel")
        self.tester = LiveFXTester()
        

        layout = QVBoxLayout()

        btn_start = QPushButton("Start (VideoPipeline)")
        btn_tester = QPushButton("EffectTester")
        btn_glitch = QPushButton("Glitch FX")

        btn_start.clicked.connect(self.run_pipeline)
        btn_tester.clicked.connect(self.run_tester)
        btn_glitch.clicked.connect(self.run_glitch)

        layout.addWidget(btn_start)
        layout.addWidget(btn_tester)
        layout.addWidget(btn_glitch)

        self.setLayout(layout)

    def notify(self, text):
        QMessageBox.information(self, "Info", text)

    def run_pipeline(self):
        
        audio_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select audio file",
            "",
            "Audio files (*.mp3 *.wav *.ogg *.flac)"
        )
        if not audio_files:
            return

        source_audio = Path(audio_files[0])
        input_dir = Path("input")
        input_dir.mkdir(exist_ok=True)

        audio_path = input_dir / "m1.mp3"
        shutil.copy(source_audio, audio_path)

        trim_seconds, ok = QInputDialog.getInt(
            self,
            "Введите число",
            "Количество секунд:",
            0,
            0,  
            10000, 
            1  
        )
        if not ok:
            return

        video_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            "",
            "Video files (*.mp4 *.mov *.mkv)"
        )
        if not video_files:
            return

        try:
            merged_video_path = input_dir / "g1.mp4"
            merger = VideoMerger(output_path=str(merged_video_path))
            merger.run(video_files, trim_seconds=trim_seconds)
        except Exception as e:
            QMessageBox.critical(self, "Merge error", str(e))
            return

        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)

        pipeline_output = tmp_dir / "pipeline.mp4"

        vp = VideoPipeline(
            input_video=str(merged_video_path),
            output_video=str(pipeline_output),
            music_file=str(audio_path),
            bpm=120,
            beats_per_clip=16
        )
        vp.run()

        final_output = Path("output/final.mp4")
        final_output.parent.mkdir(exist_ok=True)

        audio_merger = AddAudio(
            video_path=str(pipeline_output),
            audio_path=str(audio_path),
            output_path=str(final_output)
        )
        audio_merger.merge()
        
        if pipeline_output.exists() and pipeline_output.is_file():
            os.remove(pipeline_output)

        self.notify(f"ready:\n{final_output}")

    def run_tester(self):

        self.tester.run("input/test1.jpg")

    def run_glitch(self):
        
        data = self.tester.get_state()
       
        tmp2 = Path("tmp/_temp_vid_w_gltich.mp4")
        
        fx = GlitchVideoFX(input_path="output/final.mp4", output_path=tmp2)
        fx.process_video(
            brightness=data["brightness"],
            contrast=data["contrast"],
            saturation=data["saturation"],
            tone_swap=data["tone_swap"],
            mono_hue=data["mono_hue"],
            rgb_split=data["rgb_split"],
            line_glitch=data["line_glitch"]
        )
        
        merger = AddAudio(
            video_path=tmp2,
            audio_path="input/m1.mp3",
            output_path="output/glitch.mp4"
        )
        merger.merge()
        
        if tmp2.exists() and tmp2.is_file():
            tmp2.unlink()
            
        self.notify("Glitch ready")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
