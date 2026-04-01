import sys
import os
import glob
import subprocess
import shutil
import tempfile
import math

try:
    import librosa
    import numpy as np
except ImportError:
    print("download librosa: pip install librosa")
    sys.exit(1)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar,
    QTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QColor, QPalette

class VideoWorker(QObject):


    progress = Signal(int)
    log      = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, photos_dir: str, audio_path: str, output_dir: str):
        super().__init__()
        self.photos_dir  = photos_dir
        self.audio_path  = audio_path
        self.output_dir  = output_dir
        self._cancelled  = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._process()
        except Exception as exc:
            self.finished.emit(False, f"Ошибка: {exc}")

    def _process(self):
        self.log.emit("finding photos…")
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        photos = []
        for ext in extensions:
            photos += glob.glob(os.path.join(self.photos_dir, ext))
        photos = sorted(set(photos))

        if not photos:
            self.finished.emit(False, "photos not found")
            return

        self.log.emit(f"found photos: {len(photos)}")
        self.progress.emit(5)

        self.log.emit("analyze audio...")
        beat_times = self._detect_beats(self.audio_path)
        if beat_times is None:
            return

        audio_duration = self._get_audio_duration(self.audio_path)
        self.log.emit(f"found bits: {len(beat_times)}, audio length: {audio_duration:.1f} с")
        self.progress.emit(25)

        if self._cancelled:
            self.finished.emit(False, "Canceled")
            return
        
        tmp_dir = os.path.join(os.getcwd(), "tmp_photo2video")

        os.makedirs(tmp_dir, exist_ok=True)

        self.log.emit(f"tmpdir: {tmp_dir}")

        try:
            max_w, max_h = 1080, 1920
            self.log.emit(f"wideo size: {max_w}×{max_h}")

            self.log.emit("Convert photos to jepeg")
            photos = self._convert_photos(photos, tmp_dir, max_w, max_h)
            if photos is None:
                return
            self.log.emit(f"Convert ended: {len(photos)} frames ready")
            self.progress.emit(50)

            self.log.emit("load frames…")
            frame_list_path, actual_duration = self._build_frame_list(
                photos, beat_times, audio_duration, tmp_dir, max_w, max_h
            )
            self.log.emit(f"Vid length: {actual_duration:.1f} from (from {audio_duration:.1f} from audio)")
            self.progress.emit(70)

            if self._cancelled:
                self.finished.emit(False, "Canceled")
                return

            self.log.emit("Doing video…")
            output_path = os.path.join(
                self.output_dir,
                "output_" + os.path.splitext(os.path.basename(self.audio_path))[0] + ".mp4"
            )
            ok = self._run_ffmpeg(frame_list_path, self.audio_path, output_path, actual_duration, max_w, max_h)
            if not ok:
                return

            self.progress.emit(100)
            self.finished.emit(True, f"Ready: {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _convert_photos(self, photos: list, tmp_dir: str, width: int, height: int):

        converted = []
        n = len(photos)

        W, H = width, height
        vf = (
            f"scale='if(gt(iw/ih,{W}/{H}),{W},-2)':'if(gt(iw/ih,{W}/{H}),-2,{H})',"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuvj420p"
        )

        for i, photo in enumerate(photos):
            if self._cancelled:
                self.finished.emit(False, "canceled")
                return None

            out_path = os.path.join(tmp_dir, f"frame_{i:05d}.jpg")

            cmd = [
                "ffmpeg", "-y",
                "-i", photo,
                "-vf", vf,
                "-q:v", "2",
                out_path,
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=30,
                )
                if result.returncode != 0 or not os.path.exists(out_path):
                    self.log.emit(f"ERROR {os.path.basename(photo)}")
                    continue
            except Exception as exc:
                self.log.emit(f"ERROR {os.path.basename(photo)}: {exc}")
                continue

            converted.append(out_path)

            pct = 5 + int(45 * (i + 1) / n)
            self.progress.emit(pct)

        if not converted:
            self.finished.emit(False, "all photos ERROR")
            return None

        return converted

    def _detect_beats(self, audio_path: str):

        try:

            y, sr = librosa.load(audio_path, sr=None, mono=True)

            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            return beat_times
        except Exception as exc:
            self.finished.emit(False, f"Ошибка анализа аудио: {exc}")
            return None

    def _get_audio_duration(self, audio_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            return len(y) / sr

    def _max_resolution(self, photos: list) -> tuple[int, int]:

        max_w, max_h = 1280, 720

        for photo in photos:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                photo
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                parts = result.stdout.strip().split("x")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    if w * h > max_w * max_h:
                        max_w, max_h = w, h
            except Exception:
                continue

        return max_w, max_h

    def _build_frame_list(
        self,
        photos: list,
        beat_times,
        audio_duration: float,
        tmp_dir: str,
        width: int,
        height: int,
    ) -> str:

        all_times = list(beat_times) + [audio_duration]

        max_frames = min(len(photos), len(all_times) - 1)
        self.log.emit(f"frames: {max_frames} from {len(photos)} photo")

        concat_lines = []

        for i in range(max_frames):
            start_time = all_times[i]
            end_time   = all_times[i + 1]
            duration   = end_time - start_time
            if duration <= 0:
                duration = 0.04

            photo_path = photos[i]

            safe_path = photo_path.replace("'", "'\\''")

            concat_lines.append(f"file '{safe_path}'")
            concat_lines.append(f"duration {duration:.6f}")

            pct = 50 + int(20 * (i + 1) / max_frames)
            self.progress.emit(pct)

        last_photo = photos[max_frames - 1]
        concat_lines.append(f"file '{last_photo}'")

        list_path = os.path.join(tmp_dir, "frames.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(concat_lines))

        actual_duration = all_times[max_frames]
        return list_path, actual_duration

    def _run_ffmpeg(
        self,
        frame_list_path: str,
        audio_path: str,
        output_path: str,
        audio_duration: float,
        width: int = 1080,
        height: int = 1920,
    ) -> bool:

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", frame_list_path,
            "-i", audio_path,
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(audio_duration),
            "-movflags", "+faststart",
            output_path,
        ]

        self.log.emit("▶ " + " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.log.emit(line)
                if self._cancelled:
                    proc.terminate()
                    self.finished.emit(False, "denied by user")
                    return False

            proc.wait()
            if proc.returncode != 0:
                self.finished.emit(False, f"FFmpeg end with code {proc.returncode}")
                return False

            return True

        except FileNotFoundError:
            self.finished.emit(
                False,
                "FFmpeg not found"
            )
            return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo → Video")
        self.setMinimumSize(720, 580)
        self._worker = None
        self._thread = None
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Photo → Beat Video")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("divider")
        root.addWidget(line)

        root.addWidget(self._row_label("Choose photo folder:"))
        h1 = QHBoxLayout()
        self.photos_edit = QLineEdit()
        self.photos_edit.setPlaceholderText("Choose folder…")
        btn1 = QPushButton("View")
        btn1.clicked.connect(self._pick_photos_dir)
        h1.addWidget(self.photos_edit)
        h1.addWidget(btn1)
        root.addLayout(h1)

        root.addWidget(self._row_label("Music file (mp3 / wav):"))
        h2 = QHBoxLayout()
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("Choose file…")
        btn2 = QPushButton("View")
        btn2.clicked.connect(self._pick_audio)
        h2.addWidget(self.audio_edit)
        h2.addWidget(btn2)
        root.addLayout(h2)

        root.addWidget(self._row_label("where to save:"))
        h3 = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Choose folder…")
        btn3 = QPushButton("View")
        btn3.clicked.connect(self._pick_output_dir)
        h3.addWidget(self.output_edit)
        h3.addWidget(btn3)
        root.addLayout(h3)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        root.addLayout(btn_row)

        log_label = self._row_label("Log:")
        root.addWidget(log_label)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        root.addWidget(self.log_box)

    def _row_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("rowLabel")
        return lbl

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f0f13;
                color: #e0e0e8;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }
            QLabel#title {
                font-size: 22px;
                font-weight: bold;
                color: #7eb8f7;
                padding: 8px 0;
                letter-spacing: 2px;
            }
            QLabel#rowLabel {
                color: #9ba8c0;
                font-size: 12px;
                margin-top: 4px;
            }
            QFrame#divider {
                color: #2a2a3a;
                margin: 4px 0;
            }
            QLineEdit {
                background: #1a1a24;
                border: 1px solid #2e2e44;
                border-radius: 4px;
                padding: 6px 10px;
                color: #d0d8f0;
            }
            QLineEdit:focus {
                border-color: #7eb8f7;
            }
            QPushButton {
                background: #1e2a3a;
                border: 1px solid #3a4a60;
                border-radius: 4px;
                padding: 6px 16px;
                color: #a0b8d8;
                min-width: 80px;
            }
            QPushButton:hover {
                background: #253040;
                border-color: #7eb8f7;
                color: #cde4ff;
            }
            QPushButton#startBtn {
                background: #1a3a5c;
                border-color: #4a8abf;
                color: #b0d8ff;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 30px;
            }
            QPushButton#startBtn:hover {
                background: #205080;
                color: #e0f0ff;
            }
            QPushButton#startBtn:disabled {
                background: #111820;
                color: #3a4a5a;
                border-color: #1e2a36;
            }
            QPushButton#cancelBtn {
                background: #3a1a1a;
                border-color: #6a2a2a;
                color: #e08080;
            }
            QPushButton#cancelBtn:hover {
                background: #502020;
                color: #ffaaaa;
            }
            QPushButton#cancelBtn:disabled {
                background: #1a1010;
                color: #3a2020;
                border-color: #1e1010;
            }
            QProgressBar {
                background: #1a1a24;
                border: 1px solid #2e2e44;
                border-radius: 4px;
                height: 22px;
                text-align: center;
                color: #7eb8f7;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a4a7a, stop:1 #3a8abf);
                border-radius: 3px;
            }
            QTextEdit#logBox {
                background: #0a0a10;
                border: 1px solid #1e2030;
                border-radius: 4px;
                color: #70a0c0;
                font-size: 11px;
            }
        """)

    def _pick_photos_dir(self):
        d = QFileDialog.getExistingDirectory(self, "photo folder")
        if d:
            self.photos_edit.setText(d)

    def _pick_audio(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "music file", "", "audio (*.mp3 *.wav *.flac *.ogg *.m4a)"
        )
        if f:
            self.audio_edit.setText(f)

    def _pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Save folder")
        if d:
            self.output_edit.setText(d)

    def _start(self):
        photos_dir = self.photos_edit.text().strip()
        audio_path = self.audio_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        errors = []
        if not photos_dir or not os.path.isdir(photos_dir):
            errors.append("Bad photo folder")
        if not audio_path or not os.path.isfile(audio_path):
            errors.append("Bad audio")
        if not output_dir or not os.path.isdir(output_dir):
            errors.append("Bad folder")
        if errors:
            self._log("ERROR" + "\n".join(errors))
            return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self._set_running(True)

        self._thread = QThread()
        self._worker = VideoWorker(photos_dir, audio_path, output_dir)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_finished)

        self._thread.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._log("request to denie")

    def _on_finished(self, success: bool, message: str):
        self._log(("OK" if success else "ERROR") + message)
        self._set_running(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _log(self, text: str):
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())