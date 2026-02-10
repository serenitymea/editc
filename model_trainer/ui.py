from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Signal, QTimer, QUrl

class LabelingWidget(QWidget):
    good_clicked = Signal()
    bad_clicked = Signal()
    finish_clicked = Signal()

    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle("Epic Detector Trainer")

        self.video_widget = QVideoWidget(self)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)

        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio)
        self.player.setSource(QUrl.fromLocalFile(video_path))

        self.info_label = QLabel("Watch the clip and rate it")

        self.good_button = QPushButton("Good clip")
        self.bad_button = QPushButton("Bad clip")
        self.finish_button = QPushButton("Finish training")

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_widget)
        layout.addWidget(self.info_label)
        layout.addWidget(self.good_button)
        layout.addWidget(self.bad_button)
        layout.addWidget(self.finish_button)

        self.good_button.clicked.connect(self.good_clicked.emit)
        self.bad_button.clicked.connect(self.bad_clicked.emit)
        self.finish_button.clicked.connect(self.finish_clicked.emit)

        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.player.pause)

    def play_clip(self, start_sec: float, duration_sec: float):
        self.player.pause()

        def start():
            self.player.setPosition(int(start_sec * 1000))
            self.player.play()
            self.stop_timer.start(int(duration_sec * 1000))

        QTimer.singleShot(100, start)