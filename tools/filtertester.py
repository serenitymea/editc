import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider,
    QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


class LiveFXTester:
    def __init__(self):
        self.original = None
        self.h = 0
        self.w = 0
        self.prev = None
        self.app = None
        self.widget = None

        self.state = {
            "brightness": 100,
            "contrast": 100,
            "saturation": 100,
            "tone_swap": 0,
            "mono_hue": 0,
            "rgb_split": 0,
            "line_glitch": 0,
        }

    @staticmethod
    def shift_image_safe(img, dx, dy):
        h, w = img.shape[:2]
        out = np.zeros_like(img)
        x1 = max(dx, 0)
        x2 = w + min(dx, 0)
        y1 = max(dy, 0)
        y2 = h + min(dy, 0)
        out[y1:y2, x1:x2] = img[y1-dy:y2-dy, x1-dx:x2-dx]
        return out

    @staticmethod
    def shift_channel_safe(ch, dx):
        h, w = ch.shape
        out = np.zeros_like(ch)
        x1 = max(dx, 0)
        x2 = w + min(dx, 0)
        out[:, x1:x2] = ch[:, x1-dx:x2-dx]
        return out

    @staticmethod
    def tonal_swap(frame, amount):
        if amount <= 0:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[..., 2]
        hsv[..., 2] = v * (1 - amount) + (255 - v) * amount
        return cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8),
            cv2.COLOR_HSV2BGR
        )

    @staticmethod
    def monochrome_hue(frame, hue_value):
        if hue_value <= 0:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = hue_value
        return cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8),
            cv2.COLOR_HSV2BGR
        )

    def apply_fx(self, f):
        brightness = (self.state["brightness"] - 100) / 100
        contrast = self.state["contrast"] / 100
        saturation = self.state["saturation"] / 100
        tone_swap = self.state["tone_swap"] / 100
        mono_hue = self.state["mono_hue"]
        rgb_split = self.state["rgb_split"]
        line_glitch = self.state["line_glitch"]

        f = f.astype(np.float32)
        f = f * contrast + brightness * 255
        f = np.clip(f, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        f = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8),
            cv2.COLOR_HSV2BGR
        )

        if tone_swap > 0:
            f = self.tonal_swap(f, tone_swap)

        if mono_hue > 0:
            f = self.monochrome_hue(f, mono_hue)

        if rgb_split > 0:
            b, g, r = cv2.split(f)
            f = cv2.merge([
                self.shift_channel_safe(b, rgb_split),
                g,
                self.shift_channel_safe(r, -rgb_split)
            ])

        for _ in range(line_glitch):
            y = np.random.randint(0, self.h)
            shift = np.random.randint(-20, 20)
            f[y:y+1] = self.shift_image_safe(f[y:y+1], shift, 0)

        self.prev = f.copy()
        return f
    
    def get_state(self):
        return dict(self.state)

    def run(self, image_path: str):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(image_path)

        self.h, self.w = self.original.shape[:2]

        self.app = QApplication.instance() or QApplication(sys.argv)
        self.widget = LiveFXWidget(self)
        self.widget.resize(900, 700)
        self.widget.show()
        self.app.exec()


class LiveFXWidget(QWidget):
    def __init__(self, fx: LiveFXTester):
        super().__init__()
        self.fx = fx

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

        layout.addLayout(self._slider("brightness", 0, 200))
        layout.addLayout(self._slider("contrast", 0, 300))
        layout.addLayout(self._slider("saturation", 0, 300))
        layout.addLayout(self._slider("tone_swap", 0, 100))
        layout.addLayout(self._slider("mono_hue", 0, 179))
        layout.addLayout(self._slider("rgb_split", 0, 30))
        layout.addLayout(self._slider("line_glitch", 0, 50))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    def _slider(self, key, min_v, max_v):
        name_label = QLabel(key)
        name_label.setFixedWidth(100)

        value_label = QLabel(str(self.fx.state[key]))
        value_label.setFixedWidth(40)
        value_label.setAlignment(Qt.AlignRight)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(self.fx.state[key])

        def on_change(v):
            self.fx.state[key] = v
            value_label.setText(str(v))

        slider.valueChanged.connect(on_change)

        layout = QHBoxLayout()
        layout.addWidget(name_label)
        layout.addWidget(slider)
        layout.addWidget(value_label)

        return layout

    def update_frame(self):
        frame = self.fx.apply_fx(self.fx.original.copy())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

