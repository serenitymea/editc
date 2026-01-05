import cv2
import numpy as np

class GlitchVideoFX:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(input_path)

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.prev = None

        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.w, self.h)
        )

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
    def shift_channel_safe(ch, shift):
        out = np.zeros_like(ch)
        if shift > 0:
            out[:, shift:] = ch[:, :-shift]
        elif shift < 0:
            out[:, :shift] = ch[:, -shift:]
        else:
            out[:] = ch
        return out

    @staticmethod
    def tonal_swap(f, amount):
        if amount <= 0:
            return f
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[..., 2]
        hsv[..., 2] = v * (1 - amount) + (255 - v) * amount
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def monochrome_hue(f, hue_value):
        if hue_value <= 0:
            return f
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = hue_value
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_fx(self, f, brightness=0.0, contrast=1.0, saturation=1.0,
                 tone_swap=0.0, mono_hue=0, rgb_split=0, line_glitch=2):

        f = f.astype(np.float32)
        f = f * contrast + brightness * 255
        f = np.clip(f, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

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

    def process_video(self, brightness=100, contrast=100, saturation=100,
                    tone_swap=0, mono_hue=0, rgb_split=0, line_glitch=2):

        brightness = (brightness - 100) / 100
        contrast   = contrast / 100
        saturation = saturation / 100
        tone_swap  = tone_swap / 100

        print("Processing video...")
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_id += 1
            fx_frame = self.apply_fx(frame, brightness, contrast, saturation,
                                     tone_swap, mono_hue, rgb_split, line_glitch)
            self.out.write(fx_frame)

            if frame_id % 30 == 0:
                print(f"processed {frame_id} frames...")

        self.cap.release()
        self.out.release()
        print(f"saved {self.output_path}")
