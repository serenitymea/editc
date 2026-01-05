import cv2
import numpy as np


class LiveFXTester:
    def __init__(self, image_path: str):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(image_path)

        self.h, self.w = self.original.shape[:2]
        self.prev = None

        cv2.namedWindow("FX", cv2.WINDOW_NORMAL)
        self._create_controls()

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
    def tonal_swap(frame, amount):
        if amount <= 0:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[..., 2]
        hsv[..., 2] = v * (1 - amount) + (255 - v) * amount
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def monochrome_hue(frame, hue_value):
        if hue_value <= 0:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = hue_value
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _create_controls(self):
        def n(x): pass

        cv2.createTrackbar("Brightness", "FX", 100, 200, n)
        cv2.createTrackbar("Contrast", "FX", 100, 300, n)
        cv2.createTrackbar("Saturation", "FX", 100, 300, n)

        cv2.createTrackbar("Tone Swap", "FX", 0, 100, n)
        cv2.createTrackbar("Mono Hue", "FX", 0, 179, n)

        cv2.createTrackbar("RGB Split", "FX", 0, 30, n)
        cv2.createTrackbar("Line Glitch", "FX", 0, 50, n)

    def apply_fx(self, f):
        brightness = (cv2.getTrackbarPos("Brightness", "FX") - 100) / 100
        contrast = cv2.getTrackbarPos("Contrast", "FX") / 100
        saturation = cv2.getTrackbarPos("Saturation", "FX") / 100

        tone_swap = cv2.getTrackbarPos("Tone Swap", "FX") / 100
        mono_hue = cv2.getTrackbarPos("Mono Hue", "FX")

        rgb_split = cv2.getTrackbarPos("RGB Split", "FX")
        line_glitch = cv2.getTrackbarPos("Line Glitch", "FX")

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

    def run(self):
        print("E / S")

        while True:
            out = self.apply_fx(self.original.copy())
            cv2.imshow("FX", out)

            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord("s"):
                cv2.imwrite("snapshot.png", out)
                print("saved")

        cv2.destroyAllWindows()