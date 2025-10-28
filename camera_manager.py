# camera_manager.py
import numpy as np
from picamera2 import Picamera2

class CameraManager:
    """
    Picamera2.capture_array("main") daje RGB888.
    U nas zdarzało się, że kanały wyglądały jak BGR (mocno niebieska poświata).
    Żeby to obejść:
      - całą logikę wewnętrzną prowadzimy jako BGR (traktujemy bufor jak BGR),
      - tuż przed wyświetleniem konwertujemy cvtColor(BGR->RGB).
    """

    def __init__(self, width, height, rotate_dir):
        import cv2  # tylko po to, żeby mieć pewność że CV jest obecne przed startem
        _ = cv2  # ucisz linter
        self.rotate_dir = rotate_dir
        self.picam = Picamera2()
        cfg = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam.configure(cfg)
        self.picam.start()

    def get_frame_bgr(self):
        frame = self.picam.capture_array("main")  # numpy (H,W,3)
        # obrót numpy.rot90:
        if self.rotate_dir == "cw":       # 90° w prawo
            frame = np.rot90(frame, 3)
        elif self.rotate_dir == "ccw":    # 90° w lewo
            frame = np.rot90(frame, 1)
        elif self.rotate_dir == "180":
            frame = np.rot90(frame, 2)
        # traktujemy frame jako BGR od tej chwili
        return frame

    def stop(self):
        try:
            self.picam.stop()
        except Exception:
            pass
