"""Camera Management - Windows compatible with robust demo mode"""
import cv2
import numpy as np
import os
import time


class CameraManager:
    def __init__(self):
        self._cap = None

    def get_camera(self):
        if self._cap is None:
            self._cap = self._initialize_camera()
        return self._cap

    def _initialize_camera(self):
        # Try RTSP first
        rtsp_url = os.getenv("CAMERA_RTSP_URL")
        if rtsp_url:
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.any():
                    print(f"✅ RTSP camera: {rtsp_url}")
                    return cap
            cap.release()

        # Try USB webcams (Windows needs CAP_DSHOW backend)
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for idx in range(3):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Read a few frames to warm up
                        for _ in range(5):
                            ret, frame = cap.read()
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0 and frame.any():
                            print(f"✅ Webcam found: index={idx}, backend={backend}")
                            return cap
                    cap.release()
                except Exception:
                    pass

        print("⚠️  No camera found — Demo mode active")
        return DemoCapture()

    def release(self):
        if self._cap:
            self._cap.release()


class DemoCapture:
    """Generates realistic synthetic textile frames for demo/testing"""

    def __init__(self):
        self._count = 0
        self._phase = 0
        self._phase_timer = 0

    def read(self):
        self._count += 1
        # Switch texture every 5 seconds (~150 frames at 30fps)
        if self._count % 150 == 0:
            self._phase = (self._phase + 1) % 5
        frame = self._generate(self._phase)
        return True, frame

    def isOpened(self):
        return True

    def release(self):
        pass

    def _generate(self, phase: int) -> np.ndarray:
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        if phase == 0:
            frame = self._cotton(h, w)
        elif phase == 1:
            frame = self._denim(h, w)
        elif phase == 2:
            frame = self._striped(h, w)
        elif phase == 3:
            frame = self._dotted(h, w)
        else:
            frame = self._wool(h, w)

        # Scanning line animation
        scan_y = int((self._count * 3) % h)
        cv2.line(frame, (50, scan_y), (w - 50, scan_y), (0, 255, 150), 2)

        # Corner brackets (targeting area)
        c = (0, 200, 255)
        cv2.line(frame, (40, 40), (100, 40), c, 2)
        cv2.line(frame, (40, 40), (40, 100), c, 2)
        cv2.line(frame, (w-100, 40), (w-40, 40), c, 2)
        cv2.line(frame, (w-40, 40), (w-40, 100), c, 2)
        cv2.line(frame, (40, h-100), (40, h-40), c, 2)
        cv2.line(frame, (40, h-40), (100, h-40), c, 2)
        cv2.line(frame, (w-40, h-100), (w-40, h-40), c, 2)
        cv2.line(frame, (w-100, h-40), (w-40, h-40), c, 2)

        # Label
        labels = ["COTTON FABRIC","DENIM","STRIPED CLOTH","DOTTED FABRIC","WOOL BLEND"]
        cv2.putText(frame, f"DEMO: {labels[phase]}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)

        return frame

    def _cotton(self, h, w):
        base = np.full((h, w, 3), (210, 200, 185), dtype=np.uint8)
        # Woven grid
        for i in range(0, h, 5):
            cv2.line(base, (0, i), (w, i), (190, 178, 162), 1)
        for j in range(0, w, 5):
            cv2.line(base, (j, 0), (j, h), (190, 178, 162), 1)
        # Add slight noise
        noise = np.random.randint(-8, 8, base.shape, dtype=np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return base

    def _denim(self, h, w):
        base = np.full((h, w, 3), (65, 105, 155), dtype=np.uint8)
        # Diagonal twill weave
        for i in range(-h, w + h, 4):
            cv2.line(base, (i, 0), (i + h, h), (75, 115, 165), 1)
        noise = np.random.randint(-6, 6, base.shape, dtype=np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return base

    def _striped(self, h, w):
        base = np.zeros((h, w, 3), dtype=np.uint8)
        colors = [(220,60,60),(255,255,255),(60,100,200),(255,215,0),(60,180,60)]
        sw = 40
        for x in range(0, w, sw):
            c = colors[(x // sw) % len(colors)]
            base[:, x:x+sw] = c
        # Slight fade/warp
        for x in range(0, w, sw):
            if (x // sw) % 2 == 0:
                alpha = np.linspace(0.9, 1.0, sw)
                for xi, a in enumerate(alpha):
                    if x+xi < w:
                        base[:, x+xi] = (base[:, x+xi] * a).astype(np.uint8)
        return base

    def _dotted(self, h, w):
        base = np.full((h, w, 3), (245, 240, 250), dtype=np.uint8)
        for i in range(12, h, 24):
            for j in range(12, w, 24):
                cv2.circle(base, (j, i), 5, (160, 100, 200), -1)
        return base

    def _wool(self, h, w):
        base = np.full((h, w, 3), (180, 160, 140), dtype=np.uint8)
        for i in range(0, h, 8):
            pts = []
            for x in range(0, w, 4):
                y = i + int(3 * np.sin(x * 0.2))
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(base, [pts], False, (160, 140, 120), 1)
        noise = np.random.randint(-10, 10, base.shape, dtype=np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return base
