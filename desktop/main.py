"""
Trinetra -- Desktop

Same detection pipeline as the browser extension (extension/), but for
native apps a browser extension can't reach (WhatsApp Desktop, the Zoom
app, Teams desktop, etc.): captures a chosen window via the Windows
Graphics Capture API instead of reading a <video> element, everything
else -- face detection, cropping, calling the backend, showing a
REAL/FAKE/UNCERTAIN badge -- works the same way.

Requires the same backend server as the extension running at
http://127.0.0.1:8000 (see ../backend/README or just run:
  cd ../backend && .\\venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
).

Usage:
  .\\venv\\Scripts\\python.exe main.py
Then pick the window to monitor from the printed list.
"""
import ctypes
import sys
import time

import win32gui
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

from api_client import predict_face
from capture_worker import LatestFrameHolder, start_capture
from face_pipeline import create_face_detector, detect_faces
from overlay import OverlayWindow
from tracker import FaceTracker
from window_picker import pick_window

DETECTION_INTERVAL_SECONDS = 1.0


class DetectionWorker(QThread):
    faces_updated = Signal(list)

    def __init__(self, frame_holder: LatestFrameHolder, platform: str, hwnd: int):
        super().__init__()
        self._frame_holder = frame_holder
        self._platform = platform
        self._hwnd = hwnd
        self._running = True
        self._last_logged_rect: tuple[int, int] | None = None

    def stop(self) -> None:
        self._running = False

    def _check_rect_mismatch(self, frame_w: int, frame_h: int) -> None:
        # Diagnostic: bbox_fraction (face_pipeline.py) is normalized against
        # this captured frame's pixel size, but the overlay (overlay.py)
        # positions/scales the drawn box using win32gui.GetWindowRect's size
        # instead. Those two are assumed to describe the same rectangle --
        # if the capture covers a different region (e.g. client area only
        # vs the full window with title bar/borders) or the window was
        # resized since the frame was captured, the box will be scaled or
        # offset wrong even though the math in each file is individually
        # correct. Logs only when the two disagree (and only on change), so
        # a misaligned-box report can be matched against actual numbers.
        if not win32gui.IsWindow(self._hwnd):
            return
        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        rect_w, rect_h = right - left, bottom - top
        if (frame_w, frame_h) == (rect_w, rect_h):
            self._last_logged_rect = None  # back in sync -- reset so a future mismatch logs again
            return
        if self._last_logged_rect == (rect_w, rect_h):
            return  # already reported this exact mismatch, don't spam every cycle
        self._last_logged_rect = (rect_w, rect_h)
        print(
            f"[Detect] MISMATCH -- captured frame: {frame_w}x{frame_h} px  |  "
            f"GetWindowRect: {rect_w}x{rect_h} px (overlay scales using this) "
            f"-- box will be misaligned until these match"
        )

    def run(self) -> None:
        # Created here (not in __init__) so the detector and tracker are
        # only ever touched from this one thread.
        detector = create_face_detector()
        tracker = FaceTracker()

        while self._running:
            frame = self._frame_holder.get()
            if frame is not None:
                bgra, _width, _height = frame
                self._check_rect_mismatch(bgra.shape[1], bgra.shape[0])
                faces = detect_faces(detector, bgra)

                for face in faces:
                    prediction = predict_face(face["crop"], platform=self._platform)
                    # None here means "no numeric signal this cycle" -- either
                    # the API call itself failed, or it succeeded but the
                    # backend rejected the crop as insufficient quality
                    # (prediction["fake_probability"] is null in that case).
                    # tracker.py skips None rather than folding it into the
                    # smoothed average, so one bad frame doesn't corrupt or
                    # crash the rolling history.
                    face["fake_probability"] = prediction["fake_probability"] if prediction else None
                    del face["crop"]  # no longer needed, don't hold onto it

                smoothed = tracker.update(faces)
                self.faces_updated.emit(smoothed)

            time.sleep(DETECTION_INTERVAL_SECONDS)


def main() -> None:
    # Per-monitor DPI awareness so captured-frame-to-screen-pixel mapping
    # stays correct on scaled displays.
    ctypes.windll.shcore.SetProcessDpiAwareness(2)

    hwnd, title = pick_window()

    app = QApplication(sys.argv)

    overlay = OverlayWindow(hwnd)

    frame_holder = LatestFrameHolder()
    capture_control = start_capture(hwnd, frame_holder)

    worker = DetectionWorker(frame_holder, platform=f"desktop:{title}", hwnd=hwnd)
    worker.faces_updated.connect(overlay.set_faces)
    worker.start()

    def cleanup():
        worker.stop()
        worker.wait(2000)
        capture_control.stop()

    app.aboutToQuit.connect(cleanup)

    print("\nMonitoring started. Close this window or Ctrl+C to stop.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
