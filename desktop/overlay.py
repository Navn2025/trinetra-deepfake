"""
Transparent, always-on-top, click-through overlay window that tracks a
target application window's on-screen position/size and draws a
REAL/FAKE/UNCERTAIN badge over each detected face -- the desktop
equivalent of extension/overlay.js's per-<video>-tile badge, except here
there's no DOM to attach to, so this draws directly onto a borderless Qt
window positioned on top of the target window.

Only shown while the target window is the foreground window, so it
doesn't float over unrelated apps when you switch away.
"""
import win32gui
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

POSITION_SYNC_MS = 150


class OverlayWindow(QWidget):
    def __init__(self, hwnd: int):
        super().__init__()
        self._hwnd = hwnd
        self._faces: list[dict] = []
        self._font = QFont("Segoe UI", 11, QFont.Weight.DemiBold)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_to_target_window)
        self._sync_timer.start(POSITION_SYNC_MS)

    def set_faces(self, faces: list[dict]) -> None:
        self._faces = faces
        self.update()  # trigger repaint

    def _sync_to_target_window(self) -> None:
        if not win32gui.IsWindow(self._hwnd):
            print("[Overlay] Target window closed, stopping.")
            self._sync_timer.stop()
            self.close()
            return

        if win32gui.IsIconic(self._hwnd) or win32gui.GetForegroundWindow() != self._hwnd:
            self.hide()
            return

        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            self.hide()
            return

        self.setGeometry(left, top, width, height)
        if not self.isVisible():
            self.show()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)

        w, h = self.width(), self.height()

        for face in self._faces:
            x, y, fw, fh = face["bbox_fraction"]
            px, py, pw, ph = int(x * w), int(y * h), int(fw * w), int(fh * h)

            prob = face.get("smoothed_fake_probability")
            if prob is None:
                color, label = QColor(66, 133, 244, 220), "..."
            elif prob >= 0.65:
                color = QColor(217, 48, 37, 235)
                label = f"⚠ POSSIBLE DEEPFAKE {round(prob * 100)}%"
            elif prob <= 0.35:
                color = QColor(52, 168, 83, 230)
                label = f"✓ REAL {round((1 - prob) * 100)}%"
            else:
                color = QColor(178, 128, 0, 230)
                label = f"? UNCERTAIN {round(prob * 100)}%"

            painter.setPen(QPen(color, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(px, py, pw, ph)

            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label) + 12
            text_h = metrics.height() + 6
            badge_y = max(0, py - text_h - 4)

            painter.fillRect(px, badge_y, text_w, text_h, color)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(px + 6, badge_y + metrics.ascent() + 3, label)
