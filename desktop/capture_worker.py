"""
Wraps windows_capture.WindowsCapture to continuously capture a target
window and expose only its latest frame to the rest of the app.

Important: Frame.frame_buffer is documented as "a zero-copy view backed by
an owned native mapped frame" -- that native memory isn't guaranteed to
stay valid after on_frame_arrived returns, so the buffer is copied
immediately here rather than stored by reference. minimum_update_interval
caps the native capture rate (we don't need more than ~2fps since face
detection only runs every couple of seconds anyway), which also keeps the
per-frame copy cost low.
"""
import threading

from windows_capture import InternalCaptureControl, WindowsCapture

MINIMUM_UPDATE_INTERVAL_MS = 500


class LatestFrameHolder:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None  # (bgra_ndarray, width, height) or None

    def set(self, bgra, width, height) -> None:
        with self._lock:
            self._frame = (bgra, width, height)

    def get(self):
        with self._lock:
            return self._frame


def start_capture(hwnd: int, holder: LatestFrameHolder):
    """Starts capturing `hwnd` on a dedicated thread. Returns a CaptureControl
    with .stop() to end it."""
    capture = WindowsCapture(
        window_hwnd=hwnd,
        cursor_capture=False,
        minimum_update_interval=MINIMUM_UPDATE_INTERVAL_MS,
    )

    @capture.event
    def on_frame_arrived(frame, capture_control: InternalCaptureControl):
        holder.set(frame.frame_buffer.copy(), frame.width, frame.height)

    @capture.event
    def on_closed():
        print("[Capture] Target window capture session ended.")

    return capture.start_free_threaded()
