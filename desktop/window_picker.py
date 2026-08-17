"""
Lists visible top-level windows so the user can pick one to monitor
(e.g. the Zoom meeting window, or WhatsApp Desktop's call window).

Picking by HWND rather than by title substring, since window_hwnd is what
windows_capture.WindowsCapture actually uses under the hood for capture --
titles on call windows are often dynamic (meeting name, contact name) and
can change mid-call, which would break a title-based capture target.
"""
import win32gui


def _enum_handler(hwnd, results):
    if not win32gui.IsWindowVisible(hwnd):
        return
    title = win32gui.GetWindowText(hwnd)
    if not title.strip():
        return
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    if right - left <= 0 or bottom - top <= 0:
        return
    results.append((hwnd, title))


def list_windows() -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    win32gui.EnumWindows(_enum_handler, results)
    return results


def pick_window() -> tuple[int, str]:
    """Prints a numbered list of open windows and returns the chosen (hwnd, title)."""
    windows = list_windows()
    if not windows:
        raise RuntimeError("No visible windows found.")

    print("\nOpen windows:")
    for i, (hwnd, title) in enumerate(windows):
        print(f"  [{i}] {title}")

    while True:
        choice = input("\nPick a window number to monitor: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(windows):
            hwnd, title = windows[int(choice)]
            print(f"Selected: \"{title}\" (hwnd={hwnd})")
            return hwnd, title
        print("Invalid choice, try again.")
