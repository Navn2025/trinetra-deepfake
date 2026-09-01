"""
Captures the default output device's audio via WASAPI loopback
(pyaudiowpatch) -- the desktop equivalent of the extension's tab audio
capture (extension/offscreen.js). Same limitation accepted there applies
here too: this captures everything currently playing through the default
speakers, not audio isolated to the monitored app/window specifically --
there's no cheap way to do per-process capture without much deeper
platform integration (see voice-integrity/src/server.py's docstring and
the project plan this was scoped against).

Starts automatically once the app begins monitoring a window (see main.py)
-- unlike face detection, no click/arming needed, since this isn't tied to
a specific visible face.

Every VOICE_CHECK_INTERVAL_SECONDS, sends the most recently captured
VOICE_CHUNK_SECONDS of audio to the voice service (voice-integrity/src/server.py,
port 8001) and emits the result.
"""
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
from PySide6.QtCore import QThread, Signal

from api_client import check_voice

VOICE_CHUNK_SECONDS = 20
VOICE_CHECK_INTERVAL_SECONDS = 30
CHUNK_FRAMES = 4096


def _find_default_loopback_device(p: "pyaudio.PyAudio") -> dict:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if default_speakers.get("isLoopbackDevice", False):
        return default_speakers

    for loopback in p.get_loopback_device_info_generator():
        if default_speakers["name"] in loopback["name"]:
            return loopback

    raise RuntimeError(f"No loopback device found matching default output {default_speakers['name']!r}")


def _close_stream_safely(stream) -> None:
    try:
        stream.stop_stream()
    except (OSError, AttributeError):
        pass
    try:
        stream.close()
    except (OSError, AttributeError):
        pass


class VoiceWorker(QThread):
    # Same shape as the /voice/check response, or {"error": str} if the
    # request itself failed (server down, network error, etc.).
    voice_result = Signal(dict)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        p = pyaudio.PyAudio()
        try:
            device = _find_default_loopback_device(p)
        except Exception as exc:  # noqa: BLE001 -- report and exit the thread cleanly either way
            print(f"[Audio] Could not find a loopback device: {exc}")
            p.terminate()
            return

        channels = device["maxInputChannels"]
        sample_rate = int(device["defaultSampleRate"])

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=CHUNK_FRAMES,
        )

        print(f"[Audio] Capturing system audio via {device['name']!r} ({sample_rate}Hz, {channels}ch)")

        bytes_per_chunk = CHUNK_FRAMES * channels * 2
        max_chunks = max(1, (sample_rate * VOICE_CHUNK_SECONDS * channels * 2) // bytes_per_chunk)
        audio_chunks: deque[bytes] = deque(maxlen=max_chunks)
        audio_lock = threading.Lock()
        capture_ready = threading.Event()
        capture_stopped = threading.Event()

        def capture_audio() -> None:
            try:
                while self._running:
                    chunk = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                    with audio_lock:
                        audio_chunks.append(chunk)
                        if len(audio_chunks) == max_chunks:
                            capture_ready.set()
            except Exception as exc:  # noqa: BLE001 -- stop the worker cleanly on device errors
                if self._running:
                    print(f"[Audio] Capture failed: {exc}")
            finally:
                capture_stopped.set()

        capture_thread = threading.Thread(target=capture_audio, name="voice-capture", daemon=True)
        capture_thread.start()

        try:
            while self._running and not capture_stopped.is_set():
                if not capture_ready.wait(timeout=0.5):
                    continue
                self._check_latest_audio(audio_chunks, audio_lock, channels, sample_rate)
        finally:
            self._running = False
            _close_stream_safely(stream)
            capture_thread.join(timeout=2)
            p.terminate()

    def _check_latest_audio(
        self,
        audio_chunks: deque[bytes],
        audio_lock: threading.Lock,
        channels: int,
        sample_rate: int,
    ) -> None:
        cycle_start = time.monotonic()
        with audio_lock:
            collected = b"".join(audio_chunks)

        audio = np.frombuffer(collected, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

        with tempfile.TemporaryDirectory(prefix="voice_capture_") as tmp_dir:
            wav_path = Path(tmp_dir) / "chunk.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            result = check_voice(wav_path)

        self.voice_result.emit(result if result is not None else {"error": "voice check request failed"})

        # Sleep out whatever's left of the 30s cycle, in short steps so
        # stop() takes effect promptly instead of after a long single sleep.
        remaining = VOICE_CHECK_INTERVAL_SECONDS - (time.monotonic() - cycle_start)
        while remaining > 0 and self._running:
            step = min(0.5, remaining)
            time.sleep(step)
            remaining -= step
