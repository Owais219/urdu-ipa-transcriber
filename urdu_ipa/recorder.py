"""Microphone recording with sounddevice (PortAudio).

* One stream at a time; start() while recording raises RecorderError.
* The PortAudio callback only copies samples and updates a few numbers;
  the UI polls `poll()` from a Qt timer. No threads or loops are created here.
* Records at the device's native rate (many devices reject 16 kHz) and leaves
  resampling to the pipeline.
* The stream is always stopped and closed in stop()/abort(), releasing the device.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import RecordingConfig


class RecorderError(RuntimeError):
    pass


@dataclass
class RecorderStatus:
    recording: bool
    elapsed: float
    level_db: float
    speech_started: bool
    should_stop: str | None       # None | "silence" | "max_duration" | "device_lost"
    overflowed: bool


def _sd():
    try:
        import sounddevice as sd
        return sd
    except OSError as e:          # PortAudio library missing
        raise RecorderError(f"Audio system unavailable: {e}") from e


def list_input_devices() -> list[tuple[int, str]]:
    sd = _sd()
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]


class Recorder:
    def __init__(self, cfg: RecordingConfig):
        self.cfg = cfg
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.samplerate = 0
        self._reset_meters()

    def _reset_meters(self):
        self._t0 = 0.0
        self._n = 0
        self._level_db = -120.0
        self._floor_db = None
        self._calib: list[float] = []
        self._speech_started = False
        self._last_voice = 0.0
        self._overflow = False
        self._history: list[float] = []

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            raise RecorderError("A recording is already in progress.")
        sd = _sd()
        try:
            dev = self.cfg.input_device
            info = sd.query_devices(dev, "input")
            if info is None or info.get("max_input_channels", 0) < 1:
                raise RecorderError("No microphone found.")
            self.samplerate = int(info["default_samplerate"])
            self._chunks = []
            self._reset_meters()
            self._stream = sd.InputStream(device=dev, channels=1, samplerate=self.samplerate, dtype="float32",
                                          blocksize=int(self.samplerate * 0.02), callback=self._callback)
            self._t0 = time.monotonic()
            self._stream.start()
        except RecorderError:
            self._close()
            raise
        except Exception as e:     # PortAudioError: permission denied, device busy, invalid device ...
            self._close()
            raise RecorderError(f"Could not open the microphone: {e}. "
                                "Check that a microphone is connected and that this application "
                                "is allowed to use it (system privacy settings).") from e

    def _callback(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            self._overflow = True
        block = indata[:, 0].copy()
        with self._lock:
            self._chunks.append(block)
            self._n += frames
        rms = float(np.sqrt(np.mean(block ** 2) + 1e-12))
        db = 20 * np.log10(rms + 1e-12)
        self._level_db = db
        self._history.append(db)
        if len(self._history) > 300:
            del self._history[:100]
        elapsed = self._n / self.samplerate
        if elapsed < self.cfg.noise_calibration:
            self._calib.append(db)
            return
        if self._floor_db is None:
            self._floor_db = float(np.median(self._calib)) if self._calib else -60.0
        thr = max(self._floor_db + self.cfg.speech_db_above_floor, self.cfg.min_speech_dbfs)
        if db > thr:
            self._speech_started = True
            self._last_voice = elapsed

    def poll(self) -> RecorderStatus:
        if self._stream is None:
            return RecorderStatus(False, 0.0, -120.0, False, None, False)
        elapsed = self._n / self.samplerate if self.samplerate else 0.0
        stop = None
        if not self._stream.active:
            stop = "device_lost"
        elif elapsed >= self.cfg.max_duration:
            stop = "max_duration"
        elif (self.cfg.auto_stop and self._speech_started
              and elapsed - self._last_voice >= self.cfg.trailing_silence):
            stop = "silence"
        return RecorderStatus(True, elapsed, self._level_db, self._speech_started, stop, self._overflow)

    def level_history(self) -> list[float]:
        return list(self._history[-100:])

    def stop(self) -> tuple[np.ndarray, int]:
        """Stop and return (samples, samplerate). Safe to call once per start()."""
        if self._stream is None:
            raise RecorderError("Not recording.")
        self._close()
        with self._lock:
            data = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
            self._chunks = []
        return data, self.samplerate

    def abort(self) -> None:
        self._close()
        with self._lock:
            self._chunks = []

    def _close(self):
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass


class Player:
    """Non-blocking playback of the last recording."""

    def play(self, samples: np.ndarray, sr: int) -> None:
        sd = _sd()
        sd.stop()
        sd.play(samples, sr, blocking=False)

    def stop(self) -> None:
        try:
            _sd().stop()
        except RecorderError:
            pass
