"""Audio preprocessing, voice-activity detection, single-word segmentation and QC.

Design decisions (phonetically motivated):
* No spectral noise suppression by default: aspiration noise, frication and
  stop bursts are the cues this tool must preserve. Only DC removal and a
  gentle high-pass (default 60 Hz) are applied.
* The pause threshold for "more than one word" must be longer than the
  closure of an Urdu geminate stop (often 150-250 ms); the default is 350 ms.
  A pause-based check cannot detect two words spoken without a pause, so the
  ASR output is checked as well (see pipeline.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

TARGET_SR = 16000


def to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    if cutoff <= 0 or x.size < 64:
        return x
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def frame_rms_db(x: np.ndarray, sr: int, win: float = 0.025, hop: float = 0.010) -> tuple[np.ndarray, float]:
    w, h = max(1, int(win * sr)), max(1, int(hop * sr))
    if x.size < w:
        return np.array([-120.0]), hop
    n = 1 + (x.size - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12), hop


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AudioReport:
    duration: float = 0.0
    speech_segments: list[Segment] = field(default_factory=list)
    speech_start: float = 0.0
    speech_end: float = 0.0
    peak: float = 0.0
    clipped_fraction: float = 0.0
    noise_floor_db: float = -120.0
    speech_level_db: float = -120.0
    snr_db: float = 0.0
    problems: list[str] = field(default_factory=list)   # codes, see pipeline.MESSAGES
    warnings: list[str] = field(default_factory=list)

    @property
    def n_islands(self) -> int:
        return len(self.speech_segments)


@dataclass
class VADConfig:
    min_speech_db_above_floor: float = 12.0
    absolute_floor_db: float = -55.0
    min_island: float = 0.06
    merge_gap: float = 0.35            # pauses shorter than this stay inside one word
    min_word_duration: float = 0.15
    max_word_duration: float = 2.5
    clip_level: float = 0.985
    max_clipped_fraction: float = 0.002
    min_snr_db: float = 10.0
    warn_snr_db: float = 18.0
    loud_floor_dbfs: float = -35.0


def detect_speech(x: np.ndarray, sr: int, cfg: VADConfig) -> tuple[list[Segment], float, float]:
    """Energy-based VAD with an adaptive noise floor. Returns islands, floor dB, speech dB."""
    db, hop = frame_rms_db(x, sr)
    if db.size < 3:
        return [], float(db.min()), float(db.max())
    floor = float(np.percentile(db, 10))
    thr = max(floor + cfg.min_speech_db_above_floor, cfg.absolute_floor_db)
    active = db > thr
    segs: list[Segment] = []
    start = None
    for i, a in enumerate(active):
        t = i * hop
        if a and start is None:
            start = t
        elif not a and start is not None:
            segs.append(Segment(start, t + 0.025))
            start = None
    if start is not None:
        segs.append(Segment(start, len(active) * hop + 0.025))
    # merge short gaps (stop closures, geminates) then drop tiny islands (clicks)
    merged: list[Segment] = []
    for s in segs:
        if merged and s.start - merged[-1].end < cfg.merge_gap:
            merged[-1].end = s.end
        else:
            merged.append(Segment(s.start, s.end))
    merged = [s for s in merged if s.duration >= cfg.min_island]
    speech_db = float(np.percentile(db[active], 90)) if active.any() else floor
    return merged, floor, speech_db


def analyse(x: np.ndarray, sr: int, cfg: VADConfig) -> AudioReport:
    r = AudioReport(duration=x.size / sr if sr else 0.0)
    if x.size == 0 or r.duration < 0.05:
        r.problems.append("empty")
        return r
    r.peak = float(np.max(np.abs(x)))
    if r.peak < 1e-4:
        r.problems.append("no_speech")
        return r
    r.clipped_fraction = float(np.mean(np.abs(x) >= cfg.clip_level))
    if r.clipped_fraction > cfg.max_clipped_fraction:
        r.problems.append("clipped")
    segs, floor, speech = detect_speech(x, sr, cfg)
    r.speech_segments, r.noise_floor_db, r.speech_level_db = segs, floor, speech
    r.snr_db = speech - floor
    if not segs:
        # nothing rises above the floor: distinguish silence from a loud, noisy background
        r.problems.append("noisy" if floor > cfg.loud_floor_dbfs else "no_speech")
        return r
    r.speech_start, r.speech_end = segs[0].start, segs[-1].end
    if len(segs) > 1:
        r.problems.append("multiple_words")
    total = r.speech_end - r.speech_start
    if total < cfg.min_word_duration:
        r.problems.append("too_short")
    if total > cfg.max_word_duration:
        r.problems.append("too_long")
    if r.snr_db < cfg.min_snr_db:
        r.problems.append("noisy")
    elif r.snr_db < cfg.warn_snr_db:
        r.warnings.append("moderate_noise")
    if segs[0].start < 0.03:
        r.warnings.append("onset_truncated")
    if r.duration - segs[-1].end < 0.03:
        r.warnings.append("offset_truncated")
    return r


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at 16 kHz (no other processing: QC runs on this signal)."""
    return resample(to_mono_float(x), sr, TARGET_SR)


def prepare(x: np.ndarray, highpass_hz: float = 60.0, pad: float = 0.25,
            report: AudioReport | None = None) -> np.ndarray:
    """From 16 kHz mono: DC-free, high-passed, trimmed to speech (+pad), peak-normalised."""
    x = x - float(np.mean(x)) if x.size else x
    x = highpass(x, TARGET_SR, highpass_hz)
    if report is not None and report.speech_segments:
        a = max(0, int((report.speech_start - pad) * TARGET_SR))
        b = min(x.size, int((report.speech_end + pad) * TARGET_SR))
        x = x[a:b]
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = (0.9 / peak) * x
    return x.astype(np.float32)
