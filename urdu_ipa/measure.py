"""Signal-level phonetic measurements.

These measurements give acoustic evidence for contrasts the phone model
cannot represent (e.g. Allosaurus has no [ʈʰ] or [t͡ʃʰ]).

* VOT (voice onset time) for voiceless stops and affricates: time from the
  release burst to the onset of periodic voicing. Long-lag VOT indicates
  aspiration. Default category boundaries are taken from the general range
  reported for Hindi-Urdu (unaspirated short-lag, aspirated long-lag; cf.
  Lisker & Abramson 1964) and are NOT calibrated for this tool. Calibrate
  them with `evaluation.calibrate_vot` on verified recordings before relying
  on them.
* Closure duration (silent interval before a release) for singleton vs
  geminate stops, word-medially only.

Every measurement returns None when the expected acoustic landmarks are not
found; callers must treat None as "no evidence", never as a negative value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

HOP = 0.001
WIN = 0.005


@dataclass
class VOTThresholds:
    # (max unaspirated, min aspirated) in seconds, by place class
    stop_front: tuple[float, float] = (0.035, 0.055)      # bilabial, dental, retroflex
    stop_back: tuple[float, float] = (0.045, 0.065)       # velar, uvular
    affricate: tuple[float, float] = (0.075, 0.110)


@dataclass
class GeminationThresholds:
    max_singleton: float = 0.105
    min_geminate: float = 0.135


@dataclass
class VOTResult:
    burst_time: float
    voicing_onset: float
    vot: float
    category: str          # "unaspirated" | "aspirated" | "ambiguous"


def _frames(x, sr, win=WIN, hop=HOP):
    w, h = max(1, int(win * sr)), max(1, int(hop * sr))
    if x.size < w:
        return np.zeros((0, w), dtype=np.float32), hop
    n = 1 + (x.size - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx], hop


def _db(frames):
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def _periodicity(x, sr, t0, t1, fmin=70.0, fmax=400.0, win=0.025, hop=0.002):
    """Normalised autocorrelation peak (0..1) in short windows between t0 and t1."""
    w = int(win * sr)
    out_t, out_p = [], []
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    t = t0
    while t + win <= t1:
        a = int(t * sr)
        seg = x[a:a + w]
        seg = seg - seg.mean()
        e = float(np.dot(seg, seg))
        if e <= 1e-9:
            p = 0.0
        else:
            ac = np.correlate(seg, seg, mode="full")[w - 1:]
            p = float(np.max(ac[lag_min:min(lag_max, w - 1)]) / e) if lag_max < w else 0.0
        out_t.append(t + win / 2)
        out_p.append(p)
        t += hop
    return np.array(out_t), np.array(out_p)


def _is_transient(seg, hp, sr, t, dur=0.006):
    """A release burst is aperiodic and rich in high frequencies; a vowel onset is not."""
    a, b = int(t * sr), int((t + dur) * sr)
    full, high = seg[a:b], hp[a:b]
    if full.size < 16:
        return False
    ratio_db = 10 * np.log10((np.mean(high ** 2) + 1e-12) / (np.mean(full ** 2) + 1e-12))
    zcr = np.mean(np.abs(np.diff(np.sign(full))) > 0) * sr
    return ratio_db > -12 and zcr > 1500


def measure_vot(x: np.ndarray, sr: int, centre: float, kind: str,
                thresholds: VOTThresholds | None = None, search: float = 0.12) -> VOTResult | None:
    """Measure VOT of the voiceless stop/affricate whose model-estimated position is `centre`."""
    th = thresholds or VOTThresholds()
    a, b = max(0.0, centre - search), min(x.size / sr, centre + search)
    if b - a < 0.06:
        return None
    seg = x[int(a * sr):int(b * sr)]
    sos = butter(4, 2000, btype="highpass", fs=sr, output="sos")
    hp = sosfiltfilt(sos, seg)
    fr, hop = _frames(hp, sr)
    fr_all, _ = _frames(seg, sr)
    if fr.shape[0] < 20:
        return None
    hp_db = _db(fr)
    all_db = _db(fr_all)
    # burst: largest rise in high-frequency energy following a quiet interval (closure or silence)
    rise = hp_db[5:] - hp_db[:-5]
    order = np.argsort(rise)[::-1]
    burst_i = None
    floor = np.percentile(all_db, 10)
    for i in order[:15]:
        j = i + 5
        pre = all_db[max(0, j - 25):j - 3]
        if rise[i] < 12:
            break
        if pre.size and np.median(pre) < floor + 10 and _is_transient(seg, hp, sr, j * hop):
            burst_i = j
            break
    if burst_i is None:
        return None
    burst_t = a + burst_i * hop
    # voicing onset: first point after the burst where periodicity is high and sustained for 20 ms
    tt, pp = _periodicity(x, sr, burst_t + 0.003, min(b + 0.05, x.size / sr))
    if tt.size == 0:
        return None
    good = pp > 0.5
    need = max(1, int(0.020 / 0.002))
    onset = None
    for i in range(0, len(good) - need):
        if good[i:i + need].all():
            onset = tt[i] - 0.0125  # window centre -> window start (half of 25 ms)
            break
    if onset is None:
        return None
    vot = max(0.0, onset - burst_t)
    if kind == "affricate":
        lo, hi = th.affricate
    elif kind == "back":
        lo, hi = th.stop_back
    else:
        lo, hi = th.stop_front
    cat = "unaspirated" if vot <= lo else ("aspirated" if vot >= hi else "ambiguous")
    return VOTResult(burst_t, onset, vot, cat)


def measure_closure(x: np.ndarray, sr: int, burst_time: float, max_look: float = 0.35) -> float | None:
    """Duration of the low-energy interval ending at `burst_time` (medial stops only)."""
    a = max(0.0, burst_time - max_look)
    seg = x[int(a * sr):int(burst_time * sr)]
    fr, hop = _frames(seg, sr, win=0.010, hop=0.002)
    if fr.shape[0] < 10:
        return None
    d = _db(fr)
    loud = np.percentile(_db(_frames(x, sr, 0.010, 0.002)[0]), 90)
    quiet = d < loud - 25
    # walk back from the burst while quiet
    i = len(quiet) - 1
    while i >= 0 and quiet[i]:
        i -= 1
    if i < 0:
        return None          # silence all the way back: utterance-initial, closure not measurable
    dur = (len(quiet) - 1 - i) * hop
    return dur if dur > 0.01 else None


def gemination_category(closure: float | None, th: GeminationThresholds | None = None) -> str | None:
    if closure is None:
        return None
    th = th or GeminationThresholds()
    if closure <= th.max_singleton:
        return "singleton"
    if closure >= th.min_geminate:
        return "geminate"
    return "ambiguous"


@dataclass
class FinalBreathThresholds:
    min_present: float = 0.050      # s of aperiodic noise after voicing offset -> final [h]
    max_absent: float = 0.020       # s or less -> no final [h]
    db_above_floor: float = 8.0     # noise must exceed the background by this much
    voiced_db_above_floor: float = 15.0


@dataclass
class FinalBreathResult:
    voicing_offset: float
    breath_duration: float
    category: str                   # "present" | "absent" | "ambiguous"


def measure_final_breath(x: np.ndarray, sr: int, th: FinalBreathThresholds | None = None) -> FinalBreathResult | None:
    """Duration of voiceless (aperiodic) noise after the last voiced frame of the word.

    Used for word-final ہ: a breathy/aspirated release ([ɑh]) vs a plain vowel end ([ɑ]).
    Returns None when the end of the word cannot be measured (no voicing found, or
    the recording stops before the release is over).
    """
    th = th or FinalBreathThresholds()
    dur = x.size / sr
    if dur < 0.15:
        return None
    win, hop = 0.025, 0.005
    tt, pp = _periodicity(x, sr, 0.0, dur, win=win, hop=hop)
    if tt.size < 10:
        return None
    fr, _ = _frames(x, sr, win=win, hop=hop)
    n = min(len(tt), fr.shape[0])
    tt, pp, en = tt[:n], pp[:n], _db(fr[:n])
    floor = float(np.percentile(en, 10))
    voiced = (pp > 0.5) & (en > floor + th.voiced_db_above_floor)
    if not voiced.any():
        return None
    last = int(np.nonzero(voiced)[0][-1])
    offset = float(tt[last] + win / 2)
    i = last + 1
    while i < n and en[i] > floor + th.db_above_floor and pp[i] < 0.5:
        i += 1
    if i >= n:                       # noise runs to the end of the recording: release may be cut off
        return None
    breath = max(0.0, (i - last - 1) * hop)
    cat = "present" if breath >= th.min_present else ("absent" if breath <= th.max_absent else "ambiguous")
    return FinalBreathResult(offset, breath, cat)
