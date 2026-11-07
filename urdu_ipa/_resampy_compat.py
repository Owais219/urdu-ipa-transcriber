"""Minimal stand-in for the `resampy` package, used by Allosaurus only to resample audio.

resampy depends on numba/llvmlite, which slow down start-up (JIT compilation) and have no
recent builds for Intel Macs. This module provides `resample` with SciPy's polyphase
filter (Kaiser window), which is accurate for the integer-ratio conversions used here
(16 kHz -> 8 kHz).
"""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

__version__ = "0+urdu_ipa-compat"


def resample(x, sr_orig, sr_new, axis=-1, filter=None, **kwargs):  # noqa: A002 (mirror resampy API)
    sr_orig, sr_new = int(sr_orig), int(sr_new)
    if sr_orig == sr_new:
        return np.asarray(x, dtype=np.float64)
    g = gcd(sr_orig, sr_new)
    return resample_poly(np.asarray(x, dtype=np.float64), sr_new // g, sr_orig // g, axis=axis,
                         window=("kaiser", 8.6))


def install() -> None:
    """Make `import resampy` return this module (only if Allosaurus has not been imported yet)."""
    import sys
    if "allosaurus" not in sys.modules:
        sys.modules["resampy"] = sys.modules[__name__]
