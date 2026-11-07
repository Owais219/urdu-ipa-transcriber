"""Shared fakes for tests: deterministic acoustic model and synthetic audio."""
import numpy as np

from urdu_ipa.acoustic import PROXIES, AcousticBackend, Emissions, log_softmax
from urdu_ipa.asr import ASRBackend, ASRResult

SR = 16000


class FakeAcoustic(AcousticBackend):
    """Vocabulary = blank + every proxy phone. emissions() returns frames that
    strongly favour `self.target` (a list of model phones), one phone per 3 frames."""
    name = "fake"
    version = "fake/1"

    def __init__(self, target=None, peak=-0.05, floor=-9.0):
        super().__init__()
        phones = sorted({p for v in PROXIES.values() for p in v})
        self._vocab = {"<blk>": 0, **{p: i + 1 for i, p in enumerate(phones)}}
        self.target = target or []
        self.peak, self.floor = peak, floor

    def load(self):
        pass

    def vocab(self):
        return self._vocab

    def emissions(self, audio16k):
        V = len(self._vocab)
        frames = []
        def frame(idx):
            row = np.full(V, self.floor)
            row[idx] = 0.0
            return row
        frames += [frame(0)] * 2
        for p in self.target:
            frames += [frame(self._vocab[p])] * 2 + [frame(0)]
        frames += [frame(0)] * 2
        logits = np.array(frames, dtype=np.float64)
        return Emissions(log_softmax(logits * 1.0), 0.03, 0, self._vocab)


class FakeASR(ASRBackend):
    name = "fake-asr"
    version = "fake-asr/1"

    def __init__(self, text="", conf=0.95):
        self.text, self.conf = text, conf

    def load(self):
        pass

    def transcribe(self, audio16k):
        return ASRResult(self.text, self.conf, self.version)


def tone(dur, f=150.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 12))).astype(np.float32)


def silence(dur, noise=0.0005, seed=0):
    return (noise * np.random.default_rng(seed).standard_normal(int(dur * SR))).astype(np.float32)


def word_like(gap=None):
    """One 'word' (0.4 s voiced) or two separated by `gap` seconds of silence."""
    parts = [silence(0.4), tone(0.4)]
    if gap is not None:
        parts += [silence(gap, seed=1), tone(0.4, f=180)]
    parts.append(silence(0.4, seed=2))
    return np.concatenate(parts)
