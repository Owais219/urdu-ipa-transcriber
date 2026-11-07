"""Acoustic phone models and CTC forced scoring.

The acoustic model is used to SCORE candidate Urdu pronunciations against
the audio (CTC forced scoring), not to free-decode a transcription. Each Urdu
segment is mapped to a set of model phones ("proxies"). When two Urdu
segments have identical proxy sets, the model cannot tell them apart and the
contrast is reported as model-blind (e.g. Allosaurus has no [ʈʰ], so /ʈ/~/ʈʰ/
must be resolved by VOT measurement instead).

Backends
* AllosaurusBackend - universal phone model (GPL-3.0, offline, ~41 MB).
  Verified properties of model `uni2005`: 229 universal phones, 8 kHz MFCC
  input (40-3800 Hz), 30 ms output frames, trained on 14 corpora none of which
  is a South Asian language.
* HFPhonemeBackend - any Hugging Face wav2vec2-style CTC phoneme model
  (e.g. one fine-tuned on Urdu IPA). Experimental: not run in development.
"""
from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass

import numpy as np

from . import ipa

NEG_INF = -1e30

# Urdu segment -> candidate model phones (first = most faithful). Phones not in
# the model vocabulary are dropped automatically; nothing is invented.
PROXIES: dict[str, list[str]] = {
    "p": ["p"], "pʰ": ["pʰ"], "b": ["b"], "bʱ": ["b̤"], "m": ["m"],
    "t̪": ["t̪", "t"], "t̪ʰ": ["t̪ʰ", "tʰ"], "d̪": ["d̪", "d"], "d̪ʱ": ["d̪̤", "d̪", "d"],
    "n": ["n", "n̪"], "s": ["s", "s̪"], "z": ["z", "z̪"], "r": ["r", "ɾ"], "l": ["l", "l̪"],
    "ʈ": ["ʈ"], "ʈʰ": ["ʈʰ", "ʈ"], "ɖ": ["ɖ", "ɖ̤"], "ɖʱ": ["ɖ̤"], "ɳ": ["ɳ"],
    "ɽ": ["ɽ", "ɻ"], "ɽʱ": ["ɽ̤", "ɽ", "ɻ"],
    "t͡ʃ": ["t͡ʃ", "tʃ"], "t͡ʃʰ": ["t͡ʃʰ", "tʃʰ", "t͡ʃ", "tʃ"], "d͡ʒ": ["d͡ʒ", "dʒ"],
    "d͡ʒʱ": ["d͡ʒ̤"], "ʃ": ["ʃ"], "ʒ": ["ʒ"], "j": ["j"],
    "k": ["k"], "kʰ": ["kʰ"], "ɡ": ["ɡ", "g"], "ɡʱ": ["ɡ̤"], "ŋ": ["ŋ"],
    "x": ["x", "χ"], "ɣ": ["ɣ", "ʁ"], "q": ["q"], "f": ["f"], "ʋ": ["ʋ", "w", "v"],
    "h": ["h", "ɦ"], "ɦ": ["ɦ", "h"], "ʔ": ["ʔ"], "ħ": ["ħ", "h"], "ʕ": ["ʕ", "ʔ"],
    "ɪ": ["ɪ"], "iː": ["iː", "i"], "ʊ": ["ʊ"], "uː": ["uː", "u"], "ə": ["ə", "ʌ", "ɐ", "ә"],
    "ɑː": ["ɑː", "aː", "ɑ", "a", "ä"], "ɑ": ["ɑ", "a", "ä", "ɐ"], "eː": ["eː", "e", "e̞"], "ɛː": ["ɛː", "ɛ", "æ"],
    "oː": ["oː", "o", "o̞"], "ɔː": ["ɔː", "ɔ", "ɒ"], "ɛ": ["ɛ"], "ɔ": ["ɔ"],
}
# nasal vowels: use nasal model phones when they exist, otherwise the oral proxies
_NASAL_PROXIES = {"ɑ̃ː": ["ɑ̃ː", "ã"], "ɛ̃ː": ["ɛ̃ː", "ɛ̃"], "ɔ̃ː": ["ɔ̃ː", "ɔ̃"], "ə̃": ["ə̃", "ã"]}
for _v in list(ipa.VOWELS):
    if ipa.VOWELS[_v].nasal_vowel:
        _oral = ipa.denasalize(_v)
        PROXIES[_v] = _NASAL_PROXIES.get(_v, [_v] + PROXIES.get(_oral, []))


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


@dataclass
class Emissions:
    """Frame log-probabilities over the model vocabulary."""
    logp: np.ndarray           # (T, V)
    frame_shift: float         # seconds per frame
    blank: int
    vocab: dict[str, int]

    @property
    def T(self) -> int:
        return int(self.logp.shape[0])


class AcousticBackend:
    name = "base"
    version = ""

    def __init__(self):
        self._lock = threading.Lock()
        self._proxy_cache: dict[str, tuple[int, ...]] = {}

    # -- to implement ----------------------------------------------------
    def load(self) -> None: ...
    def vocab(self) -> dict[str, int]: ...
    def emissions(self, audio16k: np.ndarray) -> Emissions: ...

    # -- shared ------------------------------------------------------------
    def proxy_ids(self, seg: str) -> tuple[int, ...]:
        if seg in self._proxy_cache:
            return self._proxy_cache[seg]
        voc = self.vocab()
        base = ipa.ungeminate(seg)
        cands = PROXIES.get(base, [base])
        ids = []
        for c in cands:
            c = unicodedata.normalize("NFC", c)
            if c in voc and voc[c] not in ids:
                ids.append(voc[c])
        self._proxy_cache[seg] = tuple(ids)
        return self._proxy_cache[seg]

    def can_score(self, seg: str) -> bool:
        return bool(self.proxy_ids(seg))

    def discriminable(self, a: str, b: str) -> bool:
        if a == b:
            return False
        if a == "" or b == "":
            return True
        return set(self.proxy_ids(a)) != set(self.proxy_ids(b))


# ---------------------------------------------------------------------------
# CTC scoring
# ---------------------------------------------------------------------------

def segment_scores(em: Emissions, seq: list[str], backend: AcousticBackend) -> np.ndarray:
    """(T, len(seq)) log-scores: logsumexp of the segment's proxy phones per frame."""
    out = np.full((em.T, len(seq)), NEG_INF, dtype=np.float64)
    for j, s in enumerate(seq):
        ids = backend.proxy_ids(s)
        if ids:
            sub = em.logp[:, ids]
            m = sub.max(axis=1, keepdims=True)
            out[:, j] = (m + np.log(np.exp(sub - m).sum(axis=1, keepdims=True)))[:, 0]
    return out


def _ctc_lattice(em: Emissions, seq: list[str], backend: AcousticBackend):
    seq = [s for s in seq if s]
    S = 2 * len(seq) + 1
    blank = em.logp[:, em.blank].astype(np.float64)
    seg = segment_scores(em, seq, backend) if seq else np.zeros((em.T, 0))
    emit = np.empty((em.T, S))
    emit[:, 0::2] = blank[:, None]
    if seq:
        emit[:, 1::2] = seg
    # skip transition s-2 -> s allowed for non-blank s whose label differs from s-2's
    skip = np.zeros(S, dtype=bool)
    for k in range(1, len(seq)):
        skip[2 * k + 1] = set(backend.proxy_ids(seq[k])) != set(backend.proxy_ids(seq[k - 1]))
    return seq, S, emit, skip


def ctc_logprob(em: Emissions, seq: list[str], backend: AcousticBackend) -> float:
    """log P(seq | audio) under CTC (forward algorithm)."""
    seq, S, emit, skip = _ctc_lattice(em, seq, backend)
    T = em.T
    if T == 0 or len(seq) > T:
        return NEG_INF
    alpha = np.full(S, NEG_INF)
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]
    for t in range(1, T):
        prev = alpha
        a1 = np.concatenate(([NEG_INF], prev))[:S]
        a2 = np.concatenate(([NEG_INF, NEG_INF], prev))[:S]
        a2 = np.where(skip, a2, NEG_INF)
        stacked = np.stack([prev, a1, a2])
        m = stacked.max(axis=0)
        alpha = m + np.log(np.exp(stacked - m).sum(axis=0) + 1e-300) + emit[t]
        alpha = np.maximum(alpha, NEG_INF)
    if S == 1:
        return float(alpha[0])
    m = max(alpha[-1], alpha[-2])
    return float(m + np.log(np.exp(alpha[-1] - m) + np.exp(alpha[-2] - m)))


def ctc_align(em: Emissions, seq: list[str], backend: AcousticBackend) -> tuple[float, list[tuple[int, int]]]:
    """Viterbi alignment. Returns best-path score and (first, last) frame per non-empty segment."""
    seq, S, emit, skip = _ctc_lattice(em, seq, backend)
    T = em.T
    if T == 0 or len(seq) > T:
        return NEG_INF, []
    delta = np.full((T, S), NEG_INF)
    bp = np.zeros((T, S), dtype=np.int64)
    delta[0, 0] = emit[0, 0]
    if S > 1:
        delta[0, 1] = emit[0, 1]
    for t in range(1, T):
        p = delta[t - 1]
        c0, c1 = p, np.concatenate(([NEG_INF], p))[:S]
        c2 = np.where(skip, np.concatenate(([NEG_INF, NEG_INF], p))[:S], NEG_INF)
        stack = np.stack([c0, c1, c2])
        arg = stack.argmax(axis=0)
        delta[t] = stack.max(axis=0) + emit[t]
        bp[t] = np.arange(S) - arg
    if S == 1:
        return float(delta[-1, 0]), []
    s = S - 1 if delta[-1, S - 1] >= delta[-1, S - 2] else S - 2
    score = float(delta[-1, s])
    path = [s]
    for t in range(T - 1, 0, -1):
        s = int(bp[t, s])
        path.append(s)
    path.reverse()
    spans: list[list[int]] = [[-1, -1] for _ in seq]
    for t, st in enumerate(path):
        if st % 2 == 1:
            k = st // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t
    return score, [tuple(x) for x in spans]


def free_decode(em: Emissions, backend: AcousticBackend, symbols: list[str]) -> tuple[list[str], float]:
    """Unconstrained best path over Urdu symbols (diagnostic only)."""
    usable = [s for s in symbols if backend.can_score(s)]
    if not usable or em.T == 0:
        return [], NEG_INF
    scores = segment_scores(em, usable, backend)
    blank = em.logp[:, em.blank][:, None]
    allsc = np.concatenate([blank, scores], axis=1)
    best = allsc.argmax(axis=1)
    total = float(allsc.max(axis=1).sum())
    out, prev = [], -1
    for b in best:
        if b != prev and b != 0:
            out.append(usable[b - 1])
        prev = b
    return out, total


def subphone_preference(em: Emissions, frames: tuple[int, int], a: str, b: str) -> float:
    """log P(a) - log P(b) summed over the given frames, for model phones a and b."""
    if a not in em.vocab or b not in em.vocab or frames[0] < 0:
        return 0.0
    sl = em.logp[frames[0]:frames[1] + 1]
    return float((sl[:, em.vocab[a]] - sl[:, em.vocab[b]]).sum())


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _set_torch_threads(torch) -> None:
    import os
    n = os.cpu_count() or 2
    try:
        torch.set_num_threads(max(1, n // 2 if n >= 4 else n))   # physical cores; hyper-threads do not help
    except RuntimeError:
        pass


class AllosaurusBackend(AcousticBackend):
    name = "allosaurus"

    def __init__(self, model_name: str = "uni2005", device_id: int = -1):
        super().__init__()
        self.model_name = model_name
        self.device_id = device_id
        self._rec = None
        self._vocab: dict[str, int] | None = None
        self.version = f"allosaurus/{model_name}"

    def load(self) -> None:
        with self._lock:
            if self._rec is not None:
                return
            from . import _resampy_compat
            _resampy_compat.install()                    # avoid numba (slow start, no Intel-Mac builds)
            import torch
            _set_torch_threads(torch)
            from allosaurus.app import read_recognizer   # heavy import, done once
            self._rec = read_recognizer(self.model_name)
            unit = self._rec.lm.inventory.unit
            self._vocab = {unicodedata.normalize("NFC", u): i for u, i in unit.unit_to_id.items()}
            self.frame_shift = float(self._rec.config.window_shift)

    def vocab(self) -> dict[str, int]:
        if self._vocab is None:
            self.load()
        return self._vocab

    def emissions(self, audio16k: np.ndarray) -> Emissions:
        self.load()
        from allosaurus.audio import Audio
        from allosaurus.am.utils import move_to_tensor
        import torch
        a = Audio()
        pcm = np.clip(audio16k, -1.0, 1.0)
        a.set_header(sample_rate=16000, sample_size=len(pcm), channel_number=1, sample_width=2)
        a.samples = (pcm * 32767).astype(np.int16)
        with self._lock, torch.no_grad():
            feat = self._rec.pm.compute(a)
            feats = np.expand_dims(feat, 0)
            flen = np.array([feat.shape[0]], dtype=np.int32)
            tf, tl = move_to_tensor([feats, flen], self.device_id)
            logits = self._rec.am(tf, tl).cpu().numpy()[0].astype(np.float64)
        return Emissions(log_softmax(logits), self.frame_shift, 0, self._vocab)


class HFPhonemeBackend(AcousticBackend):
    """Wav2vec2-style CTC phoneme recogniser from Hugging Face (experimental)."""
    name = "hf-phoneme"

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self._model = self._proc = None
        self._vocab = None
        self.version = f"hf/{model_id}"

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoProcessor, AutoModelForCTC
            self._proc = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()
            tok = self._proc.tokenizer
            self._vocab = {unicodedata.normalize("NFC", k): v for k, v in tok.get_vocab().items()}
            self._blank = tok.pad_token_id if tok.pad_token_id is not None else 0
            ratio = getattr(self._model.config, "inputs_to_logits_ratio", 320)
            self.frame_shift = ratio / 16000.0

    def vocab(self):
        if self._vocab is None:
            self.load()
        return self._vocab

    def emissions(self, audio16k):
        self.load()
        import torch
        inp = self._proc(audio16k, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with self._lock, torch.no_grad():
            logits = self._model(inp).logits[0].cpu().numpy().astype(np.float64)
        return Emissions(log_softmax(logits), self.frame_shift, self._blank, self._vocab)


def make_backend(kind: str, model: str = "") -> AcousticBackend:
    if kind == "allosaurus":
        return AllosaurusBackend(model or "uni2005")
    if kind == "hf-phoneme":
        if not model:
            raise ValueError("hf-phoneme backend needs a model id")
        return HFPhonemeBackend(model)
    raise ValueError(f"unknown acoustic backend: {kind}")
