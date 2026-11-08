"""End-to-end processing of one recorded word."""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np

from . import __version__, audio as A, ipa
from .acoustic import AcousticBackend, make_backend
from .asr import ASRBackend, ASRResult, check_script, make_asr
from .config import Config
from .lexicon import Lexicon
from .orthography import ends_with_final_heh, final_h_slot, lattice_from_spelling, lookup_key, normalize, words
from .reconcile import (Reconciler, Reconciliation, all_variant_pairs, enrich_spelling_slots, load_profiles,
                        slots_from_phonemes)

UNRELIABLE = "Unable to establish a reliable phonetic transcription. Please record the word again."
MESSAGES = {
    "empty": "The recording is empty. Please record the word again.",
    "no_speech": "No speech detected. Please record the word again.",
    "multiple_words": "Please speak only one Urdu word at a time.",
    "too_short": "The recording is too short or incomplete to analyse. Please record the word again.",
    "too_long": "The recording is too long for a single word. Please speak only one Urdu word at a time.",
    "clipped": "The recording is clipped (input too loud). Move back from the microphone and record again.",
    "noisy": "Background noise is too high for reliable phonetic analysis. Please record again somewhere quieter.",
    "unrecognized": "The word could not be recognised as an Urdu word. Please record the word again.",
    "non_urdu_script": "The recogniser did not return Urdu script, so the word could not be identified. Please record the word again.",
    "ambiguous": "The pronunciation is ambiguous between several dictionary forms. " + UNRELIABLE,
    "unreliable": UNRELIABLE,
    "processing_failure": "Processing failed. Please record the word again.",
    "no_word_given": "No speech recogniser is active: type the expected word before recording.",
    "models_unavailable": "The speech models are not available. See the status bar and the README model setup section.",
}
WARNINGS = {
    "moderate_noise": "Moderate background noise: weak cues such as aspiration may be affected.",
    "onset_truncated": "Speech starts at the very beginning of the recording; the word onset may be cut off.",
    "offset_truncated": "Speech continues to the very end of the recording; the word ending may be cut off.",
    "asr_expected_mismatch": "The recogniser heard a different word from the expected word. Both were scored; see 'analysed as'.",
    "joined_tokens": "The recogniser wrote the word with an internal space; it was treated as one orthographic word.",
    "lexicon_unverified": "The dictionary entry used has not been verified by a phonologist.",
    "no_lexicon_entry": "Word not in the pronunciation lexicon: vowels and ambiguous letters were decided from the audio only. No phonemic form is shown.",
    "low_asr_confidence": "Word recognition confidence is low.",
    "asr_skipped": "Fast mode: word recognition was skipped and the typed expected word was analysed. "
                   "The pronunciation is still decided from the audio; if the speaker said a different word, "
                   "the result is refused.",
}


@dataclass
class Result:
    status: str = "error"                 # ok | refused | error
    message: str = ""
    problem: str = ""
    recognized_word: str = ""
    asr_text: str = ""
    asr_alternatives: list[str] = field(default_factory=list)
    expected_word: str = ""
    analysed_word: str = ""
    word_confidence: float | None = None
    word_confidence_level: str = "n/a"
    phonetic: str | None = None
    phonemic: str | None = None
    narrow: str | None = None
    phonetic_confidence: float | None = None
    phonetic_confidence_level: str = "n/a"
    candidate: str = ""
    lexicon_verified: bool | None = None
    segments: list[dict] = field(default_factory=list)
    free_decode: str = ""
    fit_gap: float | None = None
    other_candidates: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audio_report: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    processing_ms: float = 0.0
    profile: str = "standard"
    models: dict = field(default_factory=dict)
    app_version: str = __version__

    def warning_texts(self) -> list[str]:
        return [WARNINGS.get(w, w) for w in self.warnings]

    def to_dict(self) -> dict:
        return asdict(self)


def _level(x: float | None, hi=0.8, mid=0.5) -> str:
    if x is None:
        return "n/a"
    return "high" if x >= hi else ("medium" if x >= mid else "low")


class Pipeline:
    """Holds the loaded models; safe to reuse across recordings (one job at a time)."""

    def __init__(self, cfg: Config, asr: ASRBackend | None = None, acoustic: AcousticBackend | None = None,
                 lexicon: Lexicon | None = None):
        self.cfg = cfg
        self.asr = asr or make_asr(*cfg.resolved_asr(), device=cfg.device)
        self.acoustic = acoustic or make_backend(cfg.acoustic_backend, cfg.acoustic_model)
        self.lexicon = lexicon or Lexicon(cfg.resolved_lexicon(), base=cfg.builtin_lexicon)
        self.profiles = load_profiles(cfg.profiles_path)
        self.variant_pairs = all_variant_pairs(self.profiles)
        self._job_lock = threading.Lock()
        self.load_errors: dict[str, str] = {}
        self.load_times_s: dict[str, float] = {}
        self.loaded = threading.Event()

    def load(self, progress=None) -> dict[str, str]:
        """Load models once. Errors are recorded, not raised, so the UI can report them."""
        steps = (("acoustic", self.acoustic, "phone model"), ("asr", self.asr, "speech recogniser"))
        for name, obj, label in steps:
            if progress:
                progress(f"Loading {label} ({obj.version})… first run downloads it; see the Terminal window for progress.")
            t = time.perf_counter()
            try:
                obj.load()
            except Exception as e:  # missing package, failed download, ...
                self.load_errors[name] = f"{type(e).__name__}: {e}"
            self.load_times_s[name] = round(time.perf_counter() - t, 1)
        self.loaded.set()
        return self.load_errors

    def models(self) -> dict:
        return {"asr": self.asr.version, "acoustic": self.acoustic.version, "app": __version__}

    # ------------------------------------------------------------------
    def process(self, samples: np.ndarray, sr: int, expected_word: str = "", profile: str = "standard",
                skip_asr: bool | None = None) -> Result:
        if not self._job_lock.acquire(blocking=False):
            r = Result(status="error", problem="busy", message="Another recording is still being processed.")
            return r
        try:
            return self._process(samples, sr, expected_word, profile, skip_asr)
        except Exception:
            r = Result(status="error", problem="processing_failure", message=MESSAGES["processing_failure"])
            r.warnings.append(traceback.format_exc(limit=3))
            return r
        finally:
            self._job_lock.release()

    def _process(self, samples, sr, expected_word, profile, skip_asr=None) -> Result:
        t_start = time.perf_counter()
        cfg = self.cfg
        res = Result(profile=profile, models=self.models(), expected_word=normalize(expected_word or ""))
        timings = res.timings_ms

        def done(status, problem=""):
            res.status, res.problem = status, problem
            if problem:
                res.message = MESSAGES.get(problem, UNRELIABLE)
            res.processing_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return res

        t = time.perf_counter()
        x16 = A.to_16k_mono(samples, sr)
        rep = A.analyse(x16, A.TARGET_SR, cfg.vad)
        res.audio_report = {"duration_s": round(rep.duration, 3), "speech_s": round(rep.speech_end - rep.speech_start, 3),
                            "islands": rep.n_islands, "snr_db": round(rep.snr_db, 1),
                            "clipped_fraction": round(rep.clipped_fraction, 5), "peak": round(rep.peak, 4)}
        res.warnings += rep.warnings
        timings["audio_ms"] = round((time.perf_counter() - t) * 1000, 1)
        for p in ("empty", "no_speech", "clipped", "multiple_words", "too_short", "too_long", "noisy"):
            if p in rep.problems:
                return done("refused", p)
        x = A.prepare(x16, cfg.highpass_hz, report=rep)

        if not self.loaded.is_set():
            self.loaded.wait(timeout=cfg.processing_timeout)
        if "acoustic" in self.load_errors:
            return self._fail(res, t_start, "models_unavailable", self.load_errors["acoustic"])
        if "asr" in self.load_errors and self.asr.name != "manual":
            res.warnings.append("ASR model unavailable (" + self.load_errors["asr"] + "); type the expected word.")

        # ---- word recognition ----
        t = time.perf_counter()
        asr: ASRResult | None = None
        if skip_asr is None:
            skip_asr = self.cfg.skip_asr_with_expected_word
        fast = bool(skip_asr and res.expected_word)
        if fast:
            res.warnings.append("asr_skipped")
        elif self.asr.name != "manual" and "asr" not in self.load_errors:
            asr = self.asr.transcribe(x)
            res.asr_text, res.asr_alternatives = asr.text, asr.alternatives
            res.word_confidence = None if asr.confidence is None else round(asr.confidence, 3)
            res.word_confidence_level = _level(asr.confidence)
        timings["asr_ms"] = round((time.perf_counter() - t) * 1000, 1)

        cand_words: list[str] = []
        if asr is not None and asr.text:
            if not check_script(asr.text):
                if not res.expected_word:
                    return done("refused", "non_urdu_script")
            else:
                toks = words(asr.text)
                if len(toks) > 1:
                    joined = "".join(toks)
                    if joined in self.lexicon or (res.expected_word and lookup_key(joined) == lookup_key(res.expected_word)):
                        res.warnings.append("joined_tokens")
                        toks = [joined]
                    else:
                        res.recognized_word = asr.text
                        return done("refused", "multiple_words")
                if toks:
                    res.recognized_word = toks[0]
                    cand_words.append(toks[0])
        if res.expected_word:
            if cand_words and lookup_key(cand_words[0]) != lookup_key(res.expected_word):
                res.warnings.append("asr_expected_mismatch")
            if res.expected_word not in cand_words:
                cand_words.append(res.expected_word)
        if not cand_words:
            no_asr = self.asr.name == "manual" or "asr" in self.load_errors
            return done("refused", "no_word_given" if no_asr else "unrecognized")
        if (res.word_confidence is not None and res.word_confidence < cfg.min_asr_confidence):
            res.warnings.append("low_asr_confidence")
            if not res.expected_word:
                return done("refused", "unrecognized")

        # ---- pronunciation analysis ----
        t = time.perf_counter()
        em = self.acoustic.emissions(x)
        timings["acoustic_model_ms"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        prof = self.profiles.get(profile) or self.profiles["standard"]
        rec = Reconciler(self.acoustic, cfg.reconcile, prof)
        outcomes: list[tuple[Reconciliation, str, object]] = []
        for w in cand_words:
            entries = self.lexicon.lookup(w)
            if entries:
                for e in entries:
                    slots = slots_from_phonemes(ipa.tokenize(e.expected_phonetic), self.variant_pairs)
                    if ends_with_final_heh(w) and not e.expected_phonetic.endswith("h"):
                        slots.append(final_h_slot())
                    outcomes.append((rec.run(em, slots, x, A.TARGET_SR, f"/{e.phonemic}/ {e.gloss}".strip()), w, e))
            else:
                lat = lattice_from_spelling(w)
                if lat:
                    slots = enrich_spelling_slots(lat, self.variant_pairs)
                    outcomes.append((rec.run(em, slots, x, A.TARGET_SR, "spelling"), w, None))
        timings["phonetic_analysis_ms"] = round((time.perf_counter() - t) * 1000, 1)
        if not outcomes:
            return done("refused", "unrecognized")
        outcomes.sort(key=lambda o: o[0].score, reverse=True)
        # among candidates giving the same realisation with near-equal fit, prefer the dictionary
        # form that needs the fewest assumed deviations (acoustic variants) to explain the audio
        top = outcomes[0][0]
        def n_dev(o):
            return sum(1 for s in o[0].segments
                       if s.status == "acoustic_variant" or (s.status == "not_realised" and s.source == "lexical"))
        ties = [o for o in outcomes if top.score - o[0].score < cfg.reconcile.margin and o[0].phonetic == top.phonetic]
        chosen = min(ties, key=n_dev)
        outcomes.remove(chosen)
        outcomes.insert(0, chosen)
        best, word, entry = outcomes[0]
        res.analysed_word = word
        res.candidate = best.candidate_label
        res.segments = [asdict(s) for s in best.segments]
        res.free_decode = best.free_decode
        res.fit_gap = best.fit_gap
        res.phonetic_confidence = best.confidence
        res.phonetic_confidence_level = best.level
        res.other_candidates = [{"word": w2, "candidate": r2.candidate_label, "score": r2.score,
                                 "score_gap": round(best.score - r2.score, 3), "phonetic": r2.phonetic}
                                for r2, w2, _ in outcomes[1:]]
        if entry is not None:
            res.lexicon_verified = entry.is_verified
            if not entry.is_verified:
                res.warnings.append("lexicon_unverified")
        else:
            res.warnings.append("no_lexicon_entry")

        # homograph / competing word ambiguity: distinct pronunciations too close to call
        rivals = [o for o in outcomes[1:] if best.score - o[0].score < cfg.reconcile.margin
                  and (o[0].phonetic or "") != (best.phonetic or "")]
        if rivals:
            res.warnings.append("competing: " + ", ".join(o[0].candidate_label for o in rivals))
            return done("refused", "ambiguous")
        if not best.reliable:
            res.warnings += [f"reason: {r}" for r in best.reasons]
            return done("refused", "unreliable")
        res.phonetic = f"[{best.phonetic}]"
        res.narrow = f"[{best.narrow}]" if best.narrow and best.narrow != best.phonetic else None
        if entry is not None:
            res.phonemic = f"/{entry.phonemic}/"
        res.message = ""
        return done("ok")

    def _fail(self, res, t_start, problem, detail):
        res.status, res.problem = "error", problem
        res.message = MESSAGES[problem] + (f" ({detail})" if detail else "")
        res.processing_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return res
