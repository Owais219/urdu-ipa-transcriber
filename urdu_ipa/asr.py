"""Urdu word recognition (orthographic). This is NOT phonetic transcription.

Backends
* WhisperASR - any Whisper checkpoint via transformers, decoding forced to
  Urdu ("urdu", task "transcribe") to avoid the script confusion reported for
  zero-shot Whisper on Urdu (Devanagari / Latin output). Default checkpoint:
  kingabzpro/whisper-large-v3-turbo-urdu (fine-tuned on Common Voice 17 Urdu;
  model card reports test WER 26.2 %, CER 8.8 %, on sentences - single-word
  accuracy is not reported and must be measured).
* MMSASR - facebook/mms-1b-all with the urd-script_arabic adapter
  (licence CC-BY-NC 4.0: non-commercial use only).
* FasterWhisperASR - the same Whisper model converted to CTranslate2 int8 (used by the
  packaged Mac app: much faster on CPU, no download at run time).
* ManualASR - no model: the researcher types the target word (elicitation).

The confidence value returned is the mean token probability of the decoded
string. It is a model-internal score, NOT a calibrated probability.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .orthography import normalize, script_fraction


@dataclass
class ASRResult:
    text: str
    confidence: float | None
    model: str
    alternatives: list[str] = field(default_factory=list)


class ASRBackend:
    name = "base"
    version = ""

    def load(self) -> None: ...
    def transcribe(self, audio16k: np.ndarray) -> ASRResult: ...


class ManualASR(ASRBackend):
    name = "manual"
    version = "manual-entry"

    def transcribe(self, audio16k):
        return ASRResult("", None, self.version)


class WhisperASR(ASRBackend):
    name = "whisper"

    def __init__(self, model_id: str = "kingabzpro/whisper-large-v3-turbo-urdu", device: str = "cpu",
                 num_beams: int = 3):
        self.model_id = model_id
        self.device = device
        self.num_beams = num_beams
        self.version = f"whisper/{model_id}"
        self._model = self._proc = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self._proc = WhisperProcessor.from_pretrained(self.model_id)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()
            gc = self._model.generation_config
            gc.forced_decoder_ids = None
            self._torch = torch

    def transcribe(self, audio16k):
        self.load()
        torch = self._torch
        feats = self._proc(audio16k, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with self._lock, torch.no_grad():
            out = self._model.generate(feats, language="urdu", task="transcribe", max_new_tokens=24,
                                       num_beams=self.num_beams, num_return_sequences=min(3, self.num_beams),
                                       return_dict_in_generate=True, output_scores=True)
        texts = self._proc.batch_decode(out.sequences, skip_special_tokens=True)
        conf = None
        try:
            if getattr(out, "sequences_scores", None) is not None:
                # beam search: length-normalised log-probability of the best hypothesis
                conf = float(np.exp(out.sequences_scores[0].item()))
            else:
                ts = self._model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                row = ts[0].cpu().numpy()
                row = row[np.isfinite(row)]
                conf = float(np.exp(row.mean())) if row.size else None
        except Exception:
            conf = None
        texts = [normalize(t) for t in texts]
        top = texts[0] if texts else ""
        alts = [t for t in dict.fromkeys(texts[1:]) if t and t != top]
        return ASRResult(top, conf, self.version, alternatives=alts)


class MMSASR(ASRBackend):
    name = "mms"

    def __init__(self, model_id: str = "facebook/mms-1b-all", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.version = f"mms/{model_id}/urd-script_arabic"
        self._model = self._proc = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            self._proc = AutoProcessor.from_pretrained(self.model_id)
            self._proc.tokenizer.set_target_lang("urd-script_arabic")
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            self._model.load_adapter("urd-script_arabic")
            self._model.to(self.device).eval()
            self._torch = torch

    def transcribe(self, audio16k):
        self.load()
        torch = self._torch
        inp = self._proc(audio16k, sampling_rate=16000, return_tensors="pt").to(self.device)
        with self._lock, torch.no_grad():
            logits = self._model(**inp).logits[0]
        probs = torch.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1)
        pad = self._proc.tokenizer.pad_token_id
        keep = ids != pad
        conf = float(probs.max(dim=-1).values[keep].mean()) if keep.any() else None
        text = normalize(self._proc.decode(ids))
        return ASRResult(text, conf, self.version)


class FasterWhisperASR(ASRBackend):
    """Whisper through CTranslate2 (faster-whisper), int8 on CPU.

    Several times faster than transformers on Intel CPUs and loads in seconds. The model
    folder must be a CTranslate2 conversion (see packaging/convert_whisper.py) containing
    model.bin, config.json, tokenizer.json and preprocessor_config.json, so nothing is
    downloaded at run time.
    """
    name = "faster-whisper"

    def __init__(self, model_path: str, device: str = "cpu", compute_type: str = "int8", beam_size: int = 3,
                 cpu_threads: int = 0):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        from pathlib import Path
        info = Path(model_path) / "SOURCE.txt"
        src = info.read_text(encoding="utf-8").strip().splitlines()[0] if info.exists() else Path(model_path).name
        self.version = f"faster-whisper/{src}/{compute_type}"
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_path, device=self.device, compute_type=self.compute_type,
                                       cpu_threads=self.cpu_threads, local_files_only=True)

    def transcribe(self, audio16k):
        self.load()
        with self._lock:
            segments, _info = self._model.transcribe(
                np.asarray(audio16k, dtype=np.float32), language="ur", task="transcribe",
                beam_size=self.beam_size, temperature=0.0, condition_on_previous_text=False,
                max_new_tokens=24,                     # one word: never let Whisper run on
                without_timestamps=True, vad_filter=False, word_timestamps=False)
            segs = list(segments)
        text = normalize(" ".join(s.text for s in segs))
        conf = None
        if segs:
            lp = [s.avg_logprob for s in segs if s.avg_logprob is not None]
            conf = float(np.exp(np.mean(lp))) if lp else None
            if any(getattr(s, "no_speech_prob", 0) > 0.8 for s in segs):
                conf = min(conf or 0.0, 0.2)
        return ASRResult(text, conf, self.version)


def make_asr(kind: str, model: str = "", device: str = "cpu") -> ASRBackend:
    if kind == "faster-whisper":
        if not model:
            raise ValueError("faster-whisper needs a converted model folder (asr_model)")
        return FasterWhisperASR(model, device=device)
    if kind == "whisper":
        return WhisperASR(model, device=device) if model else WhisperASR(device=device)
    if kind == "mms":
        return MMSASR(model, device=device) if model else MMSASR(device=device)
    if kind == "manual":
        return ManualASR()
    raise ValueError(f"unknown ASR backend: {kind}")


def check_script(text: str, min_fraction: float = 0.9) -> bool:
    return bool(text) and script_fraction(text) >= min_fraction
