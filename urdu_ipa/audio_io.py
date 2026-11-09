"""Reading audio files (for "Open Audio Files…" and evaluation).

WAV, FLAC, AIFF and OGG are read with libsndfile (soundfile). MP3, M4A/AAC and other
compressed formats are decoded with FFmpeg through PyAV, which the app already ships
(faster-whisper uses it). Audio is returned unchanged (original sample rate, mono mix);
resampling and checks happen in the pipeline like for microphone recordings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = (".wav", ".wave", ".flac", ".aif", ".aiff", ".ogg", ".oga", ".mp3", ".m4a", ".aac", ".mp4",
                    ".caf", ".opus", ".wma", ".webm")
MAX_SECONDS = 60.0          # a single-word tool: longer files are read only up to this point


class AudioFileError(RuntimeError):
    pass


def load_audio(path: str | Path, max_seconds: float = MAX_SECONDS) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise AudioFileError(f"File not found: {path}")
    try:
        import soundfile as sf
        info = sf.info(str(path))
        frames = int(min(info.frames, max_seconds * info.samplerate)) if info.frames > 0 else -1
        x, sr = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
        return _mono(x), int(sr)
    except Exception:  # noqa: BLE001 - not a libsndfile format: try FFmpeg
        pass
    try:
        import av
    except ImportError as e:
        raise AudioFileError(f"Cannot read {path.suffix} files (audio decoder missing).") from e
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioFileError(f"No audio track in {path.name}.")
            sr = int(stream.rate or stream.codec_context.sample_rate)
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
            chunks, total = [], 0
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    a = out.to_ndarray().reshape(-1).astype(np.float32)
                    chunks.append(a)
                    total += a.size
                if total >= max_seconds * sr:
                    break
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1).astype(np.float32))
    except AudioFileError:
        raise
    except Exception as e:  # noqa: BLE001
        raise AudioFileError(f"Could not decode {path.name}: {e}") from e
    if not chunks:
        raise AudioFileError(f"{path.name} contains no audio.")
    x = np.concatenate(chunks)[: int(max_seconds * sr)]
    return x, sr


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def expected_word_from_filename(path: str | Path) -> str:
    """'جرگہ.m4a' or 'S01_جرگہ_2.wav' -> 'جرگہ' (the Urdu-script part of the name), else ''."""
    from .orthography import normalize, script_fraction
    stem = Path(path).stem
    parts = [p for p in stem.replace("-", "_").replace(" ", "_").split("_") if p]
    urdu = [p for p in parts if script_fraction(p) >= 0.9]
    return normalize(urdu[0]) if len(urdu) == 1 else ""
