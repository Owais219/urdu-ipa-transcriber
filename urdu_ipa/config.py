"""Application configuration (JSON file in the data folder)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .audio import VADConfig
from .measure import GeminationThresholds, VOTThresholds
from .reconcile import ReconcileConfig

import sys

PACKAGE_DIR = Path(__file__).resolve().parent
FROZEN = bool(getattr(sys, "frozen", False))
# In the packaged app (PyInstaller) resources live in sys._MEIPASS; in a source checkout, next to the package.
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", PACKAGE_DIR.parent))
PROJECT_DATA = RESOURCE_DIR / "data"
BUNDLED_WHISPER = RESOURCE_DIR / "models" / "whisper-urdu-ct2"
# the "lite" update app ships without Whisper and uses a copy kept in the user's data folder
USER_WHISPER = Path.home() / "UrduIPA" / "models" / "whisper-urdu-ct2"


def find_whisper() -> Path | None:
    """The converted Whisper model: inside the app, else the user's copy (lite update app)."""
    return _find_whisper()


@dataclass
class RecordingConfig:
    input_device: int | str | None = None
    max_duration: float = 6.0          # hard stop (s)
    auto_stop: bool = True
    trailing_silence: float = 1.0      # auto-stop after this much silence following speech (s)
    noise_calibration: float = 0.3     # first part of the recording used to estimate noise (s)
    speech_db_above_floor: float = 12.0
    min_speech_dbfs: float = -50.0


@dataclass
class Config:
    data_dir: str = str(Path.home() / "UrduIPA")
    lexicon_path: str = ""             # default: <data_dir>/lexicon.tsv (seeded on first run)
    profiles_path: str = str(PROJECT_DATA / "profiles.json")
    asr_backend: str = "auto"          # auto | faster-whisper | whisper | mms | manual
    asr_model: str = ""                # "" = backend default
    acoustic_backend: str = "allosaurus"   # allosaurus | hf-phoneme
    acoustic_model: str = ""
    device: str = "cpu"
    highpass_hz: float = 60.0
    min_asr_confidence: float = 0.5
    skip_asr_with_expected_word: bool = True   # fast mode: Whisper is not run when the expected word is typed
    processing_timeout: float = 300.0
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    reconcile: ReconcileConfig = field(default_factory=ReconcileConfig)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    def resolved_lexicon(self) -> Path:
        """The user's own lexicon (additions/overrides). The built-in lexicon is `builtin_lexicon`."""
        if self.lexicon_path:
            return Path(self.lexicon_path).expanduser()
        p = self.data_path / "my_lexicon.tsv"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# Your own pronunciation entries. They replace built-in entries with the same spelling.\n"
                         "word\tphonemic\tcitation_phonetic\tgloss\tsource\tverified\tnotes\n", encoding="utf-8")
        return p

    @property
    def builtin_lexicon(self) -> list[Path]:
        """Built-in lexicons, lowest priority first: Wiktionary (broad coverage), then the hand-checked list."""
        return [PROJECT_DATA / "lexicon_wiktionary.tsv", PROJECT_DATA / "lexicon.tsv"]

    def resolved_asr(self) -> tuple[str, str]:
        """auto: the bundled fast Whisper model when present, else Whisper via transformers."""
        if self.asr_backend != "auto":
            return self.asr_backend, self.asr_model
        found = _find_whisper()
        if found is not None:
            return "faster-whisper", self.asr_model or str(found)
        return "whisper", self.asr_model


def _fill(cls, data):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(v, dict):
            v = _fill(type(default), v)
        elif isinstance(default, tuple) and isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else cfg.data_path / "config.json"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            cfg = _fill(Config, json.load(f))
    return cfg


def save_config(cfg: Config, path: str | Path | None = None) -> Path:
    p = Path(path) if path else cfg.data_path / "config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    return p


def _find_whisper():
    # looked up at call time so tests can monkeypatch BUNDLED_WHISPER / USER_WHISPER
    import urdu_ipa.config as c
    for p in (c.BUNDLED_WHISPER, c.USER_WHISPER):
        if (p / "model.bin").exists():
            return p
    return None
