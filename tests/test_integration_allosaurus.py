"""Integration with the real Allosaurus model. Skipped when it is not installed.

Audio is synthesised with espeak-ng (if present). Synthetic speech is a
smoke test of the machinery ONLY: it says nothing about accuracy on human
speech. The assertion is that the pipeline returns a well-formed result
(transcription or explicit refusal) without errors, in reasonable time.
"""
import shutil
import subprocess

import numpy as np
import pytest

allosaurus = pytest.importorskip("allosaurus")
sf = pytest.importorskip("soundfile")

from urdu_ipa.acoustic import AllosaurusBackend
from urdu_ipa.config import Config
from urdu_ipa.pipeline import Pipeline


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    be = AllosaurusBackend()
    try:
        be.load()
    except Exception as e:
        pytest.skip(f"Allosaurus model unavailable: {e}")
    cfg = Config(data_dir=str(tmp_path_factory.mktemp("d")), asr_backend="manual")
    p = Pipeline(cfg, acoustic=be)
    p.load()
    return p


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
@pytest.mark.parametrize("word", ["پھل", "کھال", "دال", "ڈال", "ماں", "پتّا", "کتاب", "قلم"])
def test_real_model_returns_wellformed_result(pipeline, tmp_path, word):
    wav = tmp_path / "w.wav"
    subprocess.run(["espeak-ng", "-v", "ur", "-w", str(wav), word], check=True)
    x, sr = sf.read(wav)
    pad = np.zeros(int(0.4 * sr))
    x = np.concatenate([pad, x, pad]) + 0.0005 * np.random.default_rng(0).standard_normal(len(x) + 2 * len(pad))
    r = pipeline.process(x, sr, expected_word=word)
    assert r.status in ("ok", "refused"), r.warnings
    assert r.segments, "segment evidence must always be reported"
    assert (r.phonetic is None) == (r.status != "ok")
    assert r.processing_ms < 10000
