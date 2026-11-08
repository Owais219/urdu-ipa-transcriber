import json

import numpy as np
import pytest

from urdu_ipa import audio as A
from urdu_ipa.config import Config, PROJECT_DATA
from urdu_ipa.evaluation import Item, evaluate, items_from_store, per, write_report
from urdu_ipa.lexicon import Entry, Lexicon
from urdu_ipa.measure import measure_closure, measure_vot
from urdu_ipa.pipeline import MESSAGES, UNRELIABLE, Pipeline
from urdu_ipa.reconcile import ReconcileConfig, Reconciler, load_profiles, slots_from_phonemes, all_variant_pairs
from urdu_ipa.storage import Store
from tests.helpers import SR, FakeAcoustic, FakeASR, silence, tone, word_like

VAD = A.VADConfig()


# ---------------- audio QC / segmentation ----------------
def test_single_word_is_one_island():
    rep = A.analyse(word_like(), SR, VAD)
    assert rep.problems == [] and rep.n_islands == 1


def test_geminate_length_closure_does_not_split_word():
    rep = A.analyse(word_like(gap=0.2), SR, VAD)
    assert "multiple_words" not in rep.problems


def test_two_words_with_pause_detected():
    rep = A.analyse(word_like(gap=0.6), SR, VAD)
    assert "multiple_words" in rep.problems


@pytest.mark.parametrize("sig,problem", [
    (np.zeros(0, np.float32), "empty"),
    (np.zeros(SR, np.float32), "no_speech"),
    (np.concatenate([silence(0.4), tone(0.08), silence(0.4)]), "too_short"),
    (np.concatenate([silence(0.3), np.clip(tone(0.5, amp=3.0), -1, 1), silence(0.3)]), "clipped"),
    (np.concatenate([silence(0.3), tone(3.0), silence(0.3)]), "too_long"),
])
def test_invalid_recordings(sig, problem):
    assert problem in A.analyse(sig, SR, VAD).problems


def test_noisy_recording_detected():
    rng = np.random.default_rng(0)
    x = word_like() + 0.15 * rng.standard_normal(word_like().size).astype(np.float32)
    assert "noisy" in A.analyse(x, SR, VAD).problems


def test_resampling_44k_to_16k():
    x = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
    y = A.to_16k_mono(x, 44100)
    assert abs(len(y) - 16000) <= 1


def test_preprocessing_keeps_aspiration_noise():
    # 60 Hz high-pass must not remove broadband aspiration noise
    rng = np.random.default_rng(1)
    noise = 0.1 * rng.standard_normal(SR).astype(np.float32)
    y = A.prepare(noise)
    assert np.std(y) > 0.2


# ---------------- measurements ----------------
def _stop(asp, pre_vowel=0.0, closure=0.12, seed=0):
    rng = np.random.default_rng(seed)
    parts = [tone(pre_vowel)] if pre_vowel else []
    parts += [0.001 * rng.standard_normal(int(closure * SR)), 0.6 * rng.standard_normal(int(0.005 * SR)),
              0.08 * rng.standard_normal(int(asp * SR)), tone(0.25)]
    return np.concatenate(parts).astype(np.float32)


@pytest.mark.parametrize("asp,cat", [(0.012, "unaspirated"), (0.080, "aspirated")])
def test_vot_categories(asp, cat):
    r = measure_vot(_stop(asp), SR, 0.13, "front")
    assert r is not None and r.category == cat and abs(r.vot - asp) < 0.012


def test_closure_duration_medial():
    x = _stop(0.015, pre_vowel=0.15, closure=0.18)
    r = measure_vot(x, SR, 0.33, "front")
    assert abs(measure_closure(x, SR, r.burst_time) - 0.18) < 0.025


def test_vot_returns_none_without_landmarks():
    assert measure_vot(tone(0.5), SR, 0.25, "front") is None


def test_vowel_onset_after_silence_is_not_a_burst():
    x = np.concatenate([silence(0.2), tone(0.4)])
    assert measure_vot(x, SR, 0.2, "front") is None


# ---------------- reconciliation ----------------
PROFILES = load_profiles(PROJECT_DATA / "profiles.json")
PAIRS = all_variant_pairs(PROFILES)


def _reconcile(audio_phones, lexical, profile="standard", cfg=None):
    be = FakeAcoustic(audio_phones)
    rec = Reconciler(be, cfg or ReconcileConfig(), PROFILES[profile])
    return rec.run(be.emissions(None), slots_from_phonemes(lexical, PAIRS), silence(0.5), SR, "test")


def test_matching_audio_confirms_dictionary_form():
    r = _reconcile(["pʰ", "ə", "l"], ["pʰ", "ə", "l"])
    assert [s.chosen for s in r.segments] == ["pʰ", "ə", "l"]
    asp = next(d for d in r.segments[0].dims if d.dimension == "aspiration")
    assert asp.status == "supported" and asp.margin > 2


def test_unaspirated_realisation_reported_not_corrected():
    r = _reconcile(["p", "ə", "l"], ["pʰ", "ə", "l"])
    assert r.segments[0].chosen == "p" and r.segments[0].status == "acoustic_variant"


def test_retroflex_vs_dental_from_audio():
    r = _reconcile(["ʈ", "aː", "l"], ["t̪", "ɑː", "l"])
    assert r.segments[0].chosen == "ʈ"


def test_regional_variant_needs_extra_margin_outside_profile():
    weak = ReconcileConfig(margin=2.0, variant_penalty=1e6)
    r = _reconcile(["k", "ə", "l", "ə", "m"], ["q", "ə", "l", "ə", "m"], "standard", weak)
    assert r.segments[0].chosen == "q" and r.segments[0].status == "unresolved"
    r = _reconcile(["k", "ə", "l", "ə", "m"], ["q", "ə", "l", "ə", "m"], "punjabi", weak)
    assert r.segments[0].chosen == "k" and r.segments[0].status == "acoustic_variant"


def test_gemination_is_model_blind_and_flagged():
    r = _reconcile(["p", "ə", "k", "aː"], ["p", "ə", "kː", "ɑː"])
    k = r.segments[2]
    assert any(d.dimension == "gemination" and d.status == "blind" for d in k.dims)
    assert k.status == "model_blind"


def test_missing_stop_is_not_reported_as_elision():
    r = _reconcile(["ə", "l"], ["pʰ", "ə", "l"])
    assert not r.reliable and "segment_not_detected" in r.reasons
    assert r.segments[0].status == "unresolved"


def test_h_deletion_is_an_accepted_realisation():
    r = _reconcile(["n", "ə", "iː"], ["n", "ə", "h", "iː"])
    assert r.segments[2].chosen == "" and r.segments[2].status == "not_realised"
    assert "segment_not_detected" not in r.reasons


def test_garbage_audio_is_refused():
    r = _reconcile(["ʃ", "uː", "ʒ", "iː", "f"], ["pʰ", "ə", "l"])
    assert not r.reliable and r.phonetic is None


# ---------------- pipeline ----------------
def _pipeline(tmp_path, asr_text, phones, lexicon_rows=None):
    lex = Lexicon(tmp_path / "lex.tsv")
    for w, ph in (lexicon_rows or []):
        lex.add(Entry(w, ph))
    cfg = Config(data_dir=str(tmp_path), asr_backend="manual")
    cfg.skip_asr_with_expected_word = False   # exercise recognition; fast mode has its own tests
    cfg.reconcile.min_confidence = 0.3     # fake model cannot test gemination; lenient for unit tests
    cfg.reconcile.max_unresolved_fraction = 0.5
    p = Pipeline(cfg, asr=FakeASR(asr_text), acoustic=FakeAcoustic(phones), lexicon=lex)
    p.load()
    return p


def test_pipeline_ok_result(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["pʰ", "ə", "l"], [("پھل", "pʰəl")])
    r = p.process(word_like(), SR)
    assert r.status == "ok", (r.message, r.warnings)
    assert r.recognized_word == "پھل" and r.phonetic == "[pʰəl]" and r.phonemic == "/pʰəl/"
    assert r.processing_ms > 0 and "lexicon_unverified" in r.warnings


def test_pipeline_rejects_multiword_asr(tmp_path):
    p = _pipeline(tmp_path, "یہ کتاب", ["k"], [("کتاب", "kɪt̪ɑːb")])
    r = p.process(word_like(), SR)
    assert r.status == "refused" and r.message == "Please speak only one Urdu word at a time."


def test_pipeline_rejects_pause_separated_words(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["pʰ", "ə", "l"], [("پھل", "pʰəl")])
    assert p.process(word_like(gap=0.7), SR).message == MESSAGES["multiple_words"]


def test_pipeline_joins_spaced_compound_in_lexicon(tmp_path):
    p = _pipeline(tmp_path, "خوب صورت", ["x", "uː", "b", "s", "uː", "r", "ə", "t̪"], [("خوبصورت", "xuːbsuːrət̪")])
    r = p.process(word_like(), SR)
    assert "joined_tokens" in r.warnings and r.recognized_word == "خوبصورت"


def test_pipeline_non_urdu_script_refused(tmp_path):
    p = _pipeline(tmp_path, "kitab", ["k"])
    assert p.process(word_like(), SR).problem == "non_urdu_script"


def test_pipeline_homographs_resolved_by_audio(tmp_path):
    p = _pipeline(tmp_path, "پل", ["p", "ʊ", "l"], [("پل", "pəl"), ("پل", "pʊl")])
    r = p.process(word_like(), SR)
    assert r.status == "ok" and r.phonetic == "[pʊl]" and r.phonemic == "/pʊl/"


def test_pipeline_expected_word_mismatch_is_reported(tmp_path):
    p = _pipeline(tmp_path, "پل", ["pʰ", "ə", "l"], [("پل", "pəl"), ("پھل", "pʰəl")])
    r = p.process(word_like(), SR, expected_word="پھل")
    assert "asr_expected_mismatch" in r.warnings
    assert r.recognized_word == "پل" and r.analysed_word == "پھل"


def test_pipeline_unreliable_message(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["ʃ", "iː", "ʒ", "uː"], [("پھل", "pʰəl")])
    r = p.process(word_like(), SR)
    assert r.status == "refused" and r.message.endswith(UNRELIABLE)


def test_pipeline_reports_acoustic_model_failure(tmp_path):
    class Broken(FakeAcoustic):
        def load(self):
            raise RuntimeError("no model")
    cfg = Config(data_dir=str(tmp_path), asr_backend="manual")
    p = Pipeline(cfg, asr=FakeASR("پھل"), acoustic=Broken(), lexicon=Lexicon(tmp_path / "l.tsv"))
    p.load()
    r = p.process(word_like(), SR)
    assert r.status == "error" and r.problem == "models_unavailable"


def test_pipeline_rejects_concurrent_jobs(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["pʰ", "ə", "l"], [("پھل", "pʰəl")])
    p._job_lock.acquire()
    try:
        assert p.process(word_like(), SR).problem == "busy"
    finally:
        p._job_lock.release()


# ---------------- storage, export, evaluation ----------------
def test_manual_corrections_never_overwritten(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["pʰ", "ə", "l"], [("پھل", "pʰəl")])
    x = word_like()
    r = p.process(x, SR)
    st = Store(tmp_path / "data")
    rid = st.save(x, SR, r, speaker="S01")
    assert st.set_manual(rid, manual_ipa="pʰəl", by="MO")
    assert not st.set_manual(rid, manual_ipa="pəl")                 # refused without overwrite
    assert st.rows()[0]["manual_ipa"] == "pʰəl"
    rid2 = st.save(x, SR, r)                                        # machine save does not touch row 1
    assert st.rows()[0]["manual_ipa"] == "pʰəl" and st.rows()[1]["manual_ipa"] is None
    st.set_excluded(rid2, True)
    assert st.export_csv(tmp_path / "out.csv") == 1                 # excluded row left out
    assert st.export_xlsx(tmp_path / "out.xlsx", include_excluded=True) == 2
    assert (tmp_path / "data" / st.rows()[0]["audio_path"]).exists()
    txt = (tmp_path / "out.csv").read_text(encoding="utf-8-sig")
    assert "پھل" in txt and "[pʰəl]" in txt
    rep = evaluate(items_from_store(st))
    assert rep["summary"]["PER_on_transcribed"] == 0.0


def test_refused_results_saved_as_excluded(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["ʃ", "iː", "ʒ"], [("پھل", "pʰəl")])
    st = Store(tmp_path / "d")
    st.save(word_like(), SR, p.process(word_like(), SR))
    assert st.rows()[0]["excluded"] == 1 and st.rows(include_excluded=False) == []


def test_per_and_dimension_analysis(tmp_path):
    e, n, _ = per("pʰəl", "pəl")
    assert (e, n) == (1, 3)
    rep = evaluate([Item("pʰəl", "pəl", "پھل", "پھل", {"category": "aspirated"}),
                    Item("t̪ɑːl", "ʈɑːl", "تال", "ٹال", {"category": "dental"}),
                    Item("mɑ̃ː", None, "ماں", "", {"category": "nasal"}),
                    Item("pəkːɑː", "pəkːɑː", "پکا", "پکا", {})])
    s = rep["summary"]
    assert s["items"] == 4 and s["transcribed"] == 3 and s["coverage"] == 0.75
    assert s["exact_match_on_transcribed"] == pytest.approx(1 / 3, abs=1e-3)
    dims = {d["dimension"]: d["errors"] for d in rep["dimensions"]}
    assert dims == {"aspiration": 1, "retroflexion": 1}
    assert {"reference": "pʰ", "hypothesis": "p", "count": 1} in rep["confusions"]
    write_report(rep, tmp_path / "r.xlsx")
    assert (tmp_path / "r.xlsx").stat().st_size > 0


def test_manifest_evaluation_with_negative_cases(tmp_path):
    import soundfile as sf
    from urdu_ipa.evaluation import run_manifest
    sf.write(tmp_path / "phal.wav", word_like(), SR)
    sf.write(tmp_path / "silence.wav", silence(1.0), SR)
    sf.write(tmp_path / "two.wav", word_like(gap=0.7), SR)
    (tmp_path / "m.tsv").write_text(
        "audio\tword\treference_ipa\texpect\tcategory\n"
        "phal.wav\tپھل\tpʰəl\ttranscribe\taspirated\n"
        "silence.wav\t\t\trefuse\tsilence\n"
        "two.wav\t\t\trefuse\tmulti-word\n", encoding="utf-8")
    p = _pipeline(tmp_path, "پھل", ["pʰ", "ə", "l"], [("پھل", "pʰəl")])
    rep, _ = run_manifest(tmp_path / "m.tsv", p)
    s = rep["summary"]
    assert s["items"] == 1 and s["PER_on_transcribed"] == 0.0 and s["word_accuracy"] == 1.0
    assert s["negative_tests"] == 2 and s["negative_tests_correctly_refused"] == 2


def test_load_reports_progress_and_times(tmp_path):
    p = _pipeline(tmp_path, "پھل", ["pʰ"], [])
    p.load_errors.clear(); p.loaded.clear()
    msgs = []
    p.load(progress=msgs.append)
    assert any("phone model" in m for m in msgs) and any("speech recogniser" in m for m in msgs)
    assert set(p.load_times_s) == {"acoustic", "asr"}


# ---------------- word-final ہ ----------------
from urdu_ipa.measure import measure_final_breath
from urdu_ipa.orthography import ends_with_final_heh


def _word_end(breath):
    rng = np.random.default_rng(5)
    parts = [silence(0.3), tone(0.35)]
    if breath:
        parts.append((0.03 * rng.standard_normal(int(breath * SR))).astype(np.float32))
    parts.append(silence(0.3, seed=7))
    return np.concatenate(parts)


@pytest.mark.parametrize("breath,cat", [(0.0, "absent"), (0.12, "present")])
def test_final_breath_measurement(breath, cat):
    r = measure_final_breath(_word_end(breath), SR)
    assert r is not None and r.category == cat
    if breath:
        assert abs(r.breath_duration - breath) < 0.03


def test_final_heh_detection():
    assert ends_with_final_heh("جرگہ") and ends_with_final_heh("بچّہ")
    assert not ends_with_final_heh("شاہ") and not ends_with_final_heh("پانی")


@pytest.mark.parametrize("breath,expected", [(0.0, "[d͡ʒɪrɡɑ]"), (0.12, "[d͡ʒɪrɡɑh]")])
def test_jirga_final_h_decided_from_audio(tmp_path, breath, expected):
    p = _pipeline(tmp_path, "جرگہ", ["d͡ʒ", "ɪ", "r", "ɡ", "ɑ"], [("جرگہ", "d͡ʒɪrɡɑ")])
    r = p.process(_word_end(breath), SR)
    assert r.status == "ok", (r.message, r.warnings)
    assert r.phonetic == expected and r.phonemic == "/d͡ʒɪrɡɑ/"
    last = r.segments[-1]
    assert last["source"] == "final_h" and last["measurement"]["final_breath_ms"] is not None
    assert any(d["dimension"] == "final_aspiration" and d["status"] == "measured" for d in last["dims"])


def test_final_long_aa_has_no_final_h_slot(tmp_path):
    p = _pipeline(tmp_path, "کھانا", ["kʰ", "aː", "n", "aː"], [("کھانا", "kʰɑːnɑː")])
    r = p.process(word_like(), SR)
    assert r.phonetic == "[kʰɑːnɑː]" and all(s["source"] != "final_h" for s in r.segments)


# ---------------- packaging-related ----------------
def test_user_lexicon_overrides_builtin(tmp_path):
    base = tmp_path / "base.tsv"
    base.write_text("word\tphonemic\nپل\tpəl\nپل\tpʊl\nدل\td̪ɪl\n", encoding="utf-8")
    lex = Lexicon(tmp_path / "mine.tsv", base=base)
    assert len(lex.lookup("پل")) == 2 and len(lex) == 3
    lex.add(Entry("پل", "pʊl", gloss="bridge"))
    assert [e.phonemic for e in lex.lookup("پل")] == ["pʊl"]           # user row replaces built-in rows
    again = Lexicon(tmp_path / "mine.tsv", base=base)
    assert [e.phonemic for e in again.lookup("پل")] == ["pʊl"] and again.lookup("دل")


def test_builtin_lexicon_used_by_default(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    lex = Lexicon(cfg.resolved_lexicon(), base=cfg.builtin_lexicon)
    assert lex.lookup("جرگہ") and cfg.resolved_lexicon().name == "my_lexicon.tsv"


def test_resampy_replacement_matches_scipy_quality():
    from urdu_ipa import _resampy_compat as rc
    t = np.arange(16000) / 16000
    x = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 5000 * t)    # 5 kHz must be removed at 8 kHz
    y = rc.resample(x, 16000, 8000)
    assert len(y) == 8000
    spec = np.abs(np.fft.rfft(y[1000:7000]))
    f = np.fft.rfftfreq(6000, 1 / 8000)
    assert spec[np.argmin(abs(f - 440))] > 100 * spec[f > 3900].max()


def test_auto_asr_resolution(tmp_path, monkeypatch):
    import urdu_ipa.config as C
    cfg = Config(data_dir=str(tmp_path))
    monkeypatch.setattr(C, "BUNDLED_WHISPER", tmp_path / "nope")
    monkeypatch.setattr(C, "USER_WHISPER", tmp_path / "nope2")
    assert cfg.resolved_asr()[0] == "whisper"
    (tmp_path / "user").mkdir(); (tmp_path / "user" / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(C, "USER_WHISPER", tmp_path / "user")                 # lite app: copy in ~/UrduIPA
    assert cfg.resolved_asr() == ("faster-whisper", str(tmp_path / "user"))
    (tmp_path / "ct2").mkdir(); (tmp_path / "ct2" / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(C, "BUNDLED_WHISPER", tmp_path / "ct2")               # bundled copy wins
    assert cfg.resolved_asr() == ("faster-whisper", str(tmp_path / "ct2"))


def test_fast_mode_skips_recognition(tmp_path):
    class CountingASR(FakeASR):
        calls = 0
        def transcribe(self, audio16k):
            CountingASR.calls += 1
            return super().transcribe(audio16k)
    lex = Lexicon(tmp_path / "lex.tsv"); lex.add(Entry("پھل", "pʰəl"))
    cfg = Config(data_dir=str(tmp_path)); cfg.reconcile.min_confidence = 0.3
    p = Pipeline(cfg, asr=CountingASR("پل"), acoustic=FakeAcoustic(["pʰ", "ə", "l"]), lexicon=lex); p.load()
    r = p.process(word_like(), SR, expected_word="پھل")
    assert CountingASR.calls == 0 and "asr_skipped" in r.warnings
    assert r.status == "ok" and r.phonetic == "[pʰəl]" and r.recognized_word == "" and r.analysed_word == "پھل"
    assert r.timings_ms["asr_ms"] < 5
    r = p.process(word_like(), SR, expected_word="پھل", skip_asr=False)      # explicit full mode
    assert CountingASR.calls == 1 and r.recognized_word == "پل"
    p.process(word_like(), SR)                                               # no expected word: always recognises
    assert CountingASR.calls == 2


def test_fast_mode_still_refuses_wrong_word(tmp_path):
    lex = Lexicon(tmp_path / "lex.tsv"); lex.add(Entry("پھل", "pʰəl"))
    cfg = Config(data_dir=str(tmp_path))
    p = Pipeline(cfg, asr=FakeASR(""), acoustic=FakeAcoustic(["ʃ", "iː", "ʒ", "uː"]), lexicon=lex); p.load()
    r = p.process(word_like(), SR, expected_word="پھل")
    assert r.status == "refused" and r.phonetic is None


def test_refusal_still_reports_dictionary_pronunciations(tmp_path):
    lex = Lexicon(tmp_path / "lex.tsv"); lex.add(Entry("پتا", "pət̪ɑː", gloss="knowledge"))
    lex.add(Entry("پتا", "pət̪ːɑː", gloss="leaf"))
    cfg = Config(data_dir=str(tmp_path))
    p = Pipeline(cfg, asr=FakeASR(""), acoustic=FakeAcoustic(["ʃ", "iː", "ʒ", "uː"]), lexicon=lex); p.load()
    r = p.process(word_like(), SR, expected_word="پتا")
    assert r.status == "refused" and r.phonetic is None
    assert {d["phonemic"] for d in r.dictionary} == {"/pət̪ɑː/", "/pət̪ːɑː/"}
    st = Store(tmp_path / "d"); st.save(word_like(), SR, r)
    assert "/pət̪ɑː/" in st.rows()[0]["dictionary_ipa"]


def test_wiktionary_lexicon_layer(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    lex = Lexicon(cfg.resolved_lexicon(), base=cfg.builtin_lexicon)
    assert len(lex) > 7000 and lex.problems == []
    assert [e.phonemic for e in lex.lookup("سلام")] == ["səlɑːm"]
    assert [e.phonemic for e in lex.lookup("بچّہ")] == ["bət͡ʃːɑ"]               # hand-checked entry wins
    assert lex.lookup("کمرہ")[0].phonemic == "kəmrɑ"                          # final ہ -> short ɑ


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    d = tmp_path / "old"; (d / "audio").mkdir(parents=True)
    c = sqlite3.connect(d / "recordings.sqlite")
    c.execute("CREATE TABLE recordings (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, excluded INTEGER NOT NULL DEFAULT 0)")
    c.commit(); c.close()
    st = Store(d)
    cols = {r[1] for r in sqlite3.connect(d / "recordings.sqlite").execute("PRAGMA table_info(recordings)")}
    assert "dictionary_ipa" in cols


def test_standard_vs_strict_evidence(tmp_path):
    """A dictionary value the audio favours only slightly is 'consistent' (standard) or 'unresolved' (strict)."""
    from urdu_ipa.reconcile import DimEvidence
    for standard, ok_expected in (("standard", True), ("strict", False)):
        cfg = ReconcileConfig(evidence_standard=standard, margin=50.0)   # margin no fake score can reach
        r = _reconcile(["pʰ", "ə", "l"], ["pʰ", "ə", "l"], cfg=cfg)
        sts = {s.status for s in r.segments}
        if ok_expected:
            assert r.reliable and "consistent" in sts and "unresolved" not in sts
        else:
            assert not r.reliable and "unresolved" in sts


def test_contradicting_audio_is_not_consistent():
    cfg = ReconcileConfig(evidence_standard="standard")
    r = _reconcile(["ʃ", "uː", "ʒ", "iː", "f"], ["pʰ", "ə", "l"], cfg=cfg)
    assert not r.reliable and r.phonetic is None
