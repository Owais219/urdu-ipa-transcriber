import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from urdu_ipa.config import Config
from urdu_ipa.lexicon import Entry, Lexicon
from urdu_ipa.pipeline import Pipeline
from urdu_ipa.recorder import RecorderError, RecorderStatus
from urdu_ipa.storage import Store
from urdu_ipa.ui import MainWindow
import numpy as np
from tests.helpers import SR, FakeAcoustic, FakeASR, word_like


class FakeRecorder:
    def __init__(self, signal=None, fail=None):
        self.signal = word_like() if signal is None else signal
        self.fail = fail
        self.starts = 0
        self.recording = False
        self.lost = False

    @property
    def is_recording(self):
        return self.recording

    def start(self):
        if self.fail:
            raise RecorderError(self.fail)
        if self.recording:
            raise RecorderError("A recording is already in progress.")
        self.starts += 1
        self.recording = True

    def poll(self):
        return RecorderStatus(self.recording, 0.5, -20.0, True, "device_lost" if self.lost else None, False)

    def level_history(self):
        return [-30.0, -20.0]

    def stop(self):
        if not self.recording:
            raise RecorderError("Not recording.")
        self.recording = False
        return self.signal, SR

    def abort(self):
        self.recording = False


class FakePlayer:
    def __init__(self):
        self.played = 0

    def play(self, s, sr):
        self.played += 1

    def stop(self):
        pass


def wait_until(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        QCoreApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_window(tmp_path, recorder=None, asr_text="پھل", phones=("pʰ", "ə", "l")):
    lex = Lexicon(tmp_path / "lex.tsv")
    lex.add(Entry("پھل", "pʰəl"))
    cfg = Config(data_dir=str(tmp_path), asr_backend="manual")
    cfg.reconcile.min_confidence = 0.3
    p = Pipeline(cfg, asr=FakeASR(asr_text), acoustic=FakeAcoustic(list(phones)), lexicon=lex)
    p.load()
    rec = recorder or FakeRecorder()
    w = MainWindow(cfg, pipeline=p, recorder=rec, store=Store(tmp_path / "data"), player=FakePlayer(),
                   autoload=False)
    return w, rec


def test_full_cycle_and_controls(app, tmp_path):
    w, rec = make_window(tmp_path)
    assert w.state == w.IDLE and rec.starts == 0          # never records on launch
    w.start_recording()
    w.start_recording()                                   # duplicate click ignored
    assert rec.starts == 1 and w.state == w.RECORDING and not w.btn_record.isEnabled()
    w.stop_recording()
    assert wait_until(lambda: w.state == w.IDLE and w.result is not None)
    assert w.val["word"].text() == "پھل" and w.val["phonetic"].text() == "[pʰəl]"
    assert w.val["phonemic"].text() == "/pʰəl/" and w.val["time"].text().endswith(")")
    w.copy_ipa()
    assert QApplication.clipboard().text() == "[pʰəl]"
    w.replay()
    assert w.player.played == 1
    w.save_result()
    assert w.saved_id is not None and not w.btn_save.isEnabled()
    w.manual.setText("pʰəl")
    w.save_correction()
    assert w.store.rows()[0]["manual_ipa"] == "pʰəl"
    w.record_next()                                       # clears and starts again without restart
    assert rec.starts == 2 and w.result is None and w.state == w.RECORDING
    w.stop_recording()
    assert wait_until(lambda: w.state == w.IDLE and w.result is not None)
    w.clear_result()
    assert w.result is None and w.val["phonetic"].text() == "—"
    w.close()


def test_repeated_recordings_stay_stable(app, tmp_path):
    w, rec = make_window(tmp_path)
    for i in range(8):
        w.start_recording()
        w.stop_recording()
        assert wait_until(lambda: w.state == w.IDLE and w.result is not None)
    assert rec.starts == 8 and w.engine.pool.activeThreadCount() == 0
    w.close()


def test_microphone_permission_error_shown(app, tmp_path):
    w, _ = make_window(tmp_path, FakeRecorder(fail="Could not open the microphone: permission denied"))
    w.start_recording()
    assert w.state == w.IDLE and "permission denied" in w.message.text()
    w.close()


def test_device_lost_during_recording(app, tmp_path):
    w, rec = make_window(tmp_path)
    w.start_recording()
    rec.lost = True
    w._poll()
    assert w.state == w.IDLE and "disconnected" in w.message.text() and not rec.recording
    w.close()


def test_refusal_message_and_no_copy(app, tmp_path):
    w, rec = make_window(tmp_path, asr_text="یہ پھل")
    w.start_recording()
    w.stop_recording()
    assert wait_until(lambda: w.result is not None)
    assert w.message.text() == "Please speak only one Urdu word at a time."
    assert not w.btn_copy.isEnabled() and w.val["phonetic"].text() == "—"
    w.close()


def test_stale_result_after_timeout_is_ignored(app, tmp_path):
    w, rec = make_window(tmp_path)
    w._pending_job = 99
    w._processing_timeout()
    assert w._pending_job is None and w.state == w.IDLE
    w._processed(99, object())                          # late result must not be displayed
    assert w.result is None
    w.close()


def test_recording_blocked_while_models_load(app, tmp_path):
    w, rec = make_window(tmp_path)
    w._models_ready = False
    w._set_state(w.IDLE)
    assert not w.btn_record.isEnabled() and "Loading speech models" in w.lbl_state.text()
    w.start_recording()
    assert rec.starts == 0 and "still loading" in w.message.text()
    w._models_loaded({"errors": {}, "models": {}})
    assert w.btn_record.isEnabled()
    w.close()


def test_roman_expected_word_rejected(app, tmp_path):
    w, rec = make_window(tmp_path)
    w.expected.setText("Jirga")
    w.start_recording()
    assert rec.starts == 0 and "Urdu script" in w.message.text()
    w.expected.setText("جرگہ")
    w.start_recording()
    assert rec.starts == 1
    w.close()


def test_timeout_starts_only_when_job_runs(app, tmp_path):
    w, rec = make_window(tmp_path)
    w.start_recording()
    w.stop_recording()
    assert wait_until(lambda: w.result is not None)
    assert not w.timeout_timer.isActive()
    w.close()


def test_single_audio_file_uses_filename_as_expected_word(app, tmp_path, monkeypatch):
    import soundfile as sf
    from PySide6.QtWidgets import QFileDialog
    f = tmp_path / "پھل.wav"
    sf.write(f, word_like(), SR)
    w, rec = make_window(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([str(f)], "")))
    w.open_audio_files()
    assert wait_until(lambda: w.state == w.IDLE and w.result is not None)
    assert w.result.expected_word == "پھل" and w.result.phonetic == "[pʰəl]"
    assert w.samples is not None and w.btn_replay.isEnabled() and w.btn_save.isEnabled()
    assert rec.starts == 0
    w.close()


def test_batch_of_audio_files(app, tmp_path, monkeypatch):
    import soundfile as sf
    from PySide6.QtWidgets import QFileDialog
    paths = []
    for name, sig in (("پھل_1.wav", word_like()), ("پھل_2.flac", word_like()), ("silence.wav", np.zeros(SR))):
        sf.write(tmp_path / name, sig, SR)
        paths.append(str(tmp_path / name))
    (tmp_path / "broken.wav").write_bytes(b"not audio")
    paths.append(str(tmp_path / "broken.wav"))
    w, rec = make_window(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    w.open_audio_files()
    b = w.batch
    assert b is not None
    assert wait_until(lambda: b.current is None and b.idx >= 4, timeout=20)
    sts = [r["result"].status if r["result"] is not None else "error" for r in b.rows]
    assert sts == ["ok", "ok", "refused", "error"]
    b.save_all()
    assert len(w.store.rows()) == 3 and "file: پھل_1.wav" in w.store.rows()[0]["notes"]
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "out.csv"), "")))
    b.export_csv()
    text = (tmp_path / "out.csv").read_text(encoding="utf-8-sig")
    assert "[pʰəl]" in text and "broken.wav" in text
    b.close()
    assert wait_until(lambda: w.batch is None and w.state == w.IDLE)
    w.close()


def test_version_in_title(app, tmp_path):
    from urdu_ipa import __version__
    w, _ = make_window(tmp_path)
    assert __version__ in w.windowTitle()
    w.close()
