"""End-to-end self-test of an installed copy (used by the Mac build before packaging).

    UrduIPA --selftest [--wav file.wav --word پھل] [--json out.json]

Checks that every bundled model loads offline, that a recording goes through the whole
pipeline, that the window can be built, and reports load and processing times. Exit code
is non-zero on any failure. It does not measure accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time


def _synthetic_word():
    import numpy as np
    sr = 16000
    t = np.arange(int(0.45 * sr)) / sr
    voiced = 0.3 * sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 14))
    rng = np.random.default_rng(0)
    burst = 0.5 * rng.standard_normal(int(0.006 * sr))
    asp = 0.06 * rng.standard_normal(int(0.07 * sr))
    sil = 0.0005 * rng.standard_normal(int(0.4 * sr))
    return np.concatenate([sil, burst, asp, voiced, sil]).astype("float32"), sr


def run(argv) -> int:
    ap = argparse.ArgumentParser(prog="UrduIPA --selftest")
    ap.add_argument("--wav")
    ap.add_argument("--word", default="پھل")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    report: dict = {"python": sys.version.split()[0], "ok": False}
    failures: list[str] = []
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tmp = tempfile.mkdtemp(prefix="urduipa-selftest-")
    try:
        from .config import FROZEN, Config, find_whisper
        from .pipeline import Pipeline
        report["frozen"] = FROZEN
        report["whisper_model_dir"] = str(find_whisper())
        cfg = Config(data_dir=tmp)
        report["asr_backend"] = list(cfg.resolved_asr())
        t = time.perf_counter()
        p = Pipeline(cfg)
        errors = p.load()
        report["model_load_s"] = round(time.perf_counter() - t, 2)
        report["load_times_s"] = p.load_times_s
        report["models"] = p.models()
        report["lexicon_entries"] = len(p.lexicon)
        report["lexicon_problems"] = p.lexicon.problems
        if errors:
            failures += [f"model {k}: {v}" for k, v in errors.items()]
        if a.wav:
            import soundfile as sf
            x, sr = sf.read(a.wav, always_2d=False)
        else:
            x, sr = _synthetic_word()
        full = p.process(x, sr, expected_word=a.word, skip_asr=False)       # with Whisper
        report["full_mode_ms"] = full.processing_ms
        report["full_mode_timings_ms"] = full.timings_ms
        report["full_mode_asr_text"] = full.asr_text
        if full.status == "error":
            failures.append(f"full-mode pipeline error: {full.message} {full.warnings}")
        runs = [p.process(x, sr, expected_word=a.word, skip_asr=True) for _ in range(3)]   # fast mode
        r = runs[-1]
        report["result"] = {k: getattr(r, k) for k in ("status", "problem", "message", "recognized_word", "asr_text",
                                                         "word_confidence", "phonetic", "phonemic",
                                                         "phonetic_confidence", "warnings")}
        report["processing_ms_first"] = runs[0].processing_ms
        report["processing_ms_warm"] = r.processing_ms
        report["timings_ms_warm"] = r.timings_ms
        if r.status == "error":
            failures.append(f"pipeline error: {r.message} {r.warnings}")
        if not r.segments and r.status == "ok":
            failures.append("no segment evidence")
        # storage + export
        from .storage import Store
        st = Store(tmp)
        rid = st.save(x, sr, r)
        st.set_manual(rid, manual_ipa="pʰəl")
        st.export_csv(os.path.join(tmp, "x.csv"), include_excluded=True)
        st.export_xlsx(os.path.join(tmp, "x.xlsx"), include_excluded=True)
        # user interface
        from PySide6.QtWidgets import QApplication
        from .ui import MainWindow
        app = QApplication.instance() or QApplication([])
        w = MainWindow(cfg, pipeline=p, autoload=False)
        w.show()
        app.processEvents()
        report["ui_title"] = w.windowTitle()
        w.close()
        # microphone layer (no device on build machines: only check that PortAudio loads)
        try:
            from .recorder import list_input_devices
            report["input_devices"] = len(list_input_devices())
        except Exception as e:  # noqa: BLE001
            failures.append(f"audio system: {e}")
    except Exception as e:  # noqa: BLE001
        import traceback
        failures.append(f"{type(e).__name__}: {e}")
        report["traceback"] = traceback.format_exc()
    report["failures"] = failures
    report["ok"] = not failures
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(text)
    return 0 if report["ok"] else 1


class _ReplayRecorder:
    """Stands in for the microphone: every 'recording' returns the same audio."""

    def __init__(self, samples, sr):
        self.samples, self.sr, self.recording = samples, sr, False

    @property
    def is_recording(self):
        return self.recording

    def start(self):
        self.recording = True

    def poll(self):
        from .recorder import RecorderStatus
        return RecorderStatus(self.recording, 1.0, -20.0, True, None, False)

    def level_history(self):
        import random
        return [random.uniform(-60, -5) for _ in range(100)]

    def stop(self):
        self.recording = False
        return self.samples, self.sr

    def abort(self):
        self.recording = False


def run_ui_stress(argv) -> int:
    """Drive the real window with the real engine process: record -> process -> show, N times,
    while the level meter and evidence table are redrawn continuously; then transcribe audio files.
    Exit code 0 only if every job returned and the engine never stopped."""
    ap = argparse.ArgumentParser(prog="UrduIPA --ui-stress")
    ap.add_argument("n", type=int, nargs="?", default=25)
    ap.add_argument("--wav")
    ap.add_argument("--word", default="پھل")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from .config import Config
    from .ui import MainWindow
    if a.wav:
        import soundfile as sf
        x, sr = sf.read(a.wav, always_2d=False)
    else:
        x, sr = _synthetic_word()
    tmp = tempfile.mkdtemp(prefix="urduipa-stress-")
    cfg = Config(data_dir=tmp)
    app = QApplication.instance() or QApplication([])
    log_path = os.path.join(tmp, "stress.log")
    w = MainWindow(cfg, recorder=_ReplayRecorder(x, sr), log_path=log_path)
    w.show()
    st = {"results": 0, "crashes": [], "statuses": [], "files": [], "t0": time.time(), "phase": "record"}
    w.engine.crashed.connect(lambda t: st["crashes"].append(t))
    w.btn_evidence.setChecked(True)
    anim = QTimer()
    anim.timeout.connect(lambda: w.meter.set_values(w.recorder.level_history()))
    anim.start(16)

    def tick():
        if time.time() - st["t0"] > 900:
            app.exit(3)
            return
        if not w._models_ready or w.state != w.IDLE:
            return
        if st["phase"] == "record":
            if st["results"] >= a.n:
                st["phase"] = "files"
                return
            use_word = st["results"] % 2 == 0 or w.engine_info.get("asr_manual")
            w.expected.setText(a.word if use_word else "")     # alternate fast mode and Whisper
            w.start_recording()
            QTimer.singleShot(200, w.stop_recording)
        elif st["phase"] == "files":
            if not a.files:
                app.exit(0)
                return
            path = a.files.pop(0)
            w.expected.setText("")
            prof, fast, std = w._job_args()
            w._reset_result(keep_audio=False)
            w._set_state(w.PROCESSING)
            w._pending_kind = "file"
            w._pending_job = w.engine.process_file(path, "", prof, fast, std)
            st["files"].append(path)

    def on_result(job_id, r, samples, sr_):
        st["results"] += 1
        st["statuses"].append([r.status, r.problem, r.recognized_word, r.phonetic, round(r.processing_ms)])
        if samples is not None:
            st.setdefault("file_results", []).append([r.status, r.recognized_word, r.phonetic, len(samples), sr_])

    w.engine.result.connect(on_result)
    drive = QTimer()
    drive.timeout.connect(tick)
    drive.start(100)
    code = app.exec()
    w.close()
    report = {"requested": a.n, "results": st["results"], "crashes": st["crashes"], "exit": code,
              "files": st["files"], "file_results": st.get("file_results", []), "statuses": st["statuses"][-6:],
              "seconds": round(time.time() - st["t0"], 1)}
    report["ok"] = code == 0 and not st["crashes"] and st["results"] >= a.n + len(st["files"])
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(text)
    try:
        print(open(log_path, encoding="utf-8").read()[-3000:])
    except OSError:
        pass
    return 0 if report["ok"] else 1
