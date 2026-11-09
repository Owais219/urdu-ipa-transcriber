"""PySide6 user interface.

Responsiveness and robustness rules: all models and numerical processing run in a
separate engine process (engine.SubprocessEngine); this process only records audio and
draws. The recorder is polled by one QTimer that runs only while recording; every job has
an id so a late (timed-out) result is discarded. If the engine process dies, the window
reports it and a new engine is started.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QGuiApplication, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from . import __version__, ipa
from .audio_io import AUDIO_EXTENSIONS, expected_word_from_filename
from .config import Config, load_config
from .engine import EngineBase, InProcessEngine, SubprocessEngine
from .pipeline import Pipeline, Result
from .reconcile import load_profiles
from .orthography import script_fraction
from .recorder import Player, Recorder, RecorderError
from .storage import Store

URDU_FONTS = ["Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", "Noto Naskh Arabic", "Segoe UI", "Arial"]
IPA_FONTS = ["Charis SIL", "Doulos SIL", "Gentium Plus", "Noto Sans", "Segoe UI", "DejaVu Sans", "Arial Unicode MS"]
STATUS_TEXT = {"confirmed": "confirmed by recording", "consistent": "consistent with recording (not confirmed)", "acoustic_variant": "realised differently from dictionary",
               "model_blind": "partly unverifiable by model", "unresolved": "unresolved",
               "not_realised": "not realised"}
STATUS_COLOR = {"confirmed": "#1b7f3b", "consistent": "#4d7c0f", "acoustic_variant": "#1d4ed8", "model_blind": "#8a6d00",
                "unresolved": "#b45309", "not_realised": "#6b7280"}


def font(families, size, bold=False):
    f = QFont()
    f.setFamilies(families)
    f.setPointSize(size)
    f.setBold(bold)
    return f


class LevelMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.values: list[float] = []
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_values(self, v):
        self.values = v
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#f3f4f6"))
        if not self.values:
            p.end()
            return
        n = 100
        w = self.width() / n
        h = self.height()
        vals = self.values[-n:]
        for i, db in enumerate(vals):
            frac = min(1.0, max(0.0, (db + 70) / 70))
            bh = max(1, frac * (h - 4))
            colour = QColor("#dc2626") if db > -3 else QColor("#2563eb")
            p.fillRect(int(i * w), int((h - bh) / 2), max(1, int(w) - 1), int(bh), colour)
        p.end()


class MainWindow(QMainWindow):
    IDLE, LOADING, RECORDING, PROCESSING = "idle", "loading", "recording", "processing"

    def __init__(self, cfg: Config, pipeline: Pipeline | None = None, recorder: Recorder | None = None,
                 store: Store | None = None, player: Player | None = None, autoload: bool = True,
                 engine: EngineBase | None = None, log_path: str | None = None):
        super().__init__()
        self.cfg = cfg
        if engine is None:
            engine = InProcessEngine(pipeline, autoload=autoload) if pipeline is not None \
                else SubprocessEngine(cfg, log_path)
        self.engine = engine
        self.profiles = load_profiles(cfg.profiles_path)
        self.recorder = recorder or Recorder(cfg.recording)
        self.store = store or Store(cfg.data_path)
        self.player = player or Player()
        self.samples: np.ndarray | None = None
        self.sr = 0
        self.result: Result | None = None
        self.saved_id: int | None = None
        self._pending_job: int | None = None
        self._pending_kind = ""
        self._models_ready = False
        self.engine_info: dict = {}
        self.batch: BatchDialog | None = None
        self.state = self.IDLE

        self.setWindowTitle(f"Urdu IPA {__version__} — Urdu Phonetic Transcription Tool")
        self.resize(880, 900)
        self._build()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(50)
        self.poll_timer.timeout.connect(self._poll)
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self._processing_timeout)
        self.engine.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.engine.loaded.connect(self._models_loaded)
        self.engine.result.connect(self._processed)
        self.engine.failed.connect(self._failed)
        self.engine.done.connect(self._job_done)
        self.engine.crashed.connect(self._engine_crashed)
        self._t_load = time.monotonic()
        self.statusBar().showMessage("Loading speech models…")
        self.engine.start()
        self._set_state(self.IDLE)

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(20, 16, 20, 12)
        title = QLabel(f"Urdu Phonetic Transcription Tool  <span style='color:#6b7280;font-size:12pt'>"
                       f"version {__version__}</span>")
        title.setFont(font(["Segoe UI", "Noto Sans", "Arial"], 18, True))
        sub = QLabel("Single-Word Speech-to-IPA Transcription")
        sub.setStyleSheet("color:#4b5563")
        lay.addWidget(title)
        lay.addWidget(sub)

        opts = QHBoxLayout()
        self.expected = QLineEdit()
        self.expected.setPlaceholderText("متوقع لفظ (اختیاری)")
        self.expected.setLayoutDirection(Qt.RightToLeft)
        self.expected.setFont(font(URDU_FONTS, 14))
        self.expected.setToolTip("Optional: the word the speaker was asked to say (elicitation). "
                                 "It is scored alongside the recogniser's output, never silently substituted.")
        self.profile = QComboBox()
        for key, p in self.profiles.items():
            self.profile.addItem(p.get("label", key), key)
        self.profile.setToolTip("Pronunciation profile: which regional variants are expected (not treated as errors).")
        self.speaker = QLineEdit()
        self.speaker.setPlaceholderText("speaker ID")
        self.speaker.setMaximumWidth(120)
        opts.addWidget(QLabel("Expected word:"))
        opts.addWidget(self.expected, 2)
        opts.addWidget(QLabel("Profile:"))
        opts.addWidget(self.profile, 2)
        opts.addWidget(self.speaker)
        self.chk_fast = QCheckBox("Fast mode")
        self.chk_fast.setChecked(self.cfg.skip_asr_with_expected_word)
        self.chk_fast.setToolTip("When an expected word is typed, skip speech recognition (Whisper) and analyse the "
                                 "typed word directly. Results arrive in under a second. The pronunciation is still "
                                 "decided from the audio. With the box empty, recognition always runs.")
        opts.addWidget(self.chk_fast)
        lay.addLayout(opts)

        rec = QGroupBox("Recording")
        rl = QVBoxLayout(rec)
        row = QHBoxLayout()
        self.btn_record = QPushButton("●  Start Recording")
        self.btn_record.setMinimumHeight(64)
        self.btn_record.setFont(font(["Segoe UI", "Noto Sans", "Arial"], 14, True))
        self.btn_record.setStyleSheet("QPushButton{background:#b91c1c;color:white;border-radius:10px}"
                                      "QPushButton:disabled{background:#d1d5db;color:#6b7280}")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(64)
        self.btn_stop.setMaximumWidth(140)
        self.btn_file = QPushButton("Open Audio\nFile(s)…")
        self.btn_file.setMinimumHeight(64)
        self.btn_file.setMaximumWidth(140)
        self.btn_file.setToolTip("Transcribe recorded audio files (WAV, MP3, M4A, FLAC, AIFF, OGG). One word per file. "
                                 "If a file is named with the Urdu word (e.g. جرگہ.m4a) that name is used as the expected word.")
        row.addWidget(self.btn_record, 3)
        row.addWidget(self.btn_stop, 1)
        row.addWidget(self.btn_file, 1)
        rl.addLayout(row)
        self.meter = LevelMeter()
        rl.addWidget(self.meter)
        srow = QHBoxLayout()
        self.lbl_state = QLabel("Ready")
        self.lbl_duration = QLabel("0.0 s")
        srow.addWidget(self.lbl_state, 1)
        srow.addWidget(self.lbl_duration)
        rl.addLayout(srow)
        lay.addWidget(rec)

        res = QGroupBox("Result")
        g = QGridLayout(res)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color:#92400e;font-weight:bold")
        g.addWidget(self.message, 0, 0, 1, 2)
        names = [("word", "Recognized Urdu Word"), ("analysed", "Analysed as"), ("phonetic", "IPA Phonetic Transcription"),
                 ("phonemic", "IPA Phonemic Transcription"), ("narrow", "Narrow Phonetic Transcription"),
                 ("dictionary", "Dictionary Pronunciation\n(not confirmed by this recording)"),
                 ("wconf", "Word Recognition Confidence"), ("pconf", "Phonetic Transcription Confidence"),
                 ("time", "Processing Time")]
        self.val: dict[str, QLabel] = {}
        self.key: dict[str, QLabel] = {}
        for r, (k, label) in enumerate(names, start=1):
            kl = QLabel(label)
            kl.setStyleSheet("color:#374151")
            v = QLabel("—")
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            v.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if k == "word" or k == "analysed":
                v.setFont(font(URDU_FONTS, 20))
            elif k == "dictionary":
                v.setFont(font(IPA_FONTS, 16))
                v.setWordWrap(True)
                v.setStyleSheet("color:#6b7280")
            elif k in ("phonetic", "phonemic", "narrow"):
                v.setFont(font(IPA_FONTS, 20 if k == "phonetic" else 16))
            g.addWidget(kl, r, 0)
            g.addWidget(v, r, 1)
            self.val[k], self.key[k] = v, kl
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#6b7280;font-size:11px")
        g.addWidget(self.warn, len(names) + 1, 0, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addWidget(res)

        self.btn_evidence = QToolButton()
        self.btn_evidence.setText("Show segment evidence")
        self.btn_evidence.setCheckable(True)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Expected", "Realised", "Evidence status", "Contrasts tested",
                                              "Measurement", "Time (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFont(font(IPA_FONTS, 11))
        self.table.setMinimumHeight(180)
        self.table.setWordWrap(True)
        self.table.horizontalHeaderItem(3).setToolTip(
            "Margin = how many nats (log-likelihood units) worse the alternative scores than the chosen value. "
            "'supported' needs at least the configured margin; 'blind' = the phone model cannot tell them apart.")
        self.table.setVisible(False)
        self.btn_evidence.toggled.connect(lambda on: (self.table.setVisible(on),
                                                      self.btn_evidence.setText("Hide segment evidence" if on else "Show segment evidence")))
        lay.addWidget(self.btn_evidence)
        lay.addWidget(self.table, 1)

        btns = QHBoxLayout()
        self.btn_replay = QPushButton("Replay Audio")
        self.btn_next = QPushButton("Record Next Word")
        self.btn_copy = QPushButton("Copy IPA")
        self.btn_clear = QPushButton("Clear Result")
        self.btn_save = QPushButton("Save to Dataset")
        for b in (self.btn_replay, self.btn_next, self.btn_copy, self.btn_clear, self.btn_save):
            btns.addWidget(b)
        lay.addLayout(btns)

        corr = QHBoxLayout()
        self.manual = QLineEdit()
        self.manual.setFont(font(IPA_FONTS, 13))
        self.manual.setPlaceholderText("Verified IPA (manual correction), e.g. pʰəl")
        self.chk_exclude = QCheckBox("Exclude from research dataset")
        self.btn_correct = QPushButton("Save Correction")
        corr.addWidget(self.manual, 3)
        corr.addWidget(self.chk_exclude)
        corr.addWidget(self.btn_correct)
        lay.addLayout(corr)
        self.setCentralWidget(root)

        self.btn_record.clicked.connect(self.start_recording)
        self.btn_file.clicked.connect(self.open_audio_files)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_replay.clicked.connect(self.replay)
        self.btn_next.clicked.connect(self.record_next)
        self.btn_copy.clicked.connect(self.copy_ipa)
        self.btn_clear.clicked.connect(self.clear_result)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_correct.clicked.connect(self.save_correction)
        self.chk_exclude.toggled.connect(self._exclude_toggled)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._toggle_recording)

        m = self.menuBar()
        fm = m.addMenu("&File")
        a = QAction("Open Audio Files…", self)
        a.setShortcut(QKeySequence.Open)
        a.triggered.connect(self.open_audio_files)
        fm.addAction(a)
        for text, fn in (("Export CSV…", self.export_csv), ("Export Excel…", self.export_xlsx),
                         ("Open Data Folder", self.open_data_folder)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            fm.addAction(a)
        fm.addSeparator()
        q = QAction("Quit", self)
        q.triggered.connect(self.close)
        fm.addAction(q)
        tm = m.addMenu("&Tools")
        a = QAction("Evaluate Test Set…", self)
        a.triggered.connect(self.evaluate)
        tm.addAction(a)
        self.act_strict = QAction("Strict Evidence Standard", self, checkable=True)
        self.act_strict.setChecked(self.cfg.reconcile.evidence_standard == "strict")
        self.act_strict.setToolTip("Strict: give a transcription only when every tested contrast is confirmed by "
                                   "the recording. Standard: also accept dictionary values the recording favours "
                                   "without clearly confirming them (marked 'consistent').")
        self.act_strict.toggled.connect(self._toggle_strict)
        tm.addAction(self.act_strict)
        a = QAction("Add Word to Lexicon…", self)
        a.triggered.connect(self.add_lexicon_entry)
        tm.addAction(a)
        hm = m.addMenu("&Help")
        a = QAction("About and Limitations", self)
        a.triggered.connect(self.about)
        hm.addAction(a)
        self.statusBar().showMessage("Starting…")

    # ------------------------------------------------------------- state
    def _set_state(self, st):
        self.state = st
        rec = st == self.RECORDING
        busy = st == self.PROCESSING
        have = self.samples is not None and self.samples.size > 0
        ready = self._models_ready
        self.btn_record.setEnabled(not rec and not busy and ready)
        self.btn_stop.setEnabled(rec)
        self.btn_replay.setEnabled(have and not rec)
        self.btn_next.setEnabled(not rec and not busy and ready)
        self.btn_file.setEnabled(not rec and not busy and ready)
        self.btn_copy.setEnabled(bool(self.result and self.result.phonetic) and not rec)
        self.btn_clear.setEnabled(not rec and not busy and (self.result is not None or have))
        self.btn_save.setEnabled(have and self.result is not None and self.saved_id is None and not busy and not rec)
        self.btn_correct.setEnabled(self.saved_id is not None and not rec)
        self.chk_exclude.setEnabled(self.saved_id is not None and not rec)
        self.expected.setEnabled(not rec and not busy)
        self.profile.setEnabled(not rec and not busy)
        self.lbl_state.setText({self.IDLE: "Ready" if ready else
                                "Loading speech models… recording will be available in a few seconds.",
                                self.RECORDING: "Recording… speak one Urdu word",
                                self.PROCESSING: "Processing…"}.get(st, st))

    @Slot(dict)
    def _models_loaded(self, info):
        self.engine_info = info
        errors = info.get("errors", {})
        m = info.get("models", {})
        self._models_ready = "acoustic" not in errors
        if errors:
            hint = (" — the speech recogniser is unavailable; type the expected word in Urdu script before each recording"
                    if "asr" in errors and self._models_ready else "")
            self.statusBar().showMessage("Model problems: " + "; ".join(f"{k}: {v}" for k, v in errors.items()) + hint)
        else:
            self.statusBar().showMessage(f"Ready · version {__version__} · ASR: {m.get('asr')} · acoustic: {m.get('acoustic')} · "
                                         f"lexicon entries: {info.get('lexicon_entries')} · models loaded in {info.get('load_s')} s")
        if self.state == self.IDLE:
            self._set_state(self.IDLE)

    @Slot(str)
    def _engine_crashed(self, text):
        self._models_ready = False
        if self.state == self.PROCESSING:
            self._set_state(self.IDLE)
        self._show_error(text)
        if self.batch is not None:
            self.batch.engine_crashed(text)

    # --------------------------------------------------------- recording
    def _toggle_recording(self):
        if self.state == self.RECORDING:
            self.stop_recording()
        elif self.btn_record.isEnabled():
            self.start_recording()

    def start_recording(self):
        if self.state in (self.RECORDING, self.PROCESSING):
            return
        if not self._models_ready:
            self._show_error("The speech models are still loading. Please wait until the status bar says Ready.")
            return
        exp = self.expected.text().strip()
        if exp and script_fraction(exp) < 0.9:
            self._show_error(f"Type the expected word in Urdu script (for example جرگہ), not in Roman letters (“{exp}”).")
            return
        if not exp and self.engine_info.get("asr_manual"):
            self._show_error("No speech recogniser is active: type the expected word in Urdu script first.")
            return
        self.player.stop()
        try:
            self.recorder.start()
        except RecorderError as e:
            self._show_error(str(e))
            return
        self._reset_result(keep_audio=False)
        self._set_state(self.RECORDING)
        self.poll_timer.start()

    def _poll(self):
        st = self.recorder.poll()
        self.meter.set_values(self.recorder.level_history())
        self.lbl_duration.setText(f"{st.elapsed:.1f} s")
        if st.should_stop == "device_lost":
            self.poll_timer.stop()
            self.recorder.abort()
            self._set_state(self.IDLE)
            self._show_error("The microphone stopped delivering audio (disconnected?). Please reconnect it and record again.")
        elif st.should_stop:
            self.stop_recording()

    def stop_recording(self):
        if self.state != self.RECORDING:
            return
        self.poll_timer.stop()
        try:
            samples, sr = self.recorder.stop()
        except RecorderError as e:
            self._set_state(self.IDLE)
            self._show_error(str(e))
            return
        self.samples, self.sr = samples, sr
        self._process()

    def _job_args(self):
        return (self.profile.currentData() or "standard", self.chk_fast.isChecked(),
                self.cfg.reconcile.evidence_standard)

    def _process(self):
        self._set_state(self.PROCESSING)
        self.message.setText("Processing…")
        profile, fast, std = self._job_args()
        self._pending_kind = "process"
        self._pending_job = self.engine.process(self.samples, self.sr, self.expected.text(), profile, fast, std)
        self.timeout_timer.start(int(self.cfg.processing_timeout * 1000))

    @Slot(int, object, object, object)
    def _processed(self, job_id, result: Result, samples=None, sr=None):
        if self.batch is not None and self.batch.owns(job_id):
            self.batch.on_result(job_id, result, samples, sr)
            return
        if job_id != self._pending_job:
            return                         # stale (timed out) result
        self._pending_job = None
        self.timeout_timer.stop()
        if samples is not None:            # audio-file job: keep the audio for replay and saving
            self.samples, self.sr = np.asarray(samples, dtype=np.float32), int(sr)
        self.result = result
        self.saved_id = None
        self._show_result(result)
        self._set_state(self.IDLE)

    @Slot(int, str)
    def _failed(self, job_id, err):
        if self.batch is not None and self.batch.owns(job_id):
            self.batch.on_failed(job_id, err)
            return
        if job_id != self._pending_job:
            return
        self._pending_job = None
        self.timeout_timer.stop()
        self._set_state(self.IDLE)
        self._show_error("Processing failed: " + err)

    @Slot(int, object)
    def _job_done(self, job_id, payload):
        if job_id != self._pending_job:
            return
        self._pending_job = None
        self.timeout_timer.stop()
        kind, self._pending_kind = self._pending_kind, ""
        self._set_state(self.IDLE)
        if kind == "evaluate":
            s = payload["summary"]
            QMessageBox.information(self, "Evaluation", "\n".join(f"{k}: {v}" for k, v in s.items())
                                    + f"\n\nReport: {payload['out']}")
        elif kind == "lexicon_add":
            self.statusBar().showMessage(f"Added to {payload['lexicon_path']} "
                                         f"({payload['lexicon_entries']} entries)", 5000)

    # ------------------------------------------------------------ audio files
    def open_audio_files(self):
        if self.state != self.IDLE or not self._models_ready:
            return
        pattern = "Audio (" + " ".join("*" + e for e in AUDIO_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(self, "Open audio file(s) — one Urdu word per file",
                                                str(Path.home()), pattern)
        if not paths:
            return
        profile, fast, std = self._job_args()
        if len(paths) == 1:
            exp = self.expected.text().strip() or expected_word_from_filename(paths[0])
            if exp and script_fraction(exp) < 0.9:
                self._show_error(f"Type the expected word in Urdu script, not in Roman letters (“{exp}”).")
                return
            self._reset_result(keep_audio=False)
            self._set_state(self.PROCESSING)
            self.message.setText(f"Processing {Path(paths[0]).name}…")
            self._pending_kind = "file"
            self._pending_job = self.engine.process_file(paths[0], exp, profile, fast, std)
            self.timeout_timer.start(int(self.cfg.processing_timeout * 1000))
            return
        self.batch = BatchDialog(self, paths, profile, fast, std)
        self._set_state(self.PROCESSING)
        self.batch.finished.connect(self._batch_closed)
        self.batch.show()
        self.batch.start()

    def _batch_closed(self, *_):
        self.batch = None
        self._set_state(self.IDLE)

    def _processing_timeout(self):
        if self._pending_job is None:
            return
        self._pending_job = None
        self.message.setText("Processing took longer than the configured timeout and was abandoned. "
                             "Please record the word again.")
        self._set_state(self.IDLE)

    # ----------------------------------------------------------- results
    def _show_result(self, r: Result):
        self.message.setText(r.message if r.status != "ok" else "")
        self.val["word"].setText(r.recognized_word or "—")
        show_analysed = bool(r.analysed_word and r.analysed_word != r.recognized_word)
        self.val["analysed"].setText(r.analysed_word if show_analysed else "")
        self.key["analysed"].setVisible(show_analysed)
        self.val["analysed"].setVisible(show_analysed)
        self.val["phonetic"].setText(r.phonetic or "—")
        dict_text = ""
        if r.status != "ok" and r.dictionary:
            parts = []
            for d in r.dictionary[:5]:
                gloss = f" ‘{d['gloss'][:40]}’" if d.get("gloss") else ""
                parts.append(f"{d['phonemic']}{gloss}")
            dict_text = "   ·   ".join(parts)
            if len(r.dictionary) > 1:
                dict_text += "\n(ordered by how well each matches the recording)"
        for k, value in (("phonemic", r.phonemic), ("narrow", r.narrow), ("dictionary", dict_text)):
            self.val[k].setText(value or "")
            self.key[k].setVisible(bool(value))
            self.val[k].setVisible(bool(value))
        wc = "not available" if r.word_confidence is None else f"{r.word_confidence_level} ({r.word_confidence:.2f}, uncalibrated)"
        pc = "—" if r.phonetic_confidence is None else (
            f"{r.phonetic_confidence_level} ({r.phonetic_confidence:.0%} of tested contrasts confirmed by the recording)")
        self.val["wconf"].setText(wc)
        self.val["pconf"].setText(pc)
        t = f"{r.processing_ms:.0f} ms"
        if r.timings_ms:
            t += "  (" + ", ".join(f"{k.replace('_ms', '')} {v:.0f}" for k, v in r.timings_ms.items()) + ")"
        self.val["time"].setText(t)
        self.warn.setText("\n".join("• " + w for w in r.warning_texts()))
        self.table.setRowCount(0)
        for s in r.segments:
            if not s["chosen"] and s["source"] == "optional_vowel":
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            dims = "; ".join(f"{d['dimension']} vs {d['alternative'] or '∅'}: {d['status']}"
                             + (f" ({d['margin']:+.1f})" if d.get("margin") is not None else "") for d in s["dims"])
            meas = ", ".join(f"{k}={v}" for k, v in s["measurement"].items() if v is not None)
            tm = f"{s['start']:.2f}–{s['end']:.2f}" if s["start"] is not None else ""
            cells = [s["expected"] or "(spelling)", s["chosen"] or "∅", STATUS_TEXT.get(s["status"], s["status"]),
                     dims, meas + ((" · " + s["note"]) if s["note"] else ""), tm]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 2:
                    it.setForeground(QColor(STATUS_COLOR.get(s["status"], "#000")))
                self.table.setItem(row, c, it)
        if r.free_decode:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem("unconstrained"))
            self.table.setItem(row, 1, QTableWidgetItem(r.free_decode))
            self.table.setItem(row, 2, QTableWidgetItem("diagnostic only — not a transcription"))

    def _reset_result(self, keep_audio: bool):
        self.result = None
        self.saved_id = None
        if not keep_audio:
            self.samples = None
        self.message.setText("")
        for v in self.val.values():
            v.setText("—")
        for k in ("analysed", "phonemic", "narrow", "dictionary"):
            self.key[k].setVisible(False)
            self.val[k].setVisible(False)
        self.warn.setText("")
        self.table.setRowCount(0)
        self.manual.clear()
        self.chk_exclude.blockSignals(True)
        self.chk_exclude.setChecked(False)
        self.chk_exclude.blockSignals(False)

    # -------------------------------------------------------------- actions
    def replay(self):
        if self.samples is None or self.samples.size == 0:
            return
        try:
            self.player.play(self.samples, self.sr)
        except Exception as e:
            self._show_error(f"Playback failed: {e}")

    def record_next(self):
        self.clear_result()
        self.expected.clear()
        self.start_recording()

    def copy_ipa(self):
        if self.result and self.result.phonetic:
            QGuiApplication.clipboard().setText(self.result.phonetic)
            self.statusBar().showMessage("Copied " + self.result.phonetic, 3000)

    def clear_result(self):
        if self.state in (self.RECORDING, self.PROCESSING):
            return
        self.player.stop()
        self._reset_result(keep_audio=False)
        self.meter.set_values([])
        self.lbl_duration.setText("0.0 s")
        self._set_state(self.IDLE)

    def save_result(self):
        if self.samples is None or self.result is None or self.saved_id is not None:
            return
        try:
            self.saved_id = self.store.save(self.samples, self.sr, self.result, speaker=self.speaker.text().strip())
        except Exception as e:
            self._show_error(f"Saving failed: {e}")
            return
        excluded = self.result.status != "ok"
        self.chk_exclude.blockSignals(True)
        self.chk_exclude.setChecked(excluded)
        self.chk_exclude.blockSignals(False)
        self.statusBar().showMessage(f"Saved as record #{self.saved_id}" +
                                     (" (excluded: no reliable transcription)" if excluded else ""), 5000)
        self._set_state(self.state)

    def save_correction(self):
        if self.saved_id is None:
            return
        text = self.manual.text().strip().strip("[]/")
        if not text:
            return
        bad = ipa.validate(text)
        if bad and QMessageBox.question(self, "Unusual symbols",
                                        "These symbols are outside the Urdu inventory used by this tool: "
                                        + " ".join(bad) + "\nSave anyway?") != QMessageBox.Yes:
            return
        ok = self.store.set_manual(self.saved_id, manual_ipa=text, by=self.speaker.text().strip())
        if not ok:
            if QMessageBox.question(self, "Correction exists",
                                    "This record already has a manual correction. Replace it?") != QMessageBox.Yes:
                return
            self.store.set_manual(self.saved_id, manual_ipa=text, overwrite=True)
        self.statusBar().showMessage(f"Correction saved for record #{self.saved_id}", 4000)

    def _exclude_toggled(self, on):
        if self.saved_id is not None:
            self.store.set_excluded(self.saved_id, on)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", str(self.cfg.data_path / "recordings.csv"), "CSV (*.csv)")
        if path:
            n = self.store.export_csv(path, include_excluded=self._ask_include_excluded())
            self.statusBar().showMessage(f"Exported {n} rows to {path}", 5000)

    def export_xlsx(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", str(self.cfg.data_path / "recordings.xlsx"),
                                              "Excel (*.xlsx)")
        if path:
            n = self.store.export_xlsx(path, include_excluded=self._ask_include_excluded())
            self.statusBar().showMessage(f"Exported {n} rows to {path}", 5000)

    def _ask_include_excluded(self) -> bool:
        return QMessageBox.question(self, "Export", "Include records marked as excluded (uncertain results)?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def open_data_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.cfg.data_path)))

    def evaluate(self):
        if self.state != self.IDLE:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Test-set manifest (TSV)", str(self.cfg.data_path), "TSV (*.tsv *.txt)")
        if not path:
            return
        out = str(Path(path).with_name(Path(path).stem + "_report.xlsx"))
        self._set_state(self.PROCESSING)
        self.message.setText("Evaluating the test set…")
        self._pending_kind = "evaluate"
        self._pending_job = self.engine.evaluate(path, out)

    def add_lexicon_entry(self):
        word0 = (self.result.analysed_word if self.result else "") or self.expected.text()
        word, ok = QInputDialog.getText(self, "Add word", "Urdu word:", text=word0)
        if not ok or not word.strip():
            return
        ph, ok = QInputDialog.getText(self, "Add word", f"Phonemic IPA for {word} (no slashes):")
        if not ok or not ph.strip():
            return
        gloss, _ = QInputDialog.getText(self, "Add word", "Gloss (optional):")
        phon = ph.strip().strip("/[]")
        bad = ipa.validate(phon)
        if bad:
            self._show_error("Symbols outside the Urdu inventory: " + " ".join(bad))
            return
        if self.state != self.IDLE:
            self._show_error("Please wait until the current job has finished.")
            return
        self._pending_kind = "lexicon_add"
        self._pending_job = self.engine.add_lexicon(dict(word=word.strip(), phonemic=phon, citation_phonetic="",
                                                         gloss=gloss, source="user", verified="yes", notes=""))

    def _toggle_strict(self, on):
        self.cfg.reconcile.evidence_standard = "strict" if on else "standard"
        self.statusBar().showMessage("Evidence standard: " + self.cfg.reconcile.evidence_standard, 4000)

    def about(self):
        QMessageBox.information(self, "About", (
            f"Urdu Phonetic Transcription Tool {__version__}\n\n"
            "Word recognition and phonetic analysis are separate steps. The phone model scores candidate "
            "pronunciations; contrasts it cannot represent are resolved by acoustic measurement (VOT, closure) "
            "or reported as unverified. Confidence values are not calibrated probabilities.\n\n"
            "See README.md and docs/LIMITATIONS.md before using results as research data."))

    def _show_error(self, text):
        self.message.setText(text)
        self.statusBar().showMessage(text, 8000)

    def closeEvent(self, ev):
        self.poll_timer.stop()
        self.recorder.abort()
        self.player.stop()
        if self.batch is not None:
            self.batch.close()
        self.engine.shutdown()
        super().closeEvent(ev)



class BatchDialog(QDialog):
    """Transcribes several audio files one after another and lists the results."""
    COLS = ["File", "Expected", "Recognised", "IPA phonetic", "IPA phonemic", "Dictionary (not confirmed)",
            "Result", "Phonetic confidence", "Time (ms)"]

    def __init__(self, win: "MainWindow", paths, profile, fast, std):
        super().__init__(win)
        self.win, self.paths, self.args = win, list(paths), (profile, fast, std)
        self.rows: list[dict] = []
        self.idx = 0
        self.current: int | None = None
        self.saved = False
        self.setWindowTitle(f"Transcribing {len(self.paths)} audio files")
        self.resize(1100, 520)
        lay = QVBoxLayout(self)
        self.info = QLabel("")
        lay.addWidget(self.info)
        self.table = QTableWidget(len(self.paths), len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setFont(font(IPA_FONTS, 12))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, p in enumerate(self.paths):
            self.table.setItem(i, 0, QTableWidgetItem(Path(p).name))
            self.table.setItem(i, 6, QTableWidgetItem("waiting"))
        lay.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.btn_save = QPushButton("Save All to Dataset")
        self.btn_csv = QPushButton("Export These Results (CSV)…")
        self.btn_close = QPushButton("Close")
        for b in (self.btn_save, self.btn_csv, self.btn_close):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.btn_save.clicked.connect(self.save_all)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_close.clicked.connect(self.close)
        self._update_buttons()

    def owns(self, job_id) -> bool:
        return job_id is not None and job_id == self.current

    def start(self):
        self._next()

    def _next(self):
        if self.idx >= len(self.paths):
            self.current = None
            ok = sum(1 for r in self.rows if r["result"] is not None and r["result"].status == "ok")
            self.info.setText(f"Finished: {ok} of {len(self.paths)} files transcribed. "
                              "Refused files show the dictionary pronunciation where one exists.")
            self._update_buttons()
            return
        p = self.paths[self.idx]
        self.info.setText(f"Processing {self.idx + 1} of {len(self.paths)}: {Path(p).name}")
        self.table.setItem(self.idx, 6, QTableWidgetItem("processing…"))
        exp = expected_word_from_filename(p)
        self.current = self.win.engine.process_file(p, exp, *self.args)

    def on_result(self, job_id, r: Result, samples, sr):
        i = self.idx
        self.rows.append({"path": self.paths[i], "result": r, "samples": samples, "sr": sr})
        dictionary = " · ".join(d["phonemic"] for d in r.dictionary[:3]) if r.status != "ok" else ""
        status = "transcribed" if r.status == "ok" else (r.message or r.problem)
        conf = "" if r.phonetic_confidence is None else f"{r.phonetic_confidence:.0%}"
        cells = [Path(self.paths[i]).name, r.expected_word, r.recognized_word, r.phonetic or "", r.phonemic or "",
                 dictionary, status, conf, f"{r.processing_ms:.0f}"]
        for c, t in enumerate(cells):
            it = QTableWidgetItem(t)
            if c == 6:
                it.setForeground(QColor("#1b7f3b" if r.status == "ok" else "#b45309"))
            self.table.setItem(i, c, it)
        self.idx += 1
        self._next()

    def on_failed(self, job_id, err):
        i = self.idx
        self.rows.append({"path": self.paths[i], "result": None, "samples": None, "sr": None, "error": err})
        self.table.setItem(i, 6, QTableWidgetItem("could not process: " + err))
        self.idx += 1
        self._next()

    def engine_crashed(self, text):
        self.info.setText(text)

    def _update_buttons(self):
        done = self.current is None and self.idx >= len(self.paths)
        self.btn_save.setEnabled(done and not self.saved and any(r["result"] is not None for r in self.rows))
        self.btn_csv.setEnabled(done)

    def save_all(self):
        n = 0
        speaker = self.win.speaker.text().strip()
        for r in self.rows:
            if r["result"] is not None and r["samples"] is not None:
                self.win.store.save(np.asarray(r["samples"], dtype=np.float32), int(r["sr"]), r["result"],
                                    speaker=speaker, notes="file: " + Path(r["path"]).name)
                n += 1
        self.saved = True
        self._update_buttons()
        self.info.setText(f"Saved {n} results to the dataset (refused ones are marked excluded).")

    def export_csv(self):
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "Export results", str(Path(self.paths[0]).with_name("transcriptions.csv")),
                                              "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "expected_word", "recognized_word", "status", "message", "ipa_phonetic", "ipa_phonemic",
                        "dictionary_not_confirmed", "word_confidence", "phonetic_confidence", "processing_ms", "app_version"])
            for r in self.rows:
                x = r["result"]
                if x is None:
                    w.writerow([Path(r["path"]).name, "", "", "error", r.get("error", "")] + [""] * 6 + [__version__])
                    continue
                w.writerow([Path(r["path"]).name, x.expected_word, x.recognized_word, x.status, x.message, x.phonetic or "",
                            x.phonemic or "", " | ".join(d["phonemic"] for d in x.dictionary), x.word_confidence,
                            x.phonetic_confidence, x.processing_ms, x.app_version])
        self.info.setText(f"Exported {len(self.rows)} rows to {path}")

    def closeEvent(self, ev):
        self.current = None
        self.idx = len(self.paths)
        super().closeEvent(ev)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Urdu single-word speech-to-IPA transcription")
    ap.add_argument("--config", help="path to config.json")
    ap.add_argument("--asr", choices=["auto", "faster-whisper", "whisper", "mms", "manual"],
                    help="override ASR backend")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.asr:
        cfg.asr_backend = a.asr
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Urdu IPA")
    log_path = str(Path.home() / "UrduIPA" / "logs" / "app.log")
    w = MainWindow(cfg, log_path=log_path)
    w.show()
    return app.exec()
