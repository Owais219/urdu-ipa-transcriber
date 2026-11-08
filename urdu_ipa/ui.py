"""PySide6 user interface.

Responsiveness rules: model loading and processing run on a single-thread
QThreadPool (one job at a time); the recorder is polled by one QTimer that
runs only while recording; every job has an id so a late (timed-out) result
is discarded.
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from . import __version__, ipa
from .config import Config, load_config
from .lexicon import Entry
from .pipeline import Pipeline, Result
from .orthography import script_fraction
from .recorder import Player, Recorder, RecorderError
from .storage import Store

URDU_FONTS = ["Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", "Noto Naskh Arabic", "Segoe UI", "Arial"]
IPA_FONTS = ["Charis SIL", "Doulos SIL", "Gentium Plus", "Noto Sans", "Segoe UI", "DejaVu Sans", "Arial Unicode MS"]
STATUS_TEXT = {"confirmed": "confirmed", "acoustic_variant": "realised differently from dictionary",
               "model_blind": "partly unverifiable by model", "unresolved": "unresolved",
               "not_realised": "not realised"}
STATUS_COLOR = {"confirmed": "#1b7f3b", "acoustic_variant": "#1d4ed8", "model_blind": "#8a6d00",
                "unresolved": "#b45309", "not_realised": "#6b7280"}


def font(families, size, bold=False):
    f = QFont()
    f.setFamilies(families)
    f.setPointSize(size)
    f.setBold(bold)
    return f


class _Signals(QObject):
    started = Signal(int)
    done = Signal(int, object)
    failed = Signal(int, str)
    progress = Signal(str)


class Job(QRunnable):
    def __init__(self, job_id, fn, *args):
        super().__init__()
        self.job_id, self.fn, self.args = job_id, fn, args
        self.signals = _Signals()

    def run(self):
        self.signals.started.emit(self.job_id)
        try:
            out = self.fn(*self.args)
            self.signals.done.emit(self.job_id, out)
        except Exception as e:
            self.signals.failed.emit(self.job_id, f"{type(e).__name__}: {e}")


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
                 store: Store | None = None, player: Player | None = None, autoload: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pipeline = pipeline or Pipeline(cfg)
        self.recorder = recorder or Recorder(cfg.recording)
        self.store = store or Store(cfg.data_path)
        self.player = player or Player()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.samples: np.ndarray | None = None
        self.sr = 0
        self.result: Result | None = None
        self.saved_id: int | None = None
        self._job_seq = 0
        self._pending_job: int | None = None
        self._models_ready = False
        self.state = self.IDLE

        self.setWindowTitle("Urdu Phonetic Transcription Tool")
        self.resize(880, 900)
        self._build()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(50)
        self.poll_timer.timeout.connect(self._poll)
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self._processing_timeout)
        if autoload:
            self._load_models()
        else:
            self._models_ready = True
        self._set_state(self.IDLE)

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(20, 16, 20, 12)
        title = QLabel("Urdu Phonetic Transcription Tool")
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
        for key, p in self.pipeline.profiles.items():
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
        row.addWidget(self.btn_record, 3)
        row.addWidget(self.btn_stop, 1)
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
        self.btn_copy.setEnabled(bool(self.result and self.result.phonetic) and not rec)
        self.btn_clear.setEnabled(not rec and not busy and (self.result is not None or have))
        self.btn_save.setEnabled(have and self.result is not None and self.saved_id is None and not busy and not rec)
        self.btn_correct.setEnabled(self.saved_id is not None and not rec)
        self.chk_exclude.setEnabled(self.saved_id is not None and not rec)
        self.expected.setEnabled(not rec and not busy)
        self.profile.setEnabled(not rec and not busy)
        self.lbl_state.setText({self.IDLE: "Ready" if ready else
                                "Loading speech models… recording will be available when they are ready. "
                                "The first launch downloads the Whisper model (1.6 GB or more); "
                                "download progress is shown in the Terminal window.",
                                self.RECORDING: "Recording… speak one Urdu word",
                                self.PROCESSING: "Processing…"}.get(st, st))

    def _load_models(self):
        self.statusBar().showMessage("Loading speech models (first run may download them)…")
        self._t_load = time.monotonic()
        self._job_seq += 1
        job = Job(self._job_seq, None)
        job.fn = lambda: self.pipeline.load(progress=job.signals.progress.emit)
        job.signals.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        job.signals.done.connect(self._models_loaded)
        job.signals.failed.connect(self._models_failed)
        self._load_job = job
        self.pool.start(job)

    @Slot(int, str)
    def _models_failed(self, _id, err):
        self._models_ready = False
        self.statusBar().showMessage("Model loading failed: " + err)
        self._set_state(self.IDLE)

    @Slot(int, object)
    def _models_loaded(self, _id, errors):
        m = self.pipeline.models()
        self._models_ready = "acoustic" not in errors
        took = time.monotonic() - getattr(self, "_t_load", time.monotonic())
        if errors:
            hint = (" — the speech recogniser is unavailable; type the expected word in Urdu script before each recording"
                    if "asr" in errors and self._models_ready else "")
            self.statusBar().showMessage("Model problems: " + "; ".join(f"{k}: {v}" for k, v in errors.items()) + hint)
        else:
            self.statusBar().showMessage(f"Ready · ASR: {m['asr']} · acoustic: {m['acoustic']} · "
                                         f"lexicon entries: {len(self.pipeline.lexicon)} · models loaded in {took:.0f} s")
        if self.state == self.IDLE:
            self._set_state(self.IDLE)

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
        if not exp and (self.pipeline.asr.name == "manual" or "asr" in self.pipeline.load_errors):
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

    def _process(self):
        self._set_state(self.PROCESSING)
        self.message.setText("Processing…")
        self._job_seq += 1
        self._pending_job = self._job_seq
        job = Job(self._job_seq, self.pipeline.process, self.samples, self.sr, self.expected.text(),
                  self.profile.currentData() or "standard", self.chk_fast.isChecked())
        job.signals.done.connect(self._processed)
        job.signals.failed.connect(self._failed)
        # the timeout counts from when processing actually starts, not while it waits in the queue
        job.signals.started.connect(lambda jid: jid == self._pending_job and
                                    self.timeout_timer.start(int(self.cfg.processing_timeout * 1000)))
        self.pool.start(job)

    @Slot(int, object)
    def _processed(self, job_id, result: Result):
        if job_id != self._pending_job:
            return                         # stale (timed out) result
        self._pending_job = None
        self.timeout_timer.stop()
        self.result = result
        self.saved_id = None
        self._show_result(result)
        self._set_state(self.IDLE)

    @Slot(int, str)
    def _failed(self, job_id, err):
        if job_id != self._pending_job:
            return
        self._pending_job = None
        self.timeout_timer.stop()
        self._set_state(self.IDLE)
        self._show_error("Processing failed: " + err)

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
        for k, value in (("phonemic", r.phonemic), ("narrow", r.narrow)):
            self.val[k].setText(value or "")
            self.key[k].setVisible(bool(value))
            self.val[k].setVisible(bool(value))
        wc = "not available" if r.word_confidence is None else f"{r.word_confidence_level} ({r.word_confidence:.2f}, uncalibrated)"
        pc = "—" if r.phonetic_confidence is None else (
            f"{r.phonetic_confidence_level} ({r.phonetic_confidence:.0%} of tested contrasts supported)")
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
        for k in ("analysed", "phonemic", "narrow"):
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
        from .evaluation import run_manifest, write_report

        def work():
            rep, _ = run_manifest(path, self.pipeline)
            write_report(rep, out)
            return rep["summary"]

        self._set_state(self.PROCESSING)
        self._job_seq += 1
        job = Job(self._job_seq, work)
        job.signals.done.connect(lambda _i, s: (self._set_state(self.IDLE), QMessageBox.information(
            self, "Evaluation", "\n".join(f"{k}: {v}" for k, v in s.items()) + f"\n\nReport: {out}")))
        job.signals.failed.connect(lambda _i, e: (self._set_state(self.IDLE), self._show_error("Evaluation failed: " + e)))
        self.pool.start(job)

    def add_lexicon_entry(self):
        word0 = (self.result.analysed_word if self.result else "") or self.expected.text()
        word, ok = QInputDialog.getText(self, "Add word", "Urdu word:", text=word0)
        if not ok or not word.strip():
            return
        ph, ok = QInputDialog.getText(self, "Add word", f"Phonemic IPA for {word} (no slashes):")
        if not ok or not ph.strip():
            return
        gloss, _ = QInputDialog.getText(self, "Add word", "Gloss (optional):")
        try:
            self.pipeline.lexicon.add(Entry(word.strip(), ph.strip().strip("/[]"), "", gloss, "user", "yes", ""))
        except ValueError as e:
            self._show_error(str(e))
            return
        self.statusBar().showMessage(f"Added {word} /{ph.strip()}/ to {self.pipeline.lexicon.path}", 5000)

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
        self.pool.waitForDone(2000)
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
    w = MainWindow(cfg)
    w.show()
    return app.exec()
