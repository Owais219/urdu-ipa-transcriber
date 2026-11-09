"""Processing engines used by the window.

* SubprocessEngine (the app): all models and numerical work run in a separate process.
  The window's process only records audio and draws. If the engine process dies, the
  window stays open, reports it, and starts a new engine. This also keeps native
  libraries (PyTorch, CTranslate2, OpenBLAS, OpenMP) away from the GUI's threads.
* InProcessEngine (tests, self-test): the same interface around an existing Pipeline,
  running jobs on one worker thread with a large stack.

Messages (tuples) from the engine process:
  ("progress", text) | ("loaded", info) | ("result", job_id, Result, samples|None, sr|None)
  | ("failed", job_id, text) | ("done", job_id, payload)
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import asdict, replace

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

log = logging.getLogger("urdu_ipa.engine")


class EngineBase(QObject):
    progress = Signal(str)
    loaded = Signal(dict)          # {"errors", "models", "lexicon_entries", "load_s", "asr_manual", "lexicon_path"}
    result = Signal(int, object, object, object)   # job id, Result, samples (file jobs) or None, sr or None
    failed = Signal(int, str)
    done = Signal(int, object)     # generic job completion (evaluation, lexicon add)
    crashed = Signal(str)

    def __init__(self):
        super().__init__()
        self._seq = 0

    def next_id(self) -> int:
        self._seq += 1
        return self._seq

    # interface
    def start(self) -> None: ...
    def process(self, samples, sr, expected, profile, skip_asr, evidence_standard) -> int: ...
    def process_file(self, path, expected, profile, skip_asr, evidence_standard) -> int: ...
    def evaluate(self, manifest, out) -> int: ...
    def add_lexicon(self, entry: dict) -> int: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# shared job execution (runs inside the engine process or the worker thread)
# ---------------------------------------------------------------------------

def _with_standard(pipeline, evidence_standard):
    if evidence_standard and evidence_standard != pipeline.cfg.reconcile.evidence_standard:
        pipeline.cfg = replace(pipeline.cfg, reconcile=replace(pipeline.cfg.reconcile,
                                                               evidence_standard=evidence_standard))


def run_job(pipeline, kind, args):
    if kind == "process":
        samples, sr, expected, profile, skip_asr, std = args
        _with_standard(pipeline, std)
        return ("result", pipeline.process(samples, sr, expected, profile, skip_asr), None, None)
    if kind == "file":
        from .audio_io import expected_word_from_filename, load_audio
        path, expected, profile, skip_asr, std = args
        _with_standard(pipeline, std)
        samples, sr = load_audio(path)
        expected = expected or expected_word_from_filename(path)
        return ("result", pipeline.process(samples, sr, expected, profile, skip_asr), samples, sr)
    if kind == "evaluate":
        from .evaluation import run_manifest, write_report
        manifest, out = args
        rep, _ = run_manifest(manifest, pipeline)
        write_report(rep, out)
        return ("done", {"summary": rep["summary"], "out": out})
    if kind == "lexicon_add":
        from .lexicon import Entry
        pipeline.lexicon.add(Entry(**args[0]))
        return ("done", {"lexicon_path": str(pipeline.lexicon.path), "lexicon_entries": len(pipeline.lexicon)})
    raise ValueError(f"unknown job {kind}")


def _loaded_info(pipeline, errors, load_s):
    return {"errors": dict(errors), "models": pipeline.models(), "lexicon_entries": len(pipeline.lexicon),
            "lexicon_problems": list(pipeline.lexicon.problems)[:20], "load_s": round(load_s, 1),
            "asr_manual": pipeline.asr.name == "manual" or "asr" in errors,
            "lexicon_path": str(pipeline.lexicon.path)}


# ---------------------------------------------------------------------------
# in-process engine
# ---------------------------------------------------------------------------

class _Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class _Bridge(QObject):
    msg = Signal(object)


class InProcessEngine(EngineBase):
    def __init__(self, pipeline, autoload: bool = True):
        super().__init__()
        self.pipeline = pipeline
        self.autoload = autoload
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.setStackSize(16 * 1024 * 1024)       # native maths libraries need more than the 512 KB default
        self._bridge = _Bridge()
        self._bridge.msg.connect(self._dispatch)

    def start(self):
        if not self.autoload:
            self.loaded.emit(_loaded_info(self.pipeline, self.pipeline.load_errors, 0.0))
            return

        def work():
            t = time.perf_counter()
            errors = self.pipeline.load(progress=lambda m: self._bridge.msg.emit(("progress", m)))
            self._bridge.msg.emit(("loaded", _loaded_info(self.pipeline, errors, time.perf_counter() - t)))
        self.pool.start(_Task(work))

    def _submit(self, kind, args) -> int:
        jid = self.next_id()

        def work():
            try:
                out = run_job(self.pipeline, kind, args)
                self._bridge.msg.emit((out[0], jid) + tuple(out[1:]))
            except Exception as e:  # noqa: BLE001
                self._bridge.msg.emit(("failed", jid, f"{type(e).__name__}: {e}"))
        self.pool.start(_Task(work))
        return jid

    def _dispatch(self, m):
        _dispatch(self, m)

    def process(self, samples, sr, expected, profile, skip_asr, evidence_standard=None):
        return self._submit("process", (samples, sr, expected, profile, skip_asr, evidence_standard))

    def process_file(self, path, expected, profile, skip_asr, evidence_standard=None):
        return self._submit("file", (path, expected, profile, skip_asr, evidence_standard))

    def evaluate(self, manifest, out):
        return self._submit("evaluate", (manifest, out))

    def add_lexicon(self, entry):
        return self._submit("lexicon_add", (entry,))

    def shutdown(self):
        self.pool.waitForDone(3000)


def _dispatch(engine: EngineBase, m):
    kind = m[0]
    if kind == "progress":
        engine.progress.emit(m[1])
    elif kind == "loaded":
        engine.loaded.emit(m[1])
    elif kind == "result":
        engine.result.emit(m[1], m[2], m[3], m[4])
    elif kind == "failed":
        engine.failed.emit(m[1], m[2])
    elif kind == "done":
        engine.done.emit(m[1], m[2])


# ---------------------------------------------------------------------------
# separate-process engine
# ---------------------------------------------------------------------------

def engine_main(conn, cfg_dict, log_path):
    """Entry point of the engine process."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")      # tiny matrices; avoids parallel solver threads
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import faulthandler
    import sys
    f = None
    if log_path:
        f = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = f
        faulthandler.enable(f)
        logging.basicConfig(stream=f, level=logging.INFO,
                            format="%(asctime)s %(levelname)s engine[%(process)d] %(name)s: %(message)s")
    from .config import Config, _fill
    from .pipeline import Pipeline
    cfg = _fill(Config, cfg_dict)
    log.info("engine starting")
    t = time.perf_counter()
    pipeline = Pipeline(cfg)
    errors = pipeline.load(progress=lambda m: conn.send(("progress", m)))
    conn.send(("loaded", _loaded_info(pipeline, errors, time.perf_counter() - t)))
    log.info("engine ready in %.1f s, errors=%s", time.perf_counter() - t, errors)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg[0] == "stop":
            break
        _, jid, kind, args = msg
        try:
            out = run_job(pipeline, kind, args)
            conn.send((out[0], jid) + tuple(out[1:]))
        except Exception as e:  # noqa: BLE001
            log.error("job %s failed:\n%s", kind, traceback.format_exc())
            conn.send(("failed", jid, f"{type(e).__name__}: {e}"))
    log.info("engine stopped")


class SubprocessEngine(EngineBase):
    MAX_RESTARTS = 3

    def __init__(self, cfg, log_path: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.log_path = log_path
        self.proc = None
        self.conn = None
        self._pending: set[int] = set()
        self._restarts: list[float] = []
        self._stopping = False
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._poll)

    def start(self):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.conn = parent
        self.proc = ctx.Process(target=engine_main, args=(child, asdict(self.cfg), self.log_path),
                                name="UrduIPA-engine", daemon=True)
        self.proc.start()
        child.close()
        self.timer.start()
        log.info("engine process %s started", self.proc.pid)

    def _send(self, kind, args) -> int:
        jid = self.next_id()
        if self.conn is None:
            self.failed.emit(jid, "engine not running")
            return jid
        try:
            self.conn.send(("job", jid, kind, args))
            self._pending.add(jid)
        except (BrokenPipeError, OSError) as e:
            self.failed.emit(jid, f"engine unavailable: {e}")
        return jid

    def process(self, samples, sr, expected, profile, skip_asr, evidence_standard=None):
        import numpy as np
        return self._send("process", (np.asarray(samples, dtype=np.float32), int(sr), expected, profile, skip_asr,
                                      evidence_standard))

    def process_file(self, path, expected, profile, skip_asr, evidence_standard=None):
        return self._send("file", (str(path), expected, profile, skip_asr, evidence_standard))

    def evaluate(self, manifest, out):
        return self._send("evaluate", (manifest, out))

    def add_lexicon(self, entry):
        return self._send("lexicon_add", (entry,))

    def _poll(self):
        try:
            while self.conn is not None and self.conn.poll():
                m = self.conn.recv()
                if m[0] in ("result", "failed", "done"):
                    self._pending.discard(m[1])
                _dispatch(self, m)
        except (EOFError, OSError):
            pass
        if self.proc is not None and not self.proc.is_alive() and not self._stopping:
            self._on_death()

    def _on_death(self):
        code = self.proc.exitcode
        self.timer.stop()
        for jid in sorted(self._pending):
            self.failed.emit(jid, "the processing engine stopped unexpectedly")
        self._pending.clear()
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        self.conn = None
        log.error("engine process exited unexpectedly (exit code %s)", code)
        now = time.monotonic()
        self._restarts = [t for t in self._restarts if now - t < 300] + [now]
        if len(self._restarts) > self.MAX_RESTARTS:
            self.crashed.emit(f"The processing engine stopped {len(self._restarts)} times in 5 minutes "
                              f"(exit code {code}). Please quit and reopen the app, and send ~/UrduIPA/logs/app.log.")
            return
        self.crashed.emit(f"The processing engine stopped unexpectedly (exit code {code}); restarting it…")
        self.start()

    def shutdown(self):
        self._stopping = True
        self.timer.stop()
        try:
            if self.conn is not None:
                self.conn.send(("stop",))
        except Exception:  # noqa: BLE001
            pass
        if self.proc is not None:
            self.proc.join(3)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(2)
