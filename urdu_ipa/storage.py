"""Research dataset storage (SQLite + WAV files) and CSV/Excel export.

Machine output and manual corrections live in separate columns. Machine
writes never touch the manual_* columns; only `set_manual` does.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .pipeline import Result

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    audio_path TEXT,
    sample_rate INTEGER,
    expected_word TEXT,
    recognized_word TEXT,
    analysed_word TEXT,
    status TEXT,
    problem TEXT,
    phonetic_ipa TEXT,
    phonemic_ipa TEXT,
    narrow_ipa TEXT,
    word_confidence REAL,
    word_confidence_level TEXT,
    phonetic_confidence REAL,
    phonetic_confidence_level TEXT,
    candidate TEXT,
    lexicon_verified INTEGER,
    profile TEXT,
    asr_model TEXT,
    acoustic_model TEXT,
    app_version TEXT,
    processing_ms REAL,
    warnings TEXT,
    segments TEXT,
    audio_report TEXT,
    speaker TEXT,
    manual_word TEXT,
    manual_ipa TEXT,
    manual_by TEXT,
    manual_at TEXT,
    excluded INTEGER NOT NULL DEFAULT 0,
    dictionary_ipa TEXT,
    evidence_standard TEXT,
    notes TEXT
);
"""
EXPORT_COLUMNS = ["id", "created_at", "audio_path", "speaker", "profile", "expected_word", "recognized_word",
                  "analysed_word", "status", "problem", "phonetic_ipa", "phonemic_ipa", "narrow_ipa", "dictionary_ipa", "evidence_standard",
                  "word_confidence", "word_confidence_level", "phonetic_confidence", "phonetic_confidence_level",
                  "lexicon_verified", "manual_word", "manual_ipa", "manual_by", "manual_at", "excluded",
                  "asr_model", "acoustic_model", "app_version", "processing_ms", "warnings", "notes"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir).expanduser()
        (self.dir / "audio").mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "recordings.sqlite"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(recordings)")}
            for col in ("dictionary_ipa", "evidence_standard"):     # databases from version 0.1
                if col not in cols:
                    c.execute(f"ALTER TABLE recordings ADD COLUMN {col} TEXT")

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def save(self, samples: np.ndarray, sr: int, result: Result, speaker: str = "", notes: str = "",
             excluded: bool | None = None) -> int:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        wav = self.dir / "audio" / f"{ts}.wav"
        sf.write(wav, np.asarray(samples, dtype=np.float32), sr, subtype="PCM_16")
        if excluded is None:
            excluded = result.status != "ok"
        row = dict(created_at=_now(), audio_path=str(wav.relative_to(self.dir)), sample_rate=sr,
                   expected_word=result.expected_word, recognized_word=result.recognized_word,
                   analysed_word=result.analysed_word, status=result.status, problem=result.problem,
                   phonetic_ipa=result.phonetic, phonemic_ipa=result.phonemic, narrow_ipa=result.narrow,
                   dictionary_ipa=" | ".join(d["phonemic"] for d in result.dictionary) or None,
                   evidence_standard=result.evidence_standard,
                   word_confidence=result.word_confidence, word_confidence_level=result.word_confidence_level,
                   phonetic_confidence=result.phonetic_confidence,
                   phonetic_confidence_level=result.phonetic_confidence_level, candidate=result.candidate,
                   lexicon_verified=None if result.lexicon_verified is None else int(result.lexicon_verified),
                   profile=result.profile, asr_model=result.models.get("asr"),
                   acoustic_model=result.models.get("acoustic"), app_version=result.app_version,
                   processing_ms=result.processing_ms, warnings=json.dumps(result.warnings, ensure_ascii=False),
                   segments=json.dumps(result.segments, ensure_ascii=False),
                   audio_report=json.dumps(result.audio_report), speaker=speaker, excluded=int(excluded),
                   notes=notes)
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self._lock, self._conn() as c:
            cur = c.execute(f"INSERT INTO recordings ({cols}) VALUES ({qs})", list(row.values()))
            return int(cur.lastrowid)

    def set_manual(self, rec_id: int, manual_ipa: str | None = None, manual_word: str | None = None,
                   by: str = "", overwrite: bool = False) -> bool:
        """Store a human correction. Refuses to replace an existing correction unless overwrite=True."""
        with self._lock, self._conn() as c:
            cur = c.execute("SELECT manual_ipa, manual_word FROM recordings WHERE id=?", (rec_id,)).fetchone()
            if cur is None:
                raise KeyError(rec_id)
            if not overwrite and ((manual_ipa and cur["manual_ipa"]) or (manual_word and cur["manual_word"])):
                return False
            sets, vals = [], []
            if manual_ipa is not None:
                sets.append("manual_ipa=?"); vals.append(manual_ipa)
            if manual_word is not None:
                sets.append("manual_word=?"); vals.append(manual_word)
            sets += ["manual_by=?", "manual_at=?"]; vals += [by, _now()]
            c.execute(f"UPDATE recordings SET {', '.join(sets)} WHERE id=?", vals + [rec_id])
            return True

    def set_excluded(self, rec_id: int, excluded: bool) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE recordings SET excluded=? WHERE id=?", (int(excluded), rec_id))

    def rows(self, include_excluded: bool = True, only_ok: bool = False) -> list[dict]:
        q = "SELECT * FROM recordings"
        cond = []
        if not include_excluded:
            cond.append("excluded=0")
        if only_ok:
            cond.append("status='ok'")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id")]

    def export_csv(self, path: str | Path, include_excluded: bool = False, only_ok: bool = False) -> int:
        rows = self.rows(include_excluded, only_ok)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:   # BOM so Excel shows Urdu correctly
            w = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    def export_xlsx(self, path: str | Path, include_excluded: bool = False, only_ok: bool = False) -> int:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        rows = self.rows(include_excluded, only_ok)
        wb = Workbook()
        ws = wb.active
        ws.title = "recordings"
        ws.append(EXPORT_COLUMNS)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for r in rows:
            ws.append([r.get(k) for k in EXPORT_COLUMNS])
        ws2 = wb.create_sheet("segment_evidence")
        ws2.append(["recording_id", "index", "expected", "chosen", "status", "start_s", "end_s", "dimensions",
                    "measurements", "note"])
        for r in rows:
            for s in json.loads(r.get("segments") or "[]"):
                dims = "; ".join(f"{d['dimension']} vs {d['alternative'] or '∅'}: {d['status']}"
                                 + (f" ({d['margin']})" if d.get("margin") is not None else "") for d in s["dims"])
                ws2.append([r["id"], s["index"], s["expected"], s["chosen"], s["status"], s["start"], s["end"],
                            dims, json.dumps(s["measurement"], ensure_ascii=False), s["note"]])
        ws.freeze_panes = "A2"
        wb.save(path)
        return len(rows)
