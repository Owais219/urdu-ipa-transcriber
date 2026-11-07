"""Pronunciation lexicon (TSV) with homograph support.

Columns: word, phonemic, citation_phonetic, gloss, source, verified, notes

* `phonemic` is written without slashes, e.g.  pʰəl
* `citation_phonetic` (optional) is the expected broad phonetic citation form
  when it differs from the phonemic form (e.g. ʈʰəɳɖɑː for /ʈʰənɖɑː/).
* `verified` is "yes" only after a phonologist has checked the entry.
  Seed entries shipped with the tool are marked "no".
* Several rows may share a spelling (homographs such as پل /pəl/ 'moment'
  and /pʊl/ 'bridge'); all are passed to the acoustic scorer.
"""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

from . import ipa
from .orthography import lookup_key, normalize

FIELDS = ["word", "phonemic", "citation_phonetic", "gloss", "source", "verified", "notes"]


@dataclass
class Entry:
    word: str
    phonemic: str
    citation_phonetic: str = ""
    gloss: str = ""
    source: str = ""
    verified: str = "no"
    notes: str = ""

    @property
    def expected_phonetic(self) -> str:
        return self.citation_phonetic or self.phonemic

    @property
    def is_verified(self) -> bool:
        return self.verified.strip().lower() in ("yes", "y", "true", "1")


class Lexicon:
    """`path` is the writable (user) lexicon. `base` is an optional read-only lexicon shipped
    with the application; for any spelling present in the user lexicon, the user's rows
    replace the built-in rows."""

    def __init__(self, path: str | Path, base: str | Path | None = None):
        self.path = Path(path)
        self.base = Path(base) if base else None
        self._lock = threading.Lock()
        self._by_key: dict[str, list[Entry]] = {}
        self.problems: list[str] = []
        self.load()

    def load(self) -> None:
        self._by_key.clear()
        self.problems.clear()
        if self.base and self.base.exists() and self.base.resolve() != self.path.resolve():
            self._by_key.update(self._read(self.base, "built-in"))
        if self.path.exists():
            self._by_key.update(self._read(self.path, self.path.name))

    def _read(self, path: Path, label: str) -> dict[str, list[Entry]]:
        out: dict[str, list[Entry]] = {}
        with path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.reader(f, delimiter="\t") if r and not r[0].startswith("#")]
        if rows and rows[0][0].strip().lower() == "word":
            rows = rows[1:]
        for lineno, row in enumerate(rows, start=2):
            row = (row + [""] * len(FIELDS))[: len(FIELDS)]
            e = Entry(*[c.strip() for c in row])
            if not e.word or not e.phonemic:
                continue
            bad = ipa.validate(e.phonemic) + (ipa.validate(e.citation_phonetic) if e.citation_phonetic else [])
            if bad:
                self.problems.append(f"{label} line {lineno}: {e.word}: symbols outside the Urdu inventory: {' '.join(bad)}")
            out.setdefault(lookup_key(e.word), []).append(e)
        return out

    def lookup(self, word: str) -> list[Entry]:
        return list(self._by_key.get(lookup_key(word), []))

    def __contains__(self, word: str) -> bool:
        return lookup_key(word) in self._by_key

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    def entries(self) -> list[Entry]:
        return [e for v in self._by_key.values() for e in v]

    def add(self, entry: Entry) -> None:
        """Append an entry (never rewrites existing rows)."""
        bad = ipa.validate(entry.phonemic)
        if bad:
            raise ValueError(f"Symbols outside the Urdu inventory: {' '.join(bad)}")
        entry.word = normalize(entry.word)
        with self._lock:
            new_file = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t", lineterminator="\n")
                if new_file:
                    w.writerow(FIELDS)
                w.writerow([asdict(entry)[k] for k in FIELDS])
            key = lookup_key(entry.word)
            mine = [e for e in self._read(self.path, self.path.name).get(key, [])]
            self._by_key[key] = mine
