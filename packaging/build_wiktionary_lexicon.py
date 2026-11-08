"""Build data/lexicon_wiktionary.tsv from the kaikki.org Wiktionary extract of Urdu.

    python packaging/build_wiktionary_lexicon.py kaikki.org-dictionary-Urdu.jsonl data/lexicon_wiktionary.tsv

Source: English Wiktionary (CC BY-SA 4.0) via kaikki.org / Wiktextract (Ylonen 2022).
Only the phonemic "Standard Urdu" transcriptions (/.../) are used; colloquial phonetic
variants are skipped. Conventions are converted to the project standard
(docs/TRANSCRIPTION_STANDARD.md):
  * ɾ -> r ;  ɦ -> h (phonemic /h/) ;  w -> ʋ ;  g -> ɡ
  * optional material in parentheses is dropped:  (ʔ)iːd̪ -> iːd̪ ;  t͡ʃɑ(ː) -> t͡ʃɑ
  * geminates written across a syllable boundary become a length mark:
      bət̪.t͡ʃɑ -> bət͡ʃːɑ ;  mʊ.ɦəb.bət̪ -> mʊhəbːət̪ ;  ət̪.t͡ʃʰɑː -> ət͡ʃːʰɑː
  * a word ending in ہ after a consonant ends in short ɑ (not ɑː)
Entries that still contain symbols outside the inventory are left out and counted.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from urdu_ipa import ipa                       # noqa: E402
from urdu_ipa.orthography import ends_with_final_heh, normalize, script_fraction   # noqa: E402

SYMBOL_MAP = {"ɾ": "r", "ɦ": "h", "w": "ʋ", "g": "ɡ", "ᵊ": "", "̚": "", "̈": "", "̆": "",
              "ˈ": "", "ˌ": "", "‿": "", "-": "", "͜": "", "̤": "ʱ"}
AFFRICATE_GEMINATION = {("t̪", "t͡ʃ"), ("t̪", "t͡ʃʰ"), ("d̪", "d͡ʒ"), ("d̪", "d͡ʒʱ"), ("t", "t͡ʃ"), ("d", "d͡ʒ")}


def convert(raw: str, word: str) -> str | None:
    s = unicodedata.normalize("NFC", raw.strip())
    if not (s.startswith("/") and s.endswith("/")):
        return None
    s = s.strip("/")
    s = re.sub(r"\([^)]*\)", "", s)                    # optional segments
    s = "".join(SYMBOL_MAP.get(ch, ch) for ch in unicodedata.normalize("NFD", s))
    s = unicodedata.normalize("NFC", s)
    if " " in s:                                         # multi-word phrases are out of scope
        return None
    segs = ipa.tokenize(s)                               # drops '.' syllable marks
    out: list[str] = []
    for seg in segs:
        if out:
            prev = out[-1]
            base = ipa.ungeminate(seg)
            plain = base.rstrip("ʰʱ")
            if (prev == plain or (prev, base) in AFFRICATE_GEMINATION) and ipa.is_consonant(base) and not ipa.is_geminate(prev):
                out[-1] = ipa.geminate(base)
                continue
        out.append(seg)
    if ends_with_final_heh(word) and out and out[-1] == "ɑː":
        out[-1] = "ɑ"
    res = ipa.join(out)
    return res if res and not ipa.validate(res) else None


def main(src: str, dst: str) -> None:
    stats = collections.Counter()
    rows: dict[tuple[str, str], list[str]] = {}
    for line in open(src, encoding="utf-8"):
        e = json.loads(line)
        word = normalize(e.get("word", ""))
        if not word or " " in word or script_fraction(word) < 1.0:
            stats["skipped_not_single_urdu_word"] += 1
            continue
        for snd in e.get("sounds", []):
            if not snd.get("ipa"):
                continue
            tags = set(snd.get("tags") or [])
            if tags & {"colloquial", "dialectal", "obsolete", "archaic", "letter", "phoneme"}:
                stats["skipped_tag"] += 1
                continue
            conv = convert(snd["ipa"], word)
            if conv is None:
                stats["skipped_unconvertible"] += 1
                continue
            key = (word, conv)
            if key not in rows:
                gloss = ""
                for sense in e.get("senses", []):
                    if sense.get("glosses"):
                        gloss = sense["glosses"][0][:80]
                        break
                rows[key] = [word, conv, "", gloss.replace("\t", " "), "wiktionary", "no",
                             "converted from " + snd["ipa"]]
                stats["kept"] += 1
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write("# Urdu pronunciations from English Wiktionary (CC BY-SA 4.0), via kaikki.org/Wiktextract.\n"
                "# Converted to this project's IPA conventions by packaging/build_wiktionary_lexicon.py.\n"
                "# Not individually verified. Hand-checked entries in lexicon.tsv take priority.\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["word", "phonemic", "citation_phonetic", "gloss", "source", "verified", "notes"])
        for k in sorted(rows):
            w.writerow(rows[k])
    words = len({k[0] for k in rows})
    print(dict(stats), "rows", len(rows), "distinct words", words)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
