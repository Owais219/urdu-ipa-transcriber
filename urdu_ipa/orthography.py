"""Urdu orthography: normalisation and orthography-constrained pronunciation lattices.

Urdu script normally omits short vowels, and و / ی / ہ are ambiguous between
consonant and vowel values. A spelling therefore does NOT determine a single
pronunciation. This module turns a spelling into a *lattice* of slots, each
with the set of values the spelling allows. The acoustic module then chooses
among them from the audio. Nothing here is a final transcription.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import ipa

# --- normalisation ---------------------------------------------------------
_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh -> Farsi/Urdu yeh
    "ى": "ی",  # alef maksura -> yeh
    "ك": "ک",  # Arabic kaf -> keheh
    "ه": "ہ",  # Arabic heh -> heh goal
    "ە": "ہ",
    "ة": "ہ",  # teh marbuta -> heh goal (word-final -a)
    "ۀ": "ۂ",
    "ـ": "",        # tatweel
    "‌": "",        # ZWNJ (kept out of lookup keys)
    "‍": "",
    "‏": "", "‎": "",
}
HARAKAT = {"َ": "zabar", "ِ": "zer", "ُ": "pesh", "ْ": "jazm",
           "ّ": "tashdid", "ٰ": "khari_zabar", "ً": "tanwin_fath",
           "ٔ": "hamza_above", "ٖ": "subscript_alef"}
_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def normalize(text: str, keep_diacritics: bool = True) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    # presentation forms -> base letters
    text = "".join(unicodedata.normalize("NFKC", ch) if "ﭐ" <= ch <= "﻿" else ch for ch in text)
    text = "".join(_CHAR_MAP.get(ch, ch) for ch in text)
    if not keep_diacritics:
        text = "".join(ch for ch in text if ch not in HARAKAT)
    text = re.sub(r"[\s ]+", " ", text)
    text = re.sub(r"[۔،؟!?.,:;\"'«»()\[\]{}]", "", text)
    return text.strip()


def lookup_key(text: str) -> str:
    """Key used for lexicon look-up: no diacritics, no spaces."""
    return normalize(text, keep_diacritics=False).replace(" ", "")


def script_fraction(text: str) -> float:
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _ARABIC_LETTER.match(ch)) / len(letters)


def words(text: str) -> list[str]:
    return [w for w in normalize(text).split(" ") if w]


# --- lattice ---------------------------------------------------------------

@dataclass
class Slot:
    options: list[str]                 # "" means "nothing here"
    canonical: int | None = 0          # index into options, None = spelling leaves it open
    kind: str = "lexical"              # lexical | orthographic | optional_vowel
    source: str = ""                   # letter(s) or lexical phoneme it came from
    dimensions: dict[str, list[str]] = field(default_factory=dict)  # option -> contrast it tests

    def canonical_value(self) -> str | None:
        return None if self.canonical is None else self.options[self.canonical]


FINAL_H = "final_h"   # slot kind: optional word-final breathy release written with ہ

CONS = {
    "ب": "b", "پ": "p", "ت": "t̪", "ٹ": "ʈ", "ث": "s", "ج": "d͡ʒ", "چ": "t͡ʃ", "ح": "h",
    "خ": "x", "د": "d̪", "ڈ": "ɖ", "ذ": "z", "ر": "r", "ڑ": "ɽ", "ز": "z", "ژ": "ʒ",
    "س": "s", "ش": "ʃ", "ص": "s", "ض": "z", "ط": "t̪", "ظ": "z", "غ": "ɣ", "ف": "f",
    "ق": "q", "ک": "k", "گ": "ɡ", "ل": "l", "م": "m", "ن": "n",
}
ASPIRATED = {"p": "pʰ", "b": "bʱ", "t̪": "t̪ʰ", "d̪": "d̪ʱ", "ʈ": "ʈʰ", "ɖ": "ɖʱ", "k": "kʰ",
             "ɡ": "ɡʱ", "t͡ʃ": "t͡ʃʰ", "d͡ʒ": "d͡ʒʱ", "ɽ": "ɽʱ"}
DO_CHASHMI = "ھ"
SHORT_V = {"zabar": "ə", "zer": "ɪ", "pesh": "ʊ"}
OPT_SHORT = ["", "ə", "ɪ", "ʊ"]


def _is_letter(ch):
    return ch not in HARAKAT and ch != " "


def lattice_from_spelling(word: str) -> list[Slot]:
    """Build an orthography-constrained lattice for an Urdu spelling."""
    w = normalize(word)
    w = w.replace(" ", "")
    chars = list(w)
    n = len(chars)
    slots: list[Slot] = []
    i = 0

    def marks_after(j):
        m = []
        k = j + 1
        while k < n and chars[k] in HARAKAT:
            m.append(HARAKAT[chars[k]])
            k += 1
        return m, k

    def last_vowel_slot():
        for s in reversed(slots):
            if s.kind in ("orthographic", "optional_vowel") and any(ipa.is_vowel(o) for o in s.options if o):
                return s
        return None

    prev_was_cons = False
    while i < n:
        ch = chars[i]
        marks, nxt = marks_after(i)
        nextch = chars[nxt] if nxt < n else ""
        at_start = not slots
        at_end = nxt >= n

        if ch in CONS:
            val = CONS[ch]
            if nextch == DO_CHASHMI:
                val = ASPIRATED.get(val, val)
                m2, nxt = marks_after(nxt)
                marks += m2
                nextch = chars[nxt] if nxt < n else ""
                at_end = nxt >= n
            if "tashdid" in marks:
                val = ipa.geminate(val)
            cons_idx = len(slots)
            slots.append(Slot([val], 0, "orthographic", ch))
            # vowel after this consonant
            vmark = next((SHORT_V[m] for m in marks if m in SHORT_V), None)
            if vmark:
                slots.append(Slot([vmark], 0, "orthographic", "haraka"))
            elif "khari_zabar" in marks:
                slots.append(Slot(["ɑː"], 0, "orthographic", "khari zabar"))
            elif "jazm" in marks:
                pass
            elif not at_end and nextch and nextch not in "اآوی ےہعء" and nextch != "ں":
                slots.append(Slot(list(OPT_SHORT), None, "optional_vowel", "unwritten"))
            elif not at_end and nextch in "ہع" and nxt + 1 < n:
                slots.append(Slot(list(OPT_SHORT), None, "optional_vowel", "unwritten"))
            # ن after a vowel and before a consonant often only nasalises the vowel (آنکھ, چاند)
            if ch == "ن" and cons_idx >= 1 and nextch in CONS:
                prev = slots[cons_idx - 1]
                if prev.kind == "orthographic" and all(ipa.is_vowel(o) for o in prev.options if o):
                    slots[cons_idx] = Slot(["n", ""], None, "orthographic", ch)
                    prev.options = prev.options + [ipa.nasalize(o) for o in prev.options
                                                   if o and ipa.nasalize(o) not in prev.options]
                    if prev.canonical is not None:
                        prev.canonical = None
            prev_was_cons = True
            i = nxt
            continue

        if ch == "آ":
            slots.append(Slot(["ɑː"], 0, "orthographic", ch))
        elif ch == "ا":
            if at_start:
                if nextch == "و":
                    slots.append(Slot(["oː", "ɔː", "uː"], None, "orthographic", "او"))
                    nxt += 1
                elif nextch in "یے":
                    slots.append(Slot(["iː", "eː", "ɛː"], None, "orthographic", "ای"))
                    nxt += 1
                else:
                    v = next((SHORT_V[m] for m in marks if m in SHORT_V), None)
                    slots.append(Slot([v] if v else ["ə", "ɪ", "ʊ"], 0 if v else None, "orthographic", ch))
            else:
                slots.append(Slot(["ɑː"], 0, "orthographic", ch))
        elif ch == "و":
            if at_start or (nextch in "اآ" and not at_end):
                slots.append(Slot(["ʋ"], 0, "orthographic", ch))
                if at_start:
                    v = next((SHORT_V[m] for m in marks if m in SHORT_V), None)
                    slots.append(Slot([v] if v else list(OPT_SHORT), 0 if v else None, "optional_vowel", "unwritten"))
            elif "pesh" in marks:
                slots.append(Slot(["uː"], 0, "orthographic", ch))
            else:
                slots.append(Slot(["uː", "oː", "ɔː", "ʊ", "ʋ"], None, "orthographic", ch))
        elif ch == "ی":
            if at_start:
                slots.append(Slot(["j"], 0, "orthographic", ch))
                slots.append(Slot(list(OPT_SHORT), None, "optional_vowel", "unwritten"))
            elif at_end:
                slots.append(Slot(["iː"], 0, "orthographic", ch))
            elif nextch in "اآو":
                slots.append(Slot(["j", "iː"], None, "orthographic", ch))
            else:
                slots.append(Slot(["iː", "eː", "ɛː", "j"], None, "orthographic", ch))
        elif ch == "ے":
            slots.append(Slot(["eː", "ɛː"], None, "orthographic", ch))
        elif ch in "ہۂ":
            if at_end and prev_was_cons:
                # word-final ہ after a consonant: short [ɑ] (contrasting with ا = ɑː); whether a final
                # breathy [h] is audible is decided from the audio (FINAL_H slot)
                slots.append(Slot(["ɑ", "ə", "eː"], None, "orthographic", "final ہ"))
                slots.append(final_h_slot())
            elif at_end:
                slots.append(Slot(["", "h"], None, "orthographic", "final ہ"))
            else:
                slots.append(Slot(["h"], 0, "orthographic", ch))
                if not at_end and nextch not in "اآوی ے":
                    slots.append(Slot(list(OPT_SHORT), None, "optional_vowel", "unwritten"))
        elif ch == "ں":
            target = last_vowel_slot()
            if target is not None:
                target.options = [ipa.nasalize(o) if ipa.is_vowel(o) else o for o in target.options]
            else:
                slots.append(Slot(["n"], 0, "orthographic", ch))
        elif ch == "ع":
            slots.append(Slot(["", "ʔ"], None, "orthographic", ch))
            if not at_end and nextch not in "اآوی ے":
                slots.append(Slot(["ə", "ɪ", "ʊ"], None, "optional_vowel", "unwritten"))
        elif ch in "ءئؤ":
            slots.append(Slot(["", "ʔ", "j"], None, "orthographic", ch))
            if ch == "ؤ":
                slots.append(Slot(["uː", "oː"], None, "orthographic", ch))
        elif ch == DO_CHASHMI:
            # ھ not after an aspirable consonant (e.g. after م ن ل): murmured sonorant / h
            slots.append(Slot(["h"], 0, "orthographic", ch))
        # any other character is ignored (digits, punctuation already removed)
        prev_was_cons = False
        i = nxt
    # drop slots whose only option is empty
    return [s for s in slots if any(o != "" for o in s.options) or len(s.options) > 1]


def final_h_slot() -> Slot:
    return Slot(["", "h"], None, FINAL_H, "final ہ", {"": "final_aspiration", "h": "final_aspiration"})


def ends_with_final_heh(word: str) -> bool:
    """True for a word ending in ہ preceded by a consonant letter (جرگہ, بچّہ), not شاہ or راہ."""
    w = normalize(word, keep_diacritics=False).replace(" ", "")
    return len(w) >= 2 and w[-1] in "ہۂ" and (w[-2] in CONS or w[-2] == DO_CHASHMI)


def is_fully_determined(slots: list[Slot]) -> bool:
    return all(s.canonical is not None for s in slots)
