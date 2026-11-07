"""IPA inventory, tokenisation and feature description for Urdu.

Conventions used throughout the project (documented in docs/TRANSCRIPTION_STANDARD.md):

* Dental stops carry the dental diacritic: t̪ t̪ʰ d̪ d̪ʱ.
* Voiceless aspiration is written with ʰ, voiced (breathy) aspiration with ʱ.
* Affricates are written with a tie bar: t͡ʃ d͡ʒ.
* Long vowels and geminate consonants use the length mark ː.
  For an aspirated geminate the order is  base + ː + ʰ/ʱ  (e.g. t͡ʃːʰ).
* Nasalised vowels use the combining tilde: ɑ̃ː.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

LENGTH = "ː"
ASP = "ʰ"
BREATHY_ASP = "ʱ"
NASAL = "̃"  # combining tilde
TIE = "͡"
IGNORED = {"ˈ", "ˌ", ".", " ", "‿", "|", "‖", "/", "[", "]"}
_POST_MODIFIERS = {"ʰ", "ʱ", "ː", "ˑ", "ʲ", "ʷ", "ˀ", "ⁿ", "ˡ"}


def tokenize(ipa: str) -> list[str]:
    """Split an IPA string into segments.

    A segment is a base symbol plus any combining diacritics, tie-bar joined
    second symbol, and post-modifiers (ʰ ʱ ː ...). Stress, syllable marks,
    brackets and slashes are discarded. Modifiers are put in canonical order
    (combining marks, then ː, then ʰ/ʱ) so that t͡ʃʰː and t͡ʃːʰ compare equal.
    """
    text = unicodedata.normalize("NFD", ipa or "")
    segs: list[list[str]] = []
    join_next = False
    for ch in text:
        if ch in IGNORED:
            join_next = False
            continue
        cat = unicodedata.category(ch)
        if ch == TIE or ch == "͜":
            if segs:
                segs[-1].append(TIE)
                join_next = True
            continue
        if cat == "Mn" or ch in _POST_MODIFIERS:
            if segs:
                segs[-1].append(ch)
            continue
        if join_next and segs:
            segs[-1].append(ch)
            join_next = False
            continue
        segs.append([ch])
    return [_canonical(s) for s in segs]


def _canonical(parts: list[str]) -> str:
    base = []
    combining = []
    length = []
    asp = []
    other = []
    seen_base = False
    i = 0
    while i < len(parts):
        ch = parts[i]
        if not seen_base or ch == TIE:
            base.append(ch)
            seen_base = True
            if ch == TIE and i + 1 < len(parts):
                base.append(parts[i + 1])
                i += 1
        elif unicodedata.category(ch) == "Mn":
            combining.append(ch)
        elif ch in ("ː", "ˑ"):
            length.append(ch)
        elif ch in ("ʰ", "ʱ"):
            asp.append(ch)
        else:
            other.append(ch)
        i += 1
    seg = "".join(base) + "".join(sorted(set(combining), key=combining.index))
    seg = unicodedata.normalize("NFC", seg + "".join(other) + "".join(length[:1]) + "".join(asp[:1]))
    return seg


def join(segments: list[str]) -> str:
    return unicodedata.normalize("NFC", "".join(segments))


# ---------------------------------------------------------------------------
# Feature description of the Urdu segment inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Seg:
    symbol: str
    kind: str                 # "C" or "V"
    place: str = ""
    manner: str = ""
    voiced: bool = False
    aspirated: bool = False
    nasal_vowel: bool = False
    long: bool = False
    note: str = ""


def _c(sym, place, manner, voiced, asp=False, note=""):
    return Seg(sym, "C", place, manner, voiced, asp, note=note)


def _v(sym, height_backness, long, nasal=False):
    return Seg(sym, "V", place=height_backness, long=long, nasal_vowel=nasal)


CONSONANTS: dict[str, Seg] = {s.symbol: s for s in [
    _c("p", "bilabial", "stop", False), _c("pʰ", "bilabial", "stop", False, True),
    _c("b", "bilabial", "stop", True), _c("bʱ", "bilabial", "stop", True, True),
    _c("m", "bilabial", "nasal", True),
    _c("t̪", "dental", "stop", False), _c("t̪ʰ", "dental", "stop", False, True),
    _c("d̪", "dental", "stop", True), _c("d̪ʱ", "dental", "stop", True, True),
    _c("n", "alveolar", "nasal", True), _c("s", "alveolar", "fricative", False),
    _c("z", "alveolar", "fricative", True), _c("r", "alveolar", "rhotic", True),
    _c("l", "alveolar", "lateral", True),
    _c("ʈ", "retroflex", "stop", False), _c("ʈʰ", "retroflex", "stop", False, True),
    _c("ɖ", "retroflex", "stop", True), _c("ɖʱ", "retroflex", "stop", True, True),
    _c("ɳ", "retroflex", "nasal", True, note="mostly before retroflex stops"),
    _c("ɽ", "retroflex", "flap", True), _c("ɽʱ", "retroflex", "flap", True, True),
    _c("t͡ʃ", "postalveolar", "affricate", False), _c("t͡ʃʰ", "postalveolar", "affricate", False, True),
    _c("d͡ʒ", "postalveolar", "affricate", True), _c("d͡ʒʱ", "postalveolar", "affricate", True, True),
    _c("ʃ", "postalveolar", "fricative", False), _c("ʒ", "postalveolar", "fricative", True),
    _c("j", "palatal", "approximant", True),
    _c("k", "velar", "stop", False), _c("kʰ", "velar", "stop", False, True),
    _c("ɡ", "velar", "stop", True), _c("ɡʱ", "velar", "stop", True, True),
    _c("ŋ", "velar", "nasal", True, note="mostly before velar stops"),
    _c("x", "velar", "fricative", False), _c("ɣ", "velar", "fricative", True),
    _c("q", "uvular", "stop", False),
    _c("f", "labiodental", "fricative", False), _c("ʋ", "labiodental", "approximant", True),
    _c("h", "glottal", "fricative", False, note="often realised [ɦ]"),
    _c("ɦ", "glottal", "fricative", True),
    _c("ʔ", "glottal", "stop", False, note="marginal; hiatus / careful speech"),
    _c("ħ", "pharyngeal", "fricative", False, note="not used in Standard Pakistani Urdu profile"),
    _c("ʕ", "pharyngeal", "fricative", True, note="not used in Standard Pakistani Urdu profile"),
]}

_ORAL_VOWELS = [
    ("ɪ", "front-high", False), ("iː", "front-high", True),
    ("ʊ", "back-high", False), ("uː", "back-high", True),
    ("ə", "central-mid", False), ("ɑ", "back-low", False), ("ɑː", "back-low", True),
    ("eː", "front-mid", True), ("ɛː", "front-lowmid", True),
    ("oː", "back-mid", True), ("ɔː", "back-lowmid", True),
    ("ɛ", "front-lowmid", False), ("ɔ", "back-lowmid", False),
]
VOWELS: dict[str, Seg] = {}
for _sym, _hb, _long in _ORAL_VOWELS:
    VOWELS[_sym] = _v(_sym, _hb, _long)
    _ns = (_sym[0] + NASAL + _sym[1:]) if _long else _sym + NASAL
    _ns = unicodedata.normalize("NFC", _ns)
    VOWELS[_ns] = _v(_ns, _hb, _long, nasal=True)

INVENTORY: dict[str, Seg] = {**CONSONANTS, **VOWELS}


def describe(seg: str) -> Seg | None:
    """Return features for a segment, handling geminate consonants (Cː)."""
    seg = tokenize(seg)[0] if seg else seg
    if seg in INVENTORY:
        return INVENTORY[seg]
    g = ungeminate(seg)
    if g != seg and g in CONSONANTS:
        return CONSONANTS[g]
    return None


def is_geminate(seg: str) -> bool:
    base = ungeminate(seg)
    return base != seg and base in CONSONANTS


def ungeminate(seg: str) -> str:
    """t͡ʃːʰ -> t͡ʃʰ ;  kː -> k ; vowels untouched."""
    if LENGTH not in seg:
        return seg
    cand = seg.replace(LENGTH, "")
    return cand if cand in CONSONANTS else seg


def geminate(seg: str) -> str:
    if seg not in CONSONANTS:
        return seg
    if seg.endswith(ASP) or seg.endswith(BREATHY_ASP):
        return seg[:-1] + LENGTH + seg[-1]
    return seg + LENGTH


def is_vowel(seg: str) -> bool:
    d = describe(seg)
    return bool(d and d.kind == "V")


def is_consonant(seg: str) -> bool:
    d = describe(seg)
    return bool(d and d.kind == "C")


def validate(ipa: str) -> list[str]:
    """Return the segments of `ipa` that are not in the Urdu inventory."""
    return [s for s in tokenize(ipa) if describe(s) is None]


# ---------------------------------------------------------------------------
# Contrast relations used for candidate generation and error analysis
# ---------------------------------------------------------------------------

ASPIRATION_PAIRS = [("p", "pʰ"), ("b", "bʱ"), ("t̪", "t̪ʰ"), ("d̪", "d̪ʱ"), ("ʈ", "ʈʰ"),
                    ("ɖ", "ɖʱ"), ("k", "kʰ"), ("ɡ", "ɡʱ"), ("t͡ʃ", "t͡ʃʰ"), ("d͡ʒ", "d͡ʒʱ"),
                    ("ɽ", "ɽʱ")]
RETROFLEX_PAIRS = [("t̪", "ʈ"), ("t̪ʰ", "ʈʰ"), ("d̪", "ɖ"), ("d̪ʱ", "ɖʱ"), ("n", "ɳ"),
                   ("r", "ɽ"), ("r", "ɽʱ")]
VOWEL_LENGTH_PAIRS = [("ɪ", "iː"), ("ʊ", "uː"), ("ə", "ɑː"), ("ɑ", "ɑː")]
VOWEL_QUALITY_PAIRS = [("eː", "ɛː"), ("oː", "ɔː"), ("iː", "eː"), ("uː", "oː")]
NASAL_PLACE_ALTERNATIONS = [("n", "ŋ"), ("n", "ɳ")]


def _pairmap(pairs):
    m: dict[str, set[str]] = {}
    for a, b in pairs:
        m.setdefault(a, set()).add(b)
        m.setdefault(b, set()).add(a)
    return m


_ASP = _pairmap(ASPIRATION_PAIRS)
_RET = _pairmap(RETROFLEX_PAIRS)
_VLEN = _pairmap(VOWEL_LENGTH_PAIRS)
_VQ = _pairmap(VOWEL_QUALITY_PAIRS)
_NPL = _pairmap(NASAL_PLACE_ALTERNATIONS)


def nasalize(v: str) -> str:
    if v not in VOWELS or VOWELS[v].nasal_vowel:
        return v
    if v.endswith(LENGTH):
        return unicodedata.normalize("NFC", v[:-1] + NASAL + LENGTH)
    return unicodedata.normalize("NFC", v + NASAL)


def denasalize(v: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", v).replace(NASAL, ""))


def contrast_flips(seg: str) -> dict[str, list[str]]:
    """Return alternatives to `seg` grouped by the contrast dimension they test."""
    out: dict[str, list[str]] = {}
    gem = is_geminate(seg)
    base = ungeminate(seg)
    rewrap = geminate if gem else (lambda s: s)
    if base in _ASP:
        out["aspiration"] = [rewrap(x) for x in sorted(_ASP[base])]
    if base in _RET:
        out["retroflexion"] = [rewrap(x) for x in sorted(_RET[base])]
    if base in _NPL:
        out["nasal_place"] = sorted(_NPL[base])
    if base in CONSONANTS and CONSONANTS[base].manner in ("stop", "affricate", "nasal", "lateral",
                                                          "fricative", "flap", "rhotic"):
        out["gemination"] = [base if gem else geminate(base)]
    if seg in VOWELS:
        oral = denasalize(seg)
        if oral in _VLEN:
            out["vowel_length"] = [nasalize(x) if VOWELS[seg].nasal_vowel else x for x in sorted(_VLEN[oral])]
        if oral in _VQ:
            out["vowel_quality"] = [nasalize(x) if VOWELS[seg].nasal_vowel else x for x in sorted(_VQ[oral])]
        out["nasalization"] = [oral if VOWELS[seg].nasal_vowel else nasalize(seg)]
    return out


def classify_difference(ref: str, hyp: str) -> str:
    """Name the phonological dimension on which two segments differ."""
    if ref == hyp:
        return "match"
    if not ref:
        return "insertion"
    if not hyp:
        return "deletion"
    if ungeminate(ref) == ungeminate(hyp) and is_geminate(ref) != is_geminate(hyp):
        return "gemination"
    r, h = ungeminate(ref), ungeminate(hyp)
    if h in _ASP.get(r, ()):
        return "aspiration"
    if h in _RET.get(r, ()):
        return "retroflexion"
    if r in VOWELS and h in VOWELS:
        if denasalize(r) == denasalize(h):
            return "nasalization"
        if denasalize(h) in _VLEN.get(denasalize(r), ()):
            return "vowel_length"
        return "vowel_quality"
    if h in _NPL.get(r, ()):
        return "nasal_place"
    return "other"
