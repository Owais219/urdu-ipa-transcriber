import itertools

import numpy as np
import pytest

from urdu_ipa import ipa
from urdu_ipa.acoustic import ctc_align, ctc_logprob, Emissions, log_softmax
from urdu_ipa.lexicon import Lexicon
from urdu_ipa.config import PROJECT_DATA
from urdu_ipa.orthography import lattice_from_spelling, lookup_key, normalize, script_fraction
from tests.helpers import FakeAcoustic


# ---------------- IPA ----------------
def test_tokenize_keeps_aspiration_dental_and_ties():
    assert ipa.tokenize("[ˈt͡ʃʰə.t̪]") == ["t͡ʃʰ", "ə", "t̪"]
    assert ipa.tokenize("pʰəl") == ["pʰ", "ə", "l"]
    assert ipa.tokenize("d̪ʱuːp") == ["d̪ʱ", "uː", "p"]


def test_tokenize_canonical_modifier_order():
    assert ipa.tokenize("t͡ʃʰː") == ipa.tokenize("t͡ʃːʰ") == ["t͡ʃːʰ"]


def test_nasal_vowels_are_single_segments():
    assert ipa.tokenize("mɑ̃ː") == ["m", "ɑ̃ː"]
    assert ipa.describe("ɑ̃ː").nasal_vowel


def test_geminate_helpers():
    assert ipa.geminate("kʰ") == "kːʰ" and ipa.ungeminate("kːʰ") == "kʰ"
    assert ipa.is_geminate("t̪ː") and not ipa.is_geminate("iː")


@pytest.mark.parametrize("a,b,dim", [("t̪", "ʈ", "retroflexion"), ("p", "pʰ", "aspiration"),
                                     ("ɪ", "iː", "vowel_length"), ("ɑː", "ɑ̃ː", "nasalization"),
                                     ("kː", "k", "gemination"), ("eː", "ɛː", "vowel_quality"),
                                     ("d̪ʱ", "ɖʱ", "retroflexion")])
def test_classify_difference(a, b, dim):
    assert ipa.classify_difference(a, b) == dim


def test_contrast_flips_cover_core_contrasts():
    f = ipa.contrast_flips("t̪")
    assert f["aspiration"] == ["t̪ʰ"] and f["retroflexion"] == ["ʈ"] and f["gemination"] == ["t̪ː"]
    assert "nasalization" in ipa.contrast_flips("ɑː")


def test_seed_lexicon_uses_only_inventory_symbols():
    lex = Lexicon(PROJECT_DATA / "lexicon.tsv")
    assert lex.problems == []
    assert len(lex.lookup("پل")) == 2                  # homographs kept
    assert lex.lookup("ٹھنڈا")[0].expected_phonetic == "ʈʰəɳɖɑː"


# ---------------- orthography ----------------
def test_normalisation_maps_arabic_code_points():
    assert normalize("كتاب") == "کتاب"
    assert lookup_key("بچّہ") == lookup_key("بچہ")
    assert script_fraction("کتاب") == 1.0 and script_fraction("kitab") == 0.0


def test_spelling_lattice_marks_aspiration_and_unwritten_vowels():
    slots = lattice_from_spelling("پھل")
    assert slots[0].options == ["pʰ"]
    assert slots[1].canonical is None and "ə" in slots[1].options and "" in slots[1].options
    assert slots[-1].options == ["l"]


def test_spelling_lattice_tashdid_and_noon_ghunna():
    assert lattice_from_spelling("بچّہ")[2].options == ["t͡ʃː"]
    assert lattice_from_spelling("ماں")[-1].options == ["ɑ̃ː"]


def test_spelling_lattice_ambiguous_waw():
    slots = lattice_from_spelling("کون")
    assert set(slots[1].options) >= {"uː", "oː", "ɔː", "ʋ"} and slots[1].canonical is None


# ---------------- CTC ----------------
def _brute_force_ctc(logp, labels, blank=0):
    """Sum over every frame-level path that collapses to `labels`."""
    T, V = logp.shape
    total = -np.inf
    for path in itertools.product(range(V), repeat=T):
        out, prev = [], None
        for p in path:
            if p != prev and p != blank:
                out.append(p)
            prev = p
        if out == labels:
            total = np.logaddexp(total, sum(logp[t, p] for t, p in enumerate(path)))
    return total


def test_ctc_forward_matches_brute_force():
    rng = np.random.default_rng(3)
    be = FakeAcoustic()
    voc = {"<blk>": 0, "p": 1, "ə": 2, "l": 3}
    be._vocab = voc
    be._proxy_cache.clear()
    logp = log_softmax(rng.standard_normal((5, 4)))
    em = Emissions(logp, 0.03, 0, voc)
    for seq in (["p", "ə", "l"], ["p", "l"], ["ə"]):
        assert ctc_logprob(em, seq, be) == pytest.approx(_brute_force_ctc(logp, [voc[s] for s in seq]), abs=1e-6)


def test_ctc_prefers_true_sequence_and_aligns():
    be = FakeAcoustic(["pʰ", "ə", "l"])
    em = be.emissions(None)
    assert ctc_logprob(em, ["pʰ", "ə", "l"], be) > ctc_logprob(em, ["p", "ə", "l"], be) + 5
    score, spans = ctc_align(em, ["pʰ", "ə", "l"], be)
    assert [s[0] for s in spans] == sorted(s[0] for s in spans) and all(s[0] >= 0 for s in spans)


def test_empty_sequence_is_scored_without_error():
    be = FakeAcoustic(["ə"])
    em = be.emissions(None)
    assert np.isfinite(ctc_logprob(em, [], be))
    assert np.isfinite(ctc_align(em, [], be)[0])


def test_model_blind_contrasts_detected_for_allosaurus_style_vocab():
    be = FakeAcoustic()
    be._vocab = {k: v for k, v in be._vocab.items() if k not in ("ʈʰ",)}
    be._proxy_cache.clear()
    assert not be.discriminable("ʈ", "ʈʰ")
    assert be.discriminable("p", "pʰ")
