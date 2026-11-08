"""Reconcile candidate pronunciations with acoustic evidence.

For a recognised word we build a lattice of slots (from a lexicon entry or
from the spelling). Each slot lists the expected value plus the alternatives
that test each Urdu contrast (aspiration, retroflexion, vowel length,
nasalisation, gemination, regional variants, presence/absence).

The acoustic model scores whole candidate sequences (CTC). We search the
lattice by coordinate ascent, then measure for every slot how much worse each
alternative scores than the chosen value (the *margin*, in nats). Signal
measurements (VOT, closure duration) resolve contrasts the model cannot see.

Every slot ends with an explicit evidence status:
  confirmed         every tested contrast is supported by acoustic evidence
  acoustic_variant  the audio supports a value different from the dictionary
                    form (reported, never "corrected")
  model_blind       present, but at least one contrast cannot be tested by
                    the model and no measurement resolved it
  unresolved        the evidence does not separate the alternatives
  not_realised      the audio supports the segment being absent
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import ipa
from .acoustic import AcousticBackend, Emissions, ctc_align, ctc_logprob, free_decode, subphone_preference, NEG_INF
from .measure import (FinalBreathThresholds, GeminationThresholds, VOTThresholds, gemination_category,
                      measure_closure, measure_final_breath, measure_vot)
from .orthography import FINAL_H, Slot

VOICELESS_STOPS = {"p", "t̪", "ʈ", "k", "q", "t͡ʃ", "pʰ", "t̪ʰ", "ʈʰ", "kʰ", "t͡ʃʰ"}
ELIDABLE = {"h", "ɦ", "ʔ", "j"}
STOPS_AND_AFFRICATES = VOICELESS_STOPS |{"b", "bʱ", "d̪", "d̪ʱ", "ɖ", "ɖʱ", "ɡ", "ɡʱ", "d͡ʒ", "d͡ʒʱ"}


@dataclass
class ReconcileConfig:
    margin: float = 2.0                 # nats an alternative must lose by to count as rejected
    variant_penalty: float = 2.0        # extra margin for variants not expected in the profile
    max_fit_gap: float = 1.2            # nats/frame between forced and free best path
    max_unresolved_fraction: float = 0.25
    min_confidence: float = 0.5
    narrow_margin: float = 1.0
    # "standard": a dictionary value the audio favours (margin >= 0) but not by `margin` counts as
    #             CONSISTENT; a transcription is given when no segment is contradicted.
    # "strict":   every tested contrast must be confirmed by `margin` (and min_confidence applies).
    evidence_standard: str = "standard"
    no_information: float = 0.05        # |margin| below this = the model gives no information (blind)
    vot: VOTThresholds = field(default_factory=VOTThresholds)
    gemination: GeminationThresholds = field(default_factory=GeminationThresholds)
    final_breath: FinalBreathThresholds = field(default_factory=FinalBreathThresholds)


@dataclass
class DimEvidence:
    dimension: str
    alternative: str
    margin: float | None        # None when the model cannot discriminate
    status: str                 # supported | weak | blind | measured | conflict


@dataclass
class SegmentEvidence:
    index: int
    expected: str | None
    chosen: str
    source: str
    start: float | None
    end: float | None
    status: str
    dims: list[DimEvidence] = field(default_factory=list)
    measurement: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Reconciliation:
    candidate_label: str
    phonetic: str | None
    narrow: str | None
    segments: list[SegmentEvidence]
    score: float
    fit_gap: float
    confidence: float
    level: str
    reliable: bool
    reasons: list[str]
    free_decode: str

    def to_dict(self):
        return asdict(self)


def load_profiles(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def all_variant_pairs(profiles: dict) -> list[tuple[str, str]]:
    seen = []
    for p in profiles.values():
        for a, b in p.get("expected_variants", []):
            if (a, b) not in seen:
                seen.append((a, b))
    return seen


def slots_from_phonemes(segments: list[str], variant_pairs: list[tuple[str, str]]) -> list[Slot]:
    slots = []
    for seg in segments:
        opts, dims = [seg], {}
        for dim, alts in ipa.contrast_flips(seg).items():
            for a in alts:
                if a not in opts:
                    opts.append(a)
                    dims[a] = dim
        base = ipa.ungeminate(seg)
        for a, b in variant_pairs:
            if a == base:
                v = ipa.geminate(b) if ipa.is_geminate(seg) else b
                if v not in opts:
                    opts.append(v)
                    dims[v] = "variant"
        opts.append("")
        dims[""] = "presence"
        slots.append(Slot(opts, 0, "lexical", seg, dims))
    return slots


def enrich_spelling_slots(slots: list[Slot], variant_pairs) -> list[Slot]:
    out = []
    for s in slots:
        s = Slot(list(s.options), s.canonical, s.kind, s.source, dict(s.dimensions))
        if s.kind == FINAL_H:
            out.append(s)
            continue
        for o in list(s.options):
            if o and o not in s.dimensions:
                s.dimensions[o] = "spelling"
        if s.canonical is not None and ipa.is_consonant(s.options[s.canonical]):
            seg = s.options[s.canonical]
            for dim, alts in ipa.contrast_flips(seg).items():
                if dim in ("aspiration", "nasal_place", "gemination"):
                    for a in alts:
                        if a not in s.options:
                            s.options.append(a)
                            s.dimensions[a] = dim
            for a, b in variant_pairs:
                if a == ipa.ungeminate(seg) and b not in s.options:
                    s.options.append(b)
                    s.dimensions[b] = "variant"
            if "" not in s.options:
                s.options.append("")
                s.dimensions[""] = "presence"
        out.append(s)
    return out


class Reconciler:
    def __init__(self, backend: AcousticBackend, cfg: ReconcileConfig, profile: dict):
        self.b = backend
        self.cfg = cfg
        self.expected_variants = {tuple(v) for v in profile.get("expected_variants", [])}

    # -- scoring helpers -----------------------------------------------------
    def _penalty(self, slot: Slot, opt: str) -> float:
        if slot.dimensions.get(opt) != "variant" or slot.canonical is None:
            return 0.0
        canon = ipa.ungeminate(slot.options[slot.canonical])
        return 0.0 if (canon, ipa.ungeminate(opt)) in self.expected_variants else self.cfg.variant_penalty

    def run(self, em: Emissions, slots: list[Slot], audio: np.ndarray, sr: int, label: str) -> Reconciliation:
        cache: dict[tuple, float] = {}

        def score(choice):
            seq = tuple(slots[i].options[c] for i, c in enumerate(choice))
            if seq not in cache:
                cache[seq] = ctc_logprob(em, list(seq), self.b)
            return cache[seq]

        choice = []
        for s in slots:
            if s.canonical is not None:
                choice.append(s.canonical)
            else:
                choice.append(next((k for k, o in enumerate(s.options) if o), 0))

        # coordinate ascent
        for _ in range(4):
            changed = False
            for i, s in enumerate(slots):
                best_k, best_v = choice[i], score(choice) - self._penalty(s, s.options[choice[i]])
                for k in range(len(s.options)):
                    if k == choice[i]:
                        continue
                    trial = choice[:i] + [k] + choice[i + 1:]
                    v = score(trial) - self._penalty(s, s.options[k])
                    if v > best_v + 1e-9:
                        best_k, best_v = k, v
                if best_k != choice[i]:
                    choice[i] = best_k
                    changed = True
            if not changed:
                break

        # conservative: leave the dictionary value only with a clear margin
        reverted = set()
        for i, s in enumerate(slots):
            if s.canonical is not None and choice[i] != s.canonical:
                canon_trial = choice[:i] + [s.canonical] + choice[i + 1:]
                need = self.cfg.margin + self._penalty(s, s.options[choice[i]])
                if score(choice) - score(canon_trial) < need:
                    choice[i] = s.canonical
                    reverted.add(i)

        final_seq = [slots[i].options[c] for i, c in enumerate(choice)]
        best_score = score(choice)
        vit_score, spans = ctc_align(em, final_seq, self.b)
        symbols = list(ipa.INVENTORY)
        free, free_score = free_decode(em, self.b, symbols)
        fit_gap = (free_score - vit_score) / max(1, em.T) if vit_score > NEG_INF / 2 else float("inf")

        # evidence per slot
        evidence: list[SegmentEvidence] = []
        nonempty = [i for i, c in enumerate(choice) if slots[i].options[c]]
        span_of = {i: spans[k] for k, i in enumerate(nonempty)} if spans else {}
        for i, s in enumerate(slots):
            chosen = s.options[choice[i]]
            expected = s.options[s.canonical] if s.canonical is not None else None
            dims: dict[str, DimEvidence] = {}
            sc_chosen = score(choice)
            for k, o in enumerate(s.options):
                if k == choice[i]:
                    continue
                dim = s.dimensions.get(o, "spelling")
                if not self.b.discriminable(chosen, o):
                    ev = DimEvidence(dim, o, None, "blind")
                else:
                    trial = choice[:i] + [k] + choice[i + 1:]
                    m = sc_chosen - score(trial)
                    need = self.cfg.margin + (self._penalty(s, o) if o != expected else 0.0)
                    if m >= need:
                        status = "supported"
                    elif abs(m) < self.cfg.no_information:
                        status = "blind"
                    elif (self.cfg.evidence_standard == "standard" and expected is not None
                          and chosen == expected and m > 0):
                        status = "consistent"          # audio favours the dictionary value, below the margin
                    else:
                        status = "weak"
                    ev = DimEvidence(dim, o, round(m, 3), status)
                cur = dims.get(dim)
                # keep the weakest evidence per dimension
                rank = {"weak": 0, "blind": 1, "consistent": 2, "supported": 3}
                if cur is None or rank[ev.status] < rank[cur.status] or (
                        ev.status == cur.status and ev.margin is not None and cur.margin is not None and ev.margin < cur.margin):
                    dims[dim] = ev
            fr = span_of.get(i)
            start = end = None
            if fr and fr[0] >= 0:
                start, end = fr[0] * em.frame_shift, (fr[1] + 1) * em.frame_shift
            seg = SegmentEvidence(i, expected, chosen, s.kind, start, end, "", list(dims.values()))
            if i in reverted:
                seg.note = "audio leaned towards an alternative, below the evidence margin"
            evidence.append(seg)

        self._apply_measurements(evidence, slots, choice, audio, sr)

        # statuses
        for seg in evidence:
            sts = {d.status for d in seg.dims}
            if seg.chosen == "":
                seg.status = "not_realised" if "weak" not in sts else "unresolved"
            elif "weak" in sts or "conflict" in sts:
                seg.status = "unresolved"
            elif seg.expected is not None and seg.chosen != seg.expected:
                seg.status = "acoustic_variant"
            elif "consistent" in sts:
                seg.status = "consistent"
            elif "blind" in sts:
                seg.status = "model_blind"
            else:
                seg.status = "confirmed"

        # empty slots that are optional (unwritten short vowels judged absent) are not "segments"
        scored = [e for e in evidence if not (e.chosen == "" and e.source in ("optional_vowel", FINAL_H))]
        dims_all = [d for e in scored for d in e.dims]
        good = sum(1 for d in dims_all if d.status in ("supported", "measured"))
        confidence = good / len(dims_all) if dims_all else 0.0
        level = "high" if confidence >= 0.8 else ("medium" if confidence >= 0.6 else "low")
        reasons = []
        # Elision in careful single-word citation is plausible only for weak segments; a missing
        # stop or vowel is far more likely a phone-model miss than a real deletion.
        missed = [e for e in scored if e.status == "not_realised" and e.source == "lexical"
                  and ipa.ungeminate(e.expected or "") not in ELIDABLE]
        if missed:
            reasons.append("segment_not_detected")
            for e in missed:
                e.status = "unresolved"
                e.note = (e.note + "; " if e.note else "") + "not detected by the phone model (elision implausible here)"
        unresolved = sum(1 for e in scored if e.status == "unresolved")
        if not np.isfinite(fit_gap) or fit_gap > self.cfg.max_fit_gap:
            reasons.append("acoustic_mismatch")
        if scored and unresolved / len(scored) > self.cfg.max_unresolved_fraction:
            reasons.append("too_many_unresolved")
        if self.cfg.evidence_standard == "strict" and confidence < self.cfg.min_confidence:
            reasons.append("low_phonetic_confidence")
        reliable = not reasons

        final = [e.chosen for e in evidence if e.chosen]
        phonetic = ipa.join(final) if reliable else None
        narrow = None
        if reliable and all(e.status in ("confirmed", "acoustic_variant", "not_realised") for e in scored):
            narrow = self._narrow(em, evidence, span_of)
        return Reconciliation(label, phonetic, narrow, evidence, round(best_score, 3),
                              round(fit_gap, 3) if np.isfinite(fit_gap) else float("inf"),
                              round(confidence, 3), level, reliable, reasons, ipa.join(free))

    # -- measurements ----------------------------------------------------------
    def _apply_measurements(self, evidence, slots, choice, audio, sr):
        self._measure_final_h(evidence, audio, sr)
        for seg in evidence:
            base = ipa.ungeminate(seg.chosen)
            if base not in STOPS_AND_AFFRICATES or seg.start is None:
                continue
            info = ipa.describe(base)
            kind = "affricate" if info.manner == "affricate" else ("back" if info.place in ("velar", "uvular") else "front")
            r = measure_vot(audio, sr, seg.start, kind, self.cfg.vot)
            if r is None:
                seg.measurement["vot"] = None
                continue
            seg.measurement["vot_ms"] = round(r.vot * 1000, 1)
            seg.measurement["burst_s"] = round(r.burst_time, 3)
            asp_dim = next((d for d in seg.dims if d.dimension == "aspiration"), None)
            if base in VOICELESS_STOPS and asp_dim is not None:
                seg.measurement["vot_category"] = r.category + " (uncalibrated thresholds)"
                says_asp = {"aspirated": True, "unaspirated": False}.get(r.category)
                is_asp = base.endswith(ipa.ASP)
                if says_asp is None:
                    pass
                elif asp_dim.status in ("blind", "weak"):
                    if says_asp == is_asp:
                        asp_dim.status = "measured"
                    else:
                        flipped = asp_dim.alternative
                        asp_dim.alternative, asp_dim.status = seg.chosen, "measured"
                        seg.chosen = flipped
                        seg.note = (seg.note + "; " if seg.note else "") + "aspiration decided by VOT"
                elif asp_dim.status == "supported" and says_asp != is_asp:
                    asp_dim.status = "conflict"
                    seg.note = (seg.note + "; " if seg.note else "") + "phone model and VOT disagree on aspiration"
            gem_dim = next((d for d in seg.dims if d.dimension == "gemination"), None)
            if gem_dim is not None and seg.index > 0 and gem_dim.status in ("blind", "weak"):
                clo = measure_closure(audio, sr, r.burst_time)
                cat = gemination_category(clo, self.cfg.gemination)
                if clo is not None:
                    seg.measurement["closure_ms"] = round(clo * 1000, 1)
                if cat in ("singleton", "geminate"):
                    says_gem = cat == "geminate"
                    if says_gem == ipa.is_geminate(seg.chosen):
                        gem_dim.status = "measured"
                    else:
                        flipped = gem_dim.alternative
                        gem_dim.alternative, gem_dim.status = seg.chosen, "measured"
                        seg.chosen = flipped
                        seg.note = (seg.note + "; " if seg.note else "") + "length decided by closure duration"

    def _measure_final_h(self, evidence, audio, sr):
        """Word-final ہ: decide final breathy [h] from the signal, not from the phone model."""
        if not evidence or evidence[-1].source != FINAL_H:
            return
        seg = evidence[-1]
        r = measure_final_breath(audio, sr, self.cfg.final_breath)
        dim = next((d for d in seg.dims if d.dimension == "final_aspiration"), None)
        if dim is None:
            return
        if r is None:
            seg.measurement["final_breath"] = None
            seg.note = (seg.note + "; " if seg.note else "") + "end of word not measurable"
            return
        seg.measurement["final_breath_ms"] = round(r.breath_duration * 1000, 1)
        seg.measurement["final_breath"] = r.category + " (uncalibrated thresholds)"
        if r.category == "ambiguous":
            dim.status = "weak"
            return
        want = "h" if r.category == "present" else ""
        if seg.chosen != want:
            dim.alternative, seg.chosen = seg.chosen, want
            seg.note = (seg.note + "; " if seg.note else "") + "final [h] decided by breath-noise duration"
        dim.status = "measured"

    def _narrow(self, em, evidence, span_of) -> str:
        out = []
        for e in evidence:
            s = e.chosen
            if not s:
                continue
            fr = span_of.get(e.index)
            if fr and s == "h" and subphone_preference(em, fr, "ɦ", "h") > self.cfg.narrow_margin:
                s = "ɦ"
            elif fr and s == "r" and subphone_preference(em, fr, "ɾ", "r") > self.cfg.narrow_margin:
                s = "ɾ"
            out.append(s)
        return ipa.join(out)
