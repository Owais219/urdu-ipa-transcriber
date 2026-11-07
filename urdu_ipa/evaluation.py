"""Evaluation against human-verified IPA.

Metrics
* Phoneme error rate (PER) = (S + D + I) / N over IPA segments.
* Exact-match accuracy (whole transcription identical, segment-wise).
* Word recognition accuracy (orthographic, normalised spelling).
* Coverage = share of items for which the tool produced a transcription
  (refusals are reported separately, never counted as correct).
* Confusion counts and error analysis by dimension: aspiration,
  retroflexion, vowel length, vowel quality, nasalisation, gemination.

Usage
    python -m urdu_ipa.evaluation manifest.tsv --out report.xlsx
    python -m urdu_ipa.evaluation --from-db ~/UrduIPA --out report.xlsx

Manifest columns (TSV, header required):
    audio  word  reference_ipa  [expect  speaker  profile  category  condition  notes]
    expect = "transcribe" (default) or "refuse" (silence, multi-word, clipped ... must be rejected)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import ipa
from .orthography import lookup_key


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Levenshtein alignment; '' marks insertion/deletion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], "")); i -= 1
        else:
            out.append(("", hyp[j - 1])); j -= 1
    return out[::-1]


def per(ref_ipa: str, hyp_ipa: str) -> tuple[int, int, list[tuple[str, str]]]:
    r, h = ipa.tokenize(ref_ipa), ipa.tokenize(hyp_ipa)
    pairs = align(r, h)
    errors = sum(1 for a, b in pairs if a != b)
    return errors, len(r), pairs


@dataclass
class Item:
    ref_ipa: str
    hyp_ipa: str | None
    ref_word: str = ""
    hyp_word: str = ""
    meta: dict = field(default_factory=dict)


def evaluate(items: list[Item]) -> dict:
    produced = [it for it in items if it.hyp_ipa]
    err = tot = exact = 0
    conf = Counter()
    dims = Counter()
    dim_opps = Counter()
    per_cat = defaultdict(lambda: [0, 0, 0, 0])     # errors, ref segs, produced, items
    details = []
    for it in items:
        cat = it.meta.get("category", "")
        per_cat[cat][3] += 1
        row = dict(it.meta, reference=it.ref_ipa, hypothesis=it.hyp_ipa or "", ref_word=it.ref_word,
                   hyp_word=it.hyp_word,
                   word_correct=bool(it.ref_word and it.hyp_word and lookup_key(it.ref_word) == lookup_key(it.hyp_word)))
        if it.hyp_ipa:
            e, n, pairs = per(it.ref_ipa, it.hyp_ipa)
            err += e; tot += n
            exact += int(e == 0)
            per_cat[cat][0] += e; per_cat[cat][1] += n; per_cat[cat][2] += 1
            for a, b in pairs:
                if a and ipa.describe(a):
                    for dname in ipa.contrast_flips(a):
                        dim_opps[dname] += 1
                if a != b:
                    conf[(a or "∅", b or "∅")] += 1
                    dims[ipa.classify_difference(a, b)] += 1
            row.update(per=round(e / n, 4) if n else None, errors=e,
                       alignment=" ".join(f"{a or '∅'}>{b or '∅'}" for a, b in pairs if a != b))
        else:
            row.update(per=None, errors=None, alignment="(no transcription produced)")
        details.append(row)
    words_scored = [it for it in items if it.ref_word and it.hyp_word]
    summary = {
        "items": len(items),
        "transcribed": len(produced),
        "coverage": round(len(produced) / len(items), 4) if items else 0.0,
        "PER_on_transcribed": round(err / tot, 4) if tot else None,
        "exact_match_on_transcribed": round(exact / len(produced), 4) if produced else None,
        "exact_match_over_all_items": round(exact / len(items), 4) if items else None,
        "word_accuracy": round(sum(d["word_correct"] for d in details) / len(words_scored), 4) if words_scored else None,
    }
    dim_table = [{"dimension": k, "errors": v, "opportunities": dim_opps.get(k, ""),
                  "error_rate": round(v / dim_opps[k], 4) if dim_opps.get(k) else ""} for k, v in dims.most_common()]
    cat_table = [{"category": c or "(none)", "items": v[3], "transcribed": v[2],
                  "PER": round(v[0] / v[1], 4) if v[1] else None} for c, v in sorted(per_cat.items())]
    conf_table = [{"reference": a, "hypothesis": b, "count": n} for (a, b), n in conf.most_common()]
    return {"summary": summary, "details": details, "dimensions": dim_table, "categories": cat_table,
            "confusions": conf_table}


def write_report(report: dict, path: str | Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["metric", "value"])
    for k, v in report["summary"].items():
        ws.append([k, v])
    ws.append([])
    ws.append(["Note", "Refused items are excluded from PER and counted against coverage. "
                       "Confidence values are uncalibrated model scores."])
    for name in ("categories", "dimensions", "confusions", "details", "negative_tests"):
        if name not in report:
            continue
        rows = report[name]
        sh = wb.create_sheet(name)
        if not rows:
            sh.append(["(none)"])
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        sh.append(keys)
        for r in rows:
            sh.append([r.get(k) if not isinstance(r.get(k), (list, dict)) else str(r.get(k)) for k in keys])
    for sh in wb.worksheets:
        for c in sh[1]:
            c.font = Font(bold=True)
    wb.save(path)


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r.get("audio") and not r["audio"].startswith("#")]
    for r in rows:
        r["reference_ipa"] = (r.get("reference_ipa") or "").strip()
    return rows


def run_manifest(path: str | Path, pipeline, use_expected_word: bool = False):
    import soundfile as sf
    base = Path(path).parent
    items, results, negatives = [], [], []
    for r in read_manifest(path):
        x, sr = sf.read(str(base / r["audio"]), always_2d=False)
        if (r.get("expect") or "").strip().lower() == "refuse":
            res = pipeline.process(x, sr, profile=r.get("profile") or "standard")
            negatives.append({"audio": r["audio"], "category": r.get("category", ""), "status": res.status,
                              "problem": res.problem, "passed": res.status != "ok"})
            continue
        res = pipeline.process(x, sr, expected_word=r["word"] if use_expected_word else "",
                               profile=r.get("profile") or "standard")
        hyp = res.phonetic.strip("[]") if res.phonetic else None
        meta = {k: r.get(k, "") for k in ("audio", "speaker", "profile", "category", "condition")}
        meta.update(status=res.status, problem=res.problem, phonetic_confidence=res.phonetic_confidence,
                    processing_ms=res.processing_ms)
        items.append(Item(r["reference_ipa"], hyp, r["word"], res.recognized_word, meta))
        results.append(res)
    rep = evaluate(items)
    rep["negative_tests"] = negatives
    rep["summary"]["negative_tests"] = len(negatives)
    rep["summary"]["negative_tests_correctly_refused"] = sum(n["passed"] for n in negatives)
    return rep, results


def items_from_store(store) -> list[Item]:
    out = []
    for r in store.rows(include_excluded=False):
        if not r.get("manual_ipa"):
            continue
        hyp = (r.get("phonetic_ipa") or "").strip("[]") or None
        out.append(Item(r["manual_ipa"].strip("[]/"), hyp, r.get("manual_word") or r.get("analysed_word") or "",
                        r.get("recognized_word") or "",
                        {"id": r["id"], "speaker": r.get("speaker", ""), "profile": r.get("profile", ""),
                         "status": r.get("status")}))
    return out


def calibrate_vot(results, references: list[str]) -> dict:
    """Suggest VOT boundaries from verified data: midpoint between class means per place class."""
    groups = defaultdict(lambda: {"asp": [], "unasp": []})
    for res, ref in zip(results, references):
        hyp_segs = [s for s in res.segments if s["chosen"]]
        pairs = align(ipa.tokenize(ref), [s["chosen"] for s in hyp_segs])
        k = 0
        for a, b in pairs:
            if b:
                seg = hyp_segs[k]; k += 1
                vot = seg["measurement"].get("vot_ms")
                base = ipa.ungeminate(a) if a else ""
                info = ipa.describe(base) if base else None
                if vot is None or not info or info.voiced or info.manner not in ("stop", "affricate"):
                    continue
                cls = "affricate" if info.manner == "affricate" else ("back" if info.place in ("velar", "uvular") else "front")
                groups[cls]["asp" if info.aspirated else "unasp"].append(vot)
    out = {}
    for cls, g in groups.items():
        if g["asp"] and g["unasp"]:
            ma, mu = sum(g["asp"]) / len(g["asp"]), sum(g["unasp"]) / len(g["unasp"])
            mid = (ma + mu) / 2
            out[cls] = {"n_aspirated": len(g["asp"]), "n_unaspirated": len(g["unasp"]),
                        "mean_aspirated_ms": round(ma, 1), "mean_unaspirated_ms": round(mu, 1),
                        "suggested_boundary_ms": round(mid, 1)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate Urdu IPA transcription against verified references")
    ap.add_argument("manifest", nargs="?", help="TSV manifest (audio, word, reference_ipa, ...)")
    ap.add_argument("--from-db", help="data folder: evaluate stored recordings that have manual corrections")
    ap.add_argument("--out", default="evaluation_report.xlsx")
    ap.add_argument("--config", help="config.json")
    ap.add_argument("--use-expected-word", action="store_true",
                    help="give the manifest word to the pipeline (elicitation mode); word accuracy is then not meaningful")
    a = ap.parse_args(argv)
    if a.from_db:
        from .storage import Store
        rep = evaluate(items_from_store(Store(a.from_db)))
    elif a.manifest:
        from .config import load_config
        from .pipeline import Pipeline
        p = Pipeline(load_config(a.config))
        errs = p.load()
        if "acoustic" in errs:
            sys.exit(f"acoustic model unavailable: {errs['acoustic']}")
        rep, results = run_manifest(a.manifest, p, a.use_expected_word)
        refs = [r["reference_ipa"] for r in read_manifest(a.manifest)
                if (r.get("expect") or "").strip().lower() != "refuse"]
        cal = calibrate_vot(results, refs)
        if cal:
            rep["vot_calibration"] = [{"class": k, **v} for k, v in cal.items()]
    else:
        ap.error("give a manifest or --from-db")
    write_report({k: v for k, v in rep.items()}, a.out) if "vot_calibration" not in rep else _write_with_cal(rep, a.out)
    for k, v in rep["summary"].items():
        print(f"{k:32s} {v}")
    print(f"report written to {a.out}")


def _write_with_cal(rep, out):
    from openpyxl import load_workbook
    write_report(rep, out)
    wb = load_workbook(out)
    sh = wb.create_sheet("vot_calibration")
    rows = rep["vot_calibration"]
    sh.append(list(rows[0]))
    for r in rows:
        sh.append(list(r.values()))
    wb.save(out)


if __name__ == "__main__":
    main()
