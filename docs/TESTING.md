# Testing

## A. Automated tests (run after any change)

```
QT_QPA_PLATFORM=offscreen python -m pytest -q tests
```

73 tests, all passing at release:

| Area | What is tested |
|---|---|
| IPA | tokenisation (aspiration, dentals, tie bars, nasal vowels, modifier order), contrast relations, inventory validation of the seed lexicon |
| Orthography | normalisation of Arabic code points, ھ aspiration, tashdid, noon ghunna, ن + consonant nasalisation, ambiguous و |
| CTC | forward algorithm checked against brute-force path enumeration; alignment; empty candidates |
| Audio QC | single word, geminate-length gap kept inside one word, pause-separated words, empty, silence, too short, too long, clipped, noisy, resampling 44.1→16 kHz, aspiration noise kept by preprocessing |
| Measurement | VOT categories on synthetic stops, closure duration, no false burst at a vowel onset |
| Reconciliation | dictionary form confirmed, unaspirated realisation reported (not corrected), retroflex vs dental from audio, regional variants gated by profile, gemination flagged model-blind, missing stops refused, /h/ elision accepted, garbage audio refused |
| Pipeline | OK result, multi-word ASR rejected, pause-separated words rejected, spaced compound joined, non-Urdu script refused, homographs (پل pəl/pʊl) decided by audio, expected-word mismatch reported, model failure reported, concurrent jobs rejected |
| Storage / evaluation | manual corrections never overwritten, excluded rows left out of exports, CSV keeps Urdu and IPA, PER, dimension analysis, confusion counts, Excel report, manifest run with negative cases |
| UI (offscreen, fake microphone) | no recording at launch, duplicate start ignored, full cycle with copy/replay/save/correct, record next without restart, 8 consecutive recordings, permission error, device lost mid-recording, refusal message, stale result after timeout ignored |
| Real model | Allosaurus on 8 espeak-ng synthesised words: well-formed result or refusal, under 10 s |

What these tests do **not** show: accuracy on human speech.

## B. Research evaluation (required before trusting results)

1. Copy `tests/dataset/manifest_template.tsv` into a folder with your recordings.
2. Record each row (one speaker per folder, e.g. `S01/001.wav`). The template covers spec categories 1–9, 13 and 14; repeat with more speakers (11), speeds (10) and a moderately noisy room (12).
3. **Replace every `reference_ipa`** with your own verified transcription of that recording, not the dictionary form. The template values are provisional.
4. Rows with `expect = refuse` (silence, multi-word, clipped, incomplete) must be rejected.
5. Run:

```
python -m urdu_ipa.evaluation path/to/manifest.tsv --out report.xlsx
```

The report has sheets for summary (coverage, PER, exact match, word accuracy, negative tests), categories, dimensions (aspiration, retroflexion, vowel length, nasalisation, gemination), confusions, item details and suggested VOT boundaries.

Report PER **together with coverage**. A low PER at 20 % coverage means something different from the same PER at 90 %.

Corrections made in the app can be evaluated too:

```
python -m urdu_ipa.evaluation --from-db ~/UrduIPA --out corrections_report.xlsx
```

## C. Development smoke run (synthetic speech, not evidence of accuracy)

espeak-ng words through the full pipeline with Allosaurus: 12 of 13 refused, 1 produced (ماں as [mɑː], i.e. nasalisation not supported by the audio). Phone-model analysis took about 25–40 ms per word on CPU after a one-time ~0.7 s warm-up. Whisper time was not measurable in development.
