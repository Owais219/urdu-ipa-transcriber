# Urdu Phonetic Transcription Tool

**Single-word speech-to-IPA transcription for linguistic and lexicographic research.**

Speak one Urdu word → the tool recognises the word (Urdu script), then decides its pronunciation **from the audio**, and shows a broad phonetic transcription `[…]`, the dictionary phonemic form `/…/`, separate word and phonetic confidence, and per-segment evidence. When the audio does not support a transcription, it says so instead of guessing.

> Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) first. No model currently available gives trustworthy acoustic IPA for Urdu, and nothing in this release has been evaluated on human speech. The tool is built to be evaluated (see [`docs/TESTING.md`](docs/TESTING.md)) and to refuse rather than invent.

---

## 1. How it works

```
microphone ─► QC & single-word check ─► Urdu ASR (word in Urdu script)
                                              │
            lexicon entry(ies) or spelling ◄──┘
                     │
                     ▼
   candidate lattice: expected segment + aspiration / retroflex / length /
   nasalisation / gemination / regional-variant / deletion alternatives
                     │
   acoustic phone model ─► CTC forced scoring of candidates (coordinate search)
                     │
   signal measurements: VOT (aspiration), closure duration (gemination)
                     │
                     ▼
   per-segment evidence ─► reliability gate ─► [phonetic]  /phonemic/  or refusal
```

| Stage | Module | Notes |
|---|---|---|
| Microphone | `recorder.py` | sounddevice; native device rate; one stream at a time; auto-stop after trailing silence or max duration; device loss detected |
| Preprocessing | `audio.py` | mono, 16 kHz (polyphase resampling), DC removal, 60 Hz high-pass. **No noise suppression** (it would remove aspiration and bursts) |
| QC / single word | `audio.py` | energy VAD with adaptive floor; pauses > 350 ms = multiple words; empty, silent, short, long, clipped, noisy checks |
| Word recognition | `asr.py` | Whisper (forced Urdu) or MMS; script check; multi-word output rejected |
| Candidates | `lexicon.py`, `orthography.py` | lexicon (homographs supported) or an orthography lattice when the word is unknown |
| Acoustic scoring | `acoustic.py` | Allosaurus frame posteriors → CTC forward/Viterbi over Urdu segments mapped to model phones; contrasts the model cannot see are detected automatically |
| Measurement | `measure.py` | VOT (burst → voicing onset), medial closure duration |
| Decision | `reconcile.py` | margins in nats; dictionary value kept unless the audio beats it by the margin; variants outside the profile need an extra margin |
| Output | `pipeline.py`, `ui.py` | confidence, warnings, evidence table, timing |
| Research data | `storage.py`, `evaluation.py` | SQLite + WAV; CSV/Excel; PER, confusion, dimension analysis |

Segment evidence statuses: **confirmed**, **acoustic variant** (realised differently from the dictionary, reported as such), **model-blind** (a contrast the model cannot test), **unresolved**, **not realised**.

## 2. Installation

### Easiest: the Mac app (Intel Macs, macOS 12+)

Download **UrduIPA-Intel.dmg** from the repository's *Releases* page, open it, drag *Urdu IPA* to Applications and follow *Read Me First.txt*. Everything, including both speech models, is inside the app; nothing is downloaded and no Terminal is needed. The app uses Whisper through CTranslate2 (int8), which is several times faster on Intel CPUs than the source version.

The DMG is built by `.github/workflows/build-mac-intel.yml` on GitHub's Intel Mac machines. That build runs the test suite, builds the app with PyInstaller (`packaging/UrduIPA.spec`), runs the packaged app's `--selftest`, and launches the app window before publishing the DMG.

### From source

### macOS (Intel and Apple Silicon)

1. Install **Python 3.12** from <https://www.python.org/downloads/macos/> (macOS 64-bit universal2 installer).
   On an Intel Mac, Python 3.13 or newer will not work: the newest PyTorch for Intel Macs is 2.2.2, which supports Python 3.8–3.12 only.
2. Open **Applications → Python 3.12** and double-click **Install Certificates.command** (without it, the phone-model download fails with an SSL error).
3. In Terminal:
   ```
   cd ~/Downloads/urdu-ipa-transcriber      # wherever you unzipped it
   bash setup_mac.command                    # one-time, 5–15 min
   ./run_mac.command                         # start the app
   ```
   On Intel Macs the setup script installs `requirements-mac-intel.txt` (torch 2.2.2, numpy < 2, transformers < 4.58). These pins were tested.
4. The first time you click **Start Recording**, macOS asks whether **Terminal** may use the microphone; allow it. If you declined, enable it under **System Settings → Privacy & Security → Microphone → Terminal**, then quit and reopen Terminal.
5. The first launch downloads the Whisper Urdu model (~1.6 GB). To work without it, run `./run_mac.command --asr manual` and type each expected word.

### Windows / Linux

Python 3.10–3.13.

```
cd urdu-ipa-transcriber
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build (smaller)
pip install -r requirements.txt
```

* **Windows / macOS**: the sounddevice wheel includes PortAudio.
* **Linux**: `sudo apt install libportaudio2`.
* **Fonts** (recommended): *Noto Nastaliq Urdu* (Urdu) and *Charis SIL* or *Doulos SIL* (IPA). The UI falls back to Noto Naskh Arabic / Noto Sans / Segoe UI.

## 3. Model setup

Models download on first use and are then cached for offline work.

| Model | Size | Licence | Downloaded from |
|---|---|---|---|
| Allosaurus `uni2005` (phone model) | ~41 MB | GPL-3.0 | GitHub releases, on first launch |
| `kingabzpro/whisper-large-v3-turbo-urdu` (default ASR) | ~1.6–3.2 GB | base Whisper is MIT; check the fine-tune's model card | Hugging Face |
| `facebook/mms-1b-all` + `urd-script_arabic` adapter (alternative ASR) | ~4 GB | CC-BY-NC 4.0 (non-commercial) | Hugging Face |

Pre-download (optional, needs internet once):

```
python -c "from allosaurus.app import read_recognizer; read_recognizer()"
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration as W; m='kingabzpro/whisper-large-v3-turbo-urdu'; WhisperProcessor.from_pretrained(m); W.from_pretrained(m)"
```

Then work offline: set `HF_HUB_OFFLINE=1`.

Whisper large on CPU takes seconds per word. For faster runs set `"asr_model": "openai/whisper-small"` (lower Urdu accuracy) or use a GPU (`"device": "cuda"`).

## 4. Running

```
python -m urdu_ipa                 # start the application
python -m urdu_ipa --asr manual    # no ASR model: type the expected word for each recording
```

1. (Optional) type the **expected word** — useful for elicitation lists. It is scored *alongside* the recogniser's output; if they differ you are told, never silently switched.
2. Choose the **profile** (Standard / Punjabi-influenced / Pashto-influenced).
3. Click **Start Recording** (or Ctrl+R), say one word, click **Stop** (recording also stops after 1 s of silence).
4. Read the result. Open **Show segment evidence** to see what supports each segment.
5. **Replay Audio**, **Copy IPA**, **Save to Dataset**. After saving, type a verified IPA and **Save Correction**; tick **Exclude from research dataset** for doubtful items (refusals are excluded automatically).
6. **Record Next Word** clears the result and starts the next recording.
7. **File → Export CSV / Excel** (excluded rows left out unless you choose otherwise).

Data folder: `~/UrduIPA` (recordings, `recordings.sqlite`, `lexicon.tsv`, `config.json`).

## 5. Configuration

Create `~/UrduIPA/config.json` with any of the keys from `urdu_ipa/config.py`, for example:

```json
{
  "asr_backend": "whisper",
  "asr_model": "kingabzpro/whisper-large-v3-turbo-urdu",
  "recording": {"max_duration": 6.0, "trailing_silence": 1.0, "speech_db_above_floor": 12.0},
  "vad": {"merge_gap": 0.35, "min_snr_db": 10.0},
  "reconcile": {"margin": 2.0, "max_unresolved_fraction": 0.25,
                "vot": {"stop_front": [0.035, 0.055], "stop_back": [0.045, 0.065], "affricate": [0.075, 0.11]}}
}
```

## 6. Adding words and pronunciation references

The lexicon is `~/UrduIPA/lexicon.tsv` (copied from `data/lexicon.tsv` on first run). Tab-separated:

```
word	phonemic	citation_phonetic	gloss	source	verified	notes
پتّا	pət̪ːɑː		leaf	MO-2026	yes	geminate
ٹھنڈا	ʈʰənɖɑː	ʈʰəɳɖɑː	cold	MO-2026	yes	nasal place assimilation
```

* `phonemic` without slashes. `citation_phonetic` only when the expected broad phonetic form differs.
* Several rows with the same spelling = homographs; the audio decides between them.
* Symbols must be in the inventory ([`docs/TRANSCRIPTION_STANDARD.md`](docs/TRANSCRIPTION_STANDARD.md)); problems are reported at load.
* Set `verified` to `yes` only after checking. Unverified entries trigger a warning on every result.
* In the app: **Tools → Add Word to Lexicon…** (entries are appended, never rewritten).

## 7. Evaluation

See [`docs/TESTING.md`](docs/TESTING.md). In short: record the items in `tests/dataset/manifest_template.tsv`, replace the provisional references with your verified IPA, then

```
python -m urdu_ipa.evaluation manifest.tsv --out report.xlsx
```

## 8. Extending the system

* **Better phone model (largest gain).** Fine-tune a wav2vec2/XLS-R CTC model on Urdu speech with *hand-verified* IPA labels (not espeak output), then set `"acoustic_backend": "hf-phoneme", "acoustic_model": "<your-model>"`. Proxies are matched to its vocabulary automatically, so contrasts it can represent stop being model-blind. Allosaurus can also be fine-tuned (`python -m allosaurus.bin.adapt_model`).
* **Calibrate** VOT and closure thresholds from the `vot_calibration` sheet of an evaluation report.
* **New profile**: add an entry to `data/profiles.json` with `expected_variants` pairs.
* **New measurements** (e.g. breathy voice via H1–H2 for /bʱ d̪ʱ ɖʱ/, nasalisation via A1–P0): add a function to `measure.py` and apply it in `Reconciler._apply_measurements` for the matching dimension.
* **Confidence calibration**: once you have a verified evaluation set, fit a mapping from the confidence scores to observed accuracy before reporting them as probabilities.

## 9. Project layout

```
urdu-ipa-transcriber/
├── urdu_ipa/
│   ├── __main__.py        entry point (python -m urdu_ipa)
│   ├── ui.py              PySide6 interface
│   ├── recorder.py        microphone and playback
│   ├── audio.py           preprocessing, VAD, QC
│   ├── asr.py             Whisper / MMS / manual word recognition
│   ├── orthography.py     Urdu normalisation, spelling lattices
│   ├── lexicon.py         pronunciation lexicon
│   ├── ipa.py             inventory, tokeniser, contrasts
│   ├── acoustic.py        phone-model backends, CTC scoring
│   ├── measure.py         VOT and closure measurement
│   ├── reconcile.py       candidate search and evidence
│   ├── pipeline.py        end-to-end processing, messages
│   ├── storage.py         SQLite, WAV, CSV/Excel export
│   ├── evaluation.py      PER, confusions, reports (CLI)
│   └── config.py          settings
├── data/  lexicon.tsv  profiles.json
├── docs/  TRANSCRIPTION_STANDARD.md  LIMITATIONS.md  TESTING.md
├── tests/ (unit, pipeline, UI, real-model tests; dataset/manifest_template.tsv)
└── requirements.txt
```

Licence note: Allosaurus is GPL-3.0. If you distribute this tool together with it, the combined work must comply with the GPL.
