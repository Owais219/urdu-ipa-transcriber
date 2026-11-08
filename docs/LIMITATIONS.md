# Known limitations (read before using results as research data)

## 1. No available model gives reliable acoustic IPA for Urdu

Checked during development (November 2026):

| Model | What it does | Why it cannot be trusted on its own |
|---|---|---|
| Allosaurus `uni2005` (GPL-3.0) | Universal phone recogniser, offline | 14 training corpora, **none South Asian**. Input is 8 kHz MFCC (40–3800 Hz), so high-frequency burst and frication cues are lost. 30 ms output frames. Its built-in Urdu inventory lacks /t͡ʃ/, /ɡʱ/, /ɳ/ and contains odd entries (ʂ, ʃʰ). It has no phone for [ʈʰ], [t͡ʃʰ], [ɽʱ], breathy dentals, or most long nasal vowels. Free decoding on synthetic Urdu words was essentially wrong. |
| espeak-ng Urdu, and models trained on espeak labels (e.g. aurix-v1) | Spelling → IPA rules | Not acoustic. espeak's Urdu output does not even mark dentals (gave [t], [d] for ت, د). A model trained to reproduce it learns spelling rules, not pronunciation. |
| Whisper (incl. Urdu fine-tunes), MMS | Speech → Urdu script | Orthographic only. Zero-shot Whisper is reported to output the wrong script on Urdu; this tool forces Urdu decoding and rejects non-Urdu-script output. Reported WERs are for sentences; single-word accuracy is unknown and must be measured. |

**Consequence.** The tool does not free-decode IPA. It scores the candidate pronunciations allowed by the lexicon or spelling, plus every relevant contrast flip, against the audio, and refuses when the evidence is weak. Expect a high refusal rate with the default universal phone model. That is the intended, honest behaviour.

## 2. Contrasts the default phone model cannot test

Computed automatically from the model vocabulary (`AcousticBackend.discriminable`):

* /ʈ/~/ʈʰ/ and /t͡ʃ/~/t͡ʃʰ/ — resolved by **VOT measurement** when the burst and voicing onset are found.
* /d̪/~/d̪ʱ/, /ɖ/~/ɖʱ/, /ɽ/~/ɽʱ/ — no measurement implemented; reported as `model_blind`.
* Nasalisation of /ɪ iː ʊ uː eː oː/ — reported as `model_blind`.
* All gemination — resolved by **closure duration** for word-medial stops and affricates only; other geminates are `model_blind`.

A transcription with `model_blind` segments can still be produced. Check the segment evidence table before using it.

## 3. Measurement thresholds are not calibrated

VOT boundaries (short-lag vs long-lag) and singleton/geminate closure boundaries are generic defaults from the Hindi-Urdu literature range. They were checked only on synthetic signals. Run the evaluation on verified recordings; the report includes a `vot_calibration` sheet with suggested boundaries for your speakers. Put them in `config.json` under `reconcile.vot`.

The final-ہ breath thresholds (50 ms present / 20 ms absent) are also uncalibrated. Room noise or a
recording stopped too soon after the word can make the end unmeasurable; the tool then says so.
The phone model cannot separate short [ɑ] from long [ɑː], so that length contrast is reported as model-blind.

## 3b. Evidence standard (Tools → Strict Evidence Standard)

* **Standard (default).** For a word in the lexicon, a dictionary value that the recording favours, but not
  by the full margin, is accepted and marked **consistent with recording (not confirmed)**. A transcription is
  given when no segment is contradicted by the audio. "Phonetic confidence" counts only segments the
  recording *confirms*.
* **Strict.** Every tested contrast must be confirmed by the full margin, and the phonetic confidence must
  reach 50 %. With the current universal phone model this refuses most words.
* Each saved record stores the evidence standard used (`evidence_standard` column).
* Development check on espeak-ng synthetic speech (not evidence of accuracy): standard transcribed 4 of 10
  test words, strict 1 of 10.

## 3c. Lexicon sources

* `data/lexicon.tsv`: 71 hand-entered entries (unverified seed), highest built-in priority.
* `data/lexicon_wiktionary.tsv`: 7,200 pronunciations for 6,528 words from English Wiktionary
  (CC BY-SA 4.0, via kaikki.org/Wiktextract), converted to this project's conventions by
  `packaging/build_wiktionary_lexicon.py`. Not individually verified; Wiktionary's "Standard Urdu"
  transcriptions follow its Hindustani pronunciation module.
* `~/UrduIPA/my_lexicon.tsv`: your own entries; they replace built-in entries for the same spelling.
* When a word is refused, its dictionary pronunciation(s) are still shown, labelled "not confirmed by this
  recording", ordered by how well each matches the recording.

## 4. Confidence values are not probabilities

* *Word recognition confidence* is the recogniser's mean token probability (Whisper) or mean frame posterior (MMS).
* *Phonetic confidence* is the share of tested contrasts with acoustic support.

Neither has been calibrated against an evaluation set. Do not report them as probabilities of correctness.

## 5. Single-word detection

* A pause longer than 350 ms splits the recording into two "words". This is longer than Urdu geminate closures, so it will not catch two words spoken without a pause.
* The recogniser's output is also checked: more than one written word is rejected, except a spaced compound found in the lexicon (e.g. خوب صورت).

## 6. Other limits

* Words not in the lexicon: short vowels and ambiguous و/ی/ہ are decided from the audio alone, so these are refused much more often. No phonemic form is shown.
* No epenthesis candidates (e.g. Punjabi-influenced [ɪskuːl]) and no metathesis.
* Regional profiles list reported tendencies; they have not been validated with speaker data.
* The seed lexicon (70 entries) is marked unverified. Have every entry checked.
* Development tests used synthetic and synthesised audio only. Nothing in this release has been evaluated on human Urdu speech. **No accuracy figure is claimed.**
* Whisper and MMS could not be downloaded in the development environment. Their code paths were exercised only against a randomly initialised Whisper (transformers 5.22).
